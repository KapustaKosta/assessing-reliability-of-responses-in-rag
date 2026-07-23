"""
Training script for token-level hallucination classifier.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from token_classifier.config import TokenClassifierConfig, get_model_path
from token_classifier.dataset import load_data, create_dataloaders
from token_classifier.labeling import AnswerTokenizer
from token_classifier.model import (
    TokenHallucinationClassifier,
    load_tokenizer_and_model,
    get_device,
    compute_grad_norm,
)
from token_classifier.metrics import (
    compute_token_metrics,
    compute_span_metrics,
    compute_answer_metrics,
    compute_sample_level_span_metrics,
)
from token_classifier.postprocess import tokens_to_spans, TokenPrediction
from token_classifier.checkpoint import CheckpointManager, save_config
from token_classifier.schema import create_grouped_split, audit_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Seed
# =============================================================================

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.npu.is_available():
        torch.npu.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    logger.info(f"Random seed set to {seed}")


# =============================================================================
# Training
# =============================================================================

def train_epoch(
    model: TokenHallucinationClassifier,
    train_loader,
    optimizer,
    scheduler,
    device: torch.device,
    epoch: int,
    config: TokenClassifierConfig,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
) -> dict:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_tokens = 0
    total_positive = 0
    num_batches = 0
    
    optimizer.zero_grad()
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        # Forward
        outputs = model(input_ids, attention_mask, labels)
        loss = outputs["loss"]
        valid_tokens = outputs.get("valid_token_count", 0)
        positive_tokens = outputs.get("positive_token_count", 0)
        
        # Scale loss for gradient accumulation
        loss = loss / gradient_accumulation_steps
        
        # Backward
        loss.backward()
        
        # Update metrics
        total_loss += loss.item() * gradient_accumulation_steps
        total_tokens += valid_tokens
        total_positive += positive_tokens
        num_batches += 1
        
        # Optimizer step every gradient_accumulation_steps
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            # Clip gradients
            grad_norm = compute_grad_norm(model, max_grad_norm)
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # Update progress bar
        pos_rate = total_positive / total_tokens if total_tokens > 0 else 0
        pbar.set_postfix({
            "loss": f"{total_loss / num_batches:.4f}",
            "tokens": total_tokens,
            "pos_rate": f"{pos_rate:.2%}",
            "grad": f"{grad_norm:.2f}",
        })
    
    # Handle remaining gradients
    if num_batches % gradient_accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    pos_rate = total_positive / total_tokens if total_tokens > 0 else 0
    
    return {
        "loss": avg_loss,
        "valid_tokens": total_tokens,
        "positive_rate": pos_rate,
    }


@torch.no_grad()
def evaluate(
    model: TokenHallucinationClassifier,
    dev_loader,
    device: torch.device,
    threshold: float = 0.5,
) -> dict:
    """Evaluate on dev set."""
    model.eval()
    
    all_labels = []
    all_preds = []
    all_probs = []
    all_samples_info = []  # Store answer and gold spans for span metrics
    
    for batch in tqdm(dev_loader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"]
        
        outputs = model(input_ids, attention_mask)
        logits = outputs["logits"].cpu()
        
        # Get answer token predictions
        for i in range(labels.shape[0]):
            start_idx = batch["answer_start_indices"][i]
            count = batch["answer_token_counts"][i]
            
            if count <= 0:
                continue
            
            end_idx = start_idx + count
            answer_labels = labels[i, start_idx:end_idx].numpy()
            answer_logits = logits[i, start_idx:end_idx]
            answer_probs = torch.softmax(answer_logits, dim=-1)[:, 1].numpy()
            answer_preds = (answer_probs >= threshold).astype(int)
            
            all_labels.extend(answer_labels.tolist())
            all_preds.extend(answer_preds.tolist())
            all_probs.extend(answer_probs.tolist())
            
            # Store for span metrics computation
            all_samples_info.append({
                "answer": batch.get("answer_text", [""])[i] if isinstance(batch.get("answer_text"), list) else "",
                "gold_spans": batch.get("gold_spans", [[]])[i] if isinstance(batch.get("gold_spans"), list) else [],
                "offsets": batch.get("answer_offsets", [None])[i] if isinstance(batch.get("answer_offsets"), list) else None,
                "answer_probs": answer_probs,
                "answer_preds": answer_preds,
            })
    
    if not all_labels:
        return {}
    
    # Compute token-level metrics
    token_metrics = compute_token_metrics(all_labels, all_preds, all_probs)
    
    # Compute span-level metrics
    span_metrics = compute_sample_level_span_metrics(all_samples_info, threshold)
    
    return {
        "token_metrics": token_metrics,
        "positive_precision": token_metrics["positive_precision"],
        "positive_recall": token_metrics["positive_recall"],
        "positive_f1": token_metrics["positive_f1"],
        "accuracy": token_metrics["accuracy"],
        "macro_f1": token_metrics["macro_f1"],
        "character_precision": span_metrics.get("character_precision", 0.0),
        "character_recall": span_metrics.get("character_recall", 0.0),
        "character_f1": span_metrics.get("character_f1", 0.0),
        "span_metrics": span_metrics,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train token-level hallucination classifier")
    
    # Data
    parser.add_argument("--data_path", type=str, required=True, help="Path to training data")
    parser.add_argument("--model_path", type=str, default=None, help="Model path (or env var)")
    
    # Training
    parser.add_argument("--results_dir", type=str, required=True, help="Results directory")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cpu, npu")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    
    # Model
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--context_stride", type=int, default=128, help="Context window stride")
    parser.add_argument("--context_max_length", type=int, default=400, help="Max context tokens")
    
    # Class weights
    parser.add_argument("--positive_class_weight", type=float, default=None,
                        help="Positive class weight (None=auto, 0=disabled)")
    
    # Aggregation
    parser.add_argument("--context_aggregation", type=str, default="max",
                        choices=["max", "mean"], help="Context window aggregation")
    
    # Threshold
    parser.add_argument("--threshold_metric", type=str, default="character_f1",
                       choices=["token_f1", "character_f1", "answer_f1"])
    parser.add_argument("--threshold_search_min", type=float, default=0.05)
    parser.add_argument("--threshold_search_max", type=float, default=0.95)
    parser.add_argument("--threshold_search_step", type=float, default=0.05)
    
    # Sampling
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_dev_samples", type=int, default=None)
    
    # Diagnostic modes
    parser.add_argument("--smoke_test", action="store_true", help="Run smoke test")
    parser.add_argument("--overfit_diagnostic", action="store_true", help="Run overfit diagnostic")
    parser.add_argument("--overfit_learning_rate", type=float, default=1e-3,
                        help="Learning rate for overfit diagnostic (higher than normal)")
    parser.add_argument("--overfit_freeze_encoder", action="store_true",
                        help="Freeze encoder in overfit diagnostic")
    
    # Other
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Create results directory
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create config
    config = TokenClassifierConfig(
        model_path=args.model_path,
        device=args.device,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.overfit_learning_rate if args.overfit_diagnostic else args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_length,
        context_stride=args.context_stride,
        context_max_length=args.context_max_length,
        positive_class_weight=args.positive_class_weight,
        context_aggregation=args.context_aggregation,
        threshold_metric=args.threshold_metric,
        threshold_search_min=args.threshold_search_min,
        threshold_search_max=args.threshold_search_max,
        threshold_search_step=args.threshold_search_step,
    )
    
    # Save config
    save_config(config, results_dir)
    logger.info(f"Config: {config.to_dict()}")
    
    # Load data
    logger.info(f"Loading data from {args.data_path}")
    all_samples = load_data(args.data_path, strict=False)
    logger.info(f"Loaded {len(all_samples)} samples")

    # Overfit diagnostic mode: use same data for train and dev
    if args.overfit_diagnostic:
        logger.info("Running in overfit diagnostic mode - using same data for train and dev")
        train_samples = all_samples
        dev_samples = all_samples
    else:
        # Create grouped split
        split_result = create_grouped_split(all_samples, dev_fraction=0.2, seed=args.seed)
        train_samples = split_result["train_samples"]
        dev_samples = split_result["dev_samples"]

    # Audit split
    all_split_samples = train_samples + dev_samples
    audit = audit_split(all_split_samples)
    logger.info(f"Split audit: {audit}")
    
    # Save split audit
    with open(results_dir / "split_audit.json", "w") as f:
        json.dump(audit, f, indent=2)
    
    # Load tokenizer and model
    logger.info("Loading tokenizer and model...")
    tokenizer, model = load_tokenizer_and_model(config)
    device = get_device(config.device)
    logger.info(f"Using device: {device}")

    # Overfit diagnostic: freeze encoder, set dropout=0, disable early stopping
    if args.overfit_diagnostic:
        if args.overfit_freeze_encoder:
            logger.info("Overfit mode: freezing encoder parameters")
            model.encoder.eval()  # Disable dropout in encoder
            for param in model.encoder.parameters():
                param.requires_grad = False
            # Classifier stays in train mode
            model.classifier.train()
            # Set dropout rate to 0
            if hasattr(model, 'dropout') and model.dropout is not None:
                model.dropout.p = 0.0
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"Overfit mode: trainable parameters = {trainable_params:,}")
    
    # Create dataloaders
    train_loader, dev_loader = create_dataloaders(
        train_samples, dev_samples, tokenizer,
        batch_size=config.batch_size,
        max_length=config.max_length,
        context_stride=config.context_stride,
        context_max_length=config.context_max_length,
        num_workers=args.num_workers,
        max_train_samples=args.max_train_samples,
        max_dev_samples=args.max_dev_samples,
    )
    
    logger.info(f"Train batches: {len(train_loader)}, Dev batches: {len(dev_loader)}")
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    
    # Scheduler
    num_training_steps = len(train_loader) * config.epochs // config.gradient_accumulation_steps
    scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps)
    
    # Checkpoint manager
    checkpoint_manager = CheckpointManager(results_dir)
    
    # Resume if specified
    start_epoch = 0
    if args.resume:
        checkpoint_manager.checkpoint_path = Path(args.resume)
        checkpoint_data = checkpoint_manager.load(model, optimizer, scheduler)
        start_epoch = checkpoint_data["epoch"] + 1
        logger.info(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    best_metric = 0.0
    patience_counter = 0
    
    training_history = []
    
    for epoch in range(start_epoch, config.epochs):
        logger.info(f"\n=== Epoch {epoch + 1}/{config.epochs} ===")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, device, epoch, config,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            max_grad_norm=config.max_grad_norm,
        )
        logger.info(f"Train: {train_metrics}")
        
        # Evaluate
        eval_metrics = evaluate(model, dev_loader, device)
        logger.info(f"Eval: {eval_metrics}")
        
        # Track history
        history_entry = {
            "epoch": epoch,
            "train": train_metrics,
            "eval": eval_metrics,
        }
        training_history.append(history_entry)
        
        # Save history
        with open(results_dir / "training_history.jsonl", "w") as f:
            for entry in training_history:
                f.write(json.dumps(entry) + "\n")
        
        # Save checkpoint if best (overfit mode: always save if metric improves)
        metric_name = config.threshold_metric
        current_metric = eval_metrics.get(metric_name, 0)
        
        if current_metric > best_metric:
            best_metric = current_metric
            patience_counter = 0
            checkpoint_manager.save(
                model, optimizer, scheduler,
                epoch=epoch, step=(epoch + 1) * len(train_loader),
                best_metric=best_metric,
                config=config,
                metrics=eval_metrics,
            )
            logger.info(f"New best {metric_name}: {best_metric:.4f}")
        else:
            patience_counter += 1
            # Overfit mode: disable early stopping, always train all epochs
            if not args.overfit_diagnostic and patience_counter >= config.patience:
                logger.info(f"Early stopping after {patience_counter} epochs without improvement")
                break
    
    # Save final metrics
    final_metrics = {
        "best_metric": best_metric,
        "best_metric_name": config.threshold_metric,
        "total_epochs": len(training_history),
        "target_achieved": best_metric >= 0.95 if args.overfit_diagnostic else True,
    }
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)
    
    logger.info(f"\nTraining complete. Best {config.threshold_metric}: {best_metric:.4f}")
    logger.info(f"Results saved to {results_dir}")
    
    # Overfit diagnostic: return non-zero if targets not met
    if args.overfit_diagnostic:
        token_f1 = eval_metrics.get("token_metrics", {}).get("positive_f1", 0)
        span_f1 = eval_metrics.get("character_f1", 0)  # FIX: use correct span_f1
        pos_preds = eval_metrics.get("token_metrics", {}).get("support_positive", 0)
        neg_preds = eval_metrics.get("token_metrics", {}).get("support_negative", 0)
        
        overfit_report = {
            "initial_loss": training_history[0]["train"]["loss"] if training_history else None,
            "final_loss": training_history[-1]["train"]["loss"] if training_history else None,
            "best_token_f1": float(token_f1),
            "best_span_f1": float(span_f1),
            "best_character_precision": float(eval_metrics.get("character_precision", 0)),
            "best_character_recall": float(eval_metrics.get("character_recall", 0)),
            "positive_predictions": int(pos_preds),
            "negative_predictions": int(neg_preds),
            "target_token_f1": 0.95,
            "target_span_f1": 0.95,
            "loss_decreased": training_history[-1]["train"]["loss"] < training_history[0]["train"]["loss"] if len(training_history) > 1 else False,
            "target_met": token_f1 >= 0.95 and span_f1 >= 0.95,
        }
        with open(results_dir / "overfit_diagnostic.json", "w") as f:
            json.dump(overfit_report, f, indent=2)
        
        logger.info(f"\nOverfit Diagnostic Report:")
        logger.info(f"  Initial loss: {overfit_report['initial_loss']:.4f}")
        logger.info(f"  Final loss: {overfit_report['final_loss']:.4f}")
        logger.info(f"  Best Token F1: {token_f1:.4f}")
        logger.info(f"  Best Span F1: {span_f1:.4f}")
        logger.info(f"  Positive predictions: {pos_preds}")
        logger.info(f"  Negative predictions: {neg_preds}")
        
        if token_f1 < 0.95 or span_f1 < 0.95:
            logger.warning(f"\nOverfit targets NOT met! Token F1={token_f1:.4f}, Span F1={span_f1:.4f}")
            return 1  # Non-zero exit code
    
    return 0


if __name__ == "__main__":
    main()
