"""
Main script for RAGognize Token-level Hallucination Detection MVP.
Day 1 - Real data, real results
"""

import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ragognize_token_mvp import (
    TokenClassifier,
    RAGognizeTokenDataset,
    Evaluator,
    load_ragognize_token_data,
    sample_balanced_subset,
    get_device_info,
)
from ragognize_token_mvp.trainer import TrainConfig, train_tiny_overfit, train_full

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / "results" / "ragognize_token_mvp" / "train.log"),
    ],
)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    start_time = time.time()
    
    # Output directory
    output_dir = PROJECT_ROOT / "results" / "ragognize_token_mvp"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("RAGognize Token-level Hallucination Detection MVP")
    logger.info("=" * 60)
    
    # Set seeds
    set_seed(42)
    
    # Environment info
    device_info = get_device_info()
    logger.info(f"Environment: {json.dumps(device_info, indent=2)}")
    
    # Save environment info
    with open(output_dir / "environment.json", "w") as f:
        json.dump(device_info, f, indent=2)
    
    # ==========================================================================
    # Phase 1: Load Data
    # ==========================================================================
    logger.info("\n" + "=" * 40)
    logger.info("Phase 1: Loading Data")
    logger.info("=" * 40)
    
    data_dir = PROJECT_ROOT / "data" / "raw" / "ragognize" / "data"
    
    # Load RAGognize data
    expanded = load_ragognize_token_data(data_dir=data_dir, val_ratio=0.15, seed=42)
    
    train_samples = expanded["train"]
    val_samples = expanded["val"]
    test_samples = expanded["test"]
    
    logger.info(f"Train samples: {len(train_samples)}")
    logger.info(f"Val samples: {len(val_samples)}")
    logger.info(f"Test samples: {len(test_samples)} (NOT USED)")
    
    # Sample distribution
    train_pos = sum(1 for s in train_samples if s.has_hallucination == 1)
    train_neg = sum(1 for s in train_samples if s.has_hallucination == 0)
    logger.info(f"Train: {train_pos} hallucinated, {train_neg} faithful")
    
    # Save data statistics
    data_stats = {
        "train_total": len(train_samples),
        "train_hallucinated": train_pos,
        "train_faithful": train_neg,
        "val_total": len(val_samples),
        "test_total": len(test_samples),
        "test_used": False,
    }
    with open(output_dir / "data_statistics.json", "w") as f:
        json.dump(data_stats, f, indent=2)
    
    # ==========================================================================
    # Phase 2: Tokenizer and Model Setup
    # ==========================================================================
    logger.info("\n" + "=" * 40)
    logger.info("Phase 2: Model Setup")
    logger.info("=" * 40)
    
    # Model selection - try to download ModernBERT, fallback to DistilBERT
    model_options = [
        "answerdotai/ModernBERT-base",
        "distilbert-base-uncased",
    ]
    
    tokenizer = None
    model_name = None
    
    for model_candidate in model_options:
        try:
            from transformers import AutoTokenizer
            logger.info(f"Trying to load tokenizer: {model_candidate}")
            tokenizer = AutoTokenizer.from_pretrained(model_candidate)
            model_name = model_candidate
            logger.info(f"Successfully loaded: {model_candidate}")
            break
        except Exception as e:
            logger.warning(f"Failed to load {model_candidate}: {e}")
            continue
    
    if tokenizer is None:
        raise RuntimeError("Could not load any tokenizer")
    
    # Create model
    device = "cpu"  # Force CPU since no GPU/NPU available
    model = TokenClassifier(model_name=model_name, device=device)
    model = model.to(device)
    
    logger.info(f"Model: {model_name}")
    logger.info(f"Device: {device}")
    
    # Save config
    config = {
        "model_name": model_name,
        "max_length": 512,
        "batch_size": 8,
        "device": device,
        "seed": 42,
        "threshold": 0.5,
    }
    with open(output_dir / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # ==========================================================================
    # Phase 3: Tiny Overfit Test
    # ==========================================================================
    logger.info("\n" + "=" * 40)
    logger.info("Phase 3: Tiny Overfit Test")
    logger.info("=" * 40)
    
    # Sample 32 balanced examples for tiny overfit
    tiny_samples = sample_balanced_subset(
        train_samples,
        n_positive=16,
        n_negative=16,
        seed=42,
    )
    logger.info(f"Tiny overfit samples: {len(tiny_samples)}")
    
    # Create dataset
    tiny_dataset = RAGognizeTokenDataset(
        tiny_samples,
        tokenizer,
        max_length=512,
    )
    logger.info(f"Tiny dataset tokenized: {len(tiny_dataset)} samples")
    
    # Create dataloader
    from torch.utils.data import DataLoader
    from ragognize_token_mvp.dataset import collate_fn
    tiny_loader = DataLoader(
        tiny_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn,
    )
    
    # Train config for tiny overfit
    tiny_config = TrainConfig(
        model_name=model_name,
        max_length=512,
        batch_size=4,
        gradient_accumulation=1,
        learning_rate=5e-5,
        max_steps=100,
        max_epochs=1,
        device=device,
    )
    
    # Train
    logger.info("Starting tiny overfit training...")
    tiny_result = train_tiny_overfit(
        model,
        tiny_loader,
        tiny_config,
        output_dir / "tiny_overfit",
    )
    
    logger.info(f"Tiny overfit completed in {tiny_result.train_time:.1f}s")
    
    # Evaluate on tiny set
    evaluator = Evaluator(tokenizer)
    
    # Quick eval on same tiny set
    model.eval()
    tiny_metrics = evaluator.evaluate(model, tiny_loader)
    
    logger.info("Tiny Overfit Results:")
    logger.info(f"  Token F1: {tiny_metrics.get('positive_f1', 0):.4f}")
    logger.info(f"  Span F1: {tiny_metrics.get('character_f1', 0):.4f}")
    logger.info(f"  Answer Acc: {tiny_metrics.get('answer_accuracy', 0):.4f}")
    
    # Check if passed
    token_f1_pass = tiny_metrics.get("positive_f1", 0) >= 0.90
    span_f1_pass = tiny_metrics.get("character_f1", 0) >= 0.80
    
    tiny_passed = token_f1_pass and span_f1_pass
    
    # Save tiny overfit results
    tiny_overfit_results = {
        "passed": tiny_passed,
        "token_f1": tiny_metrics.get("positive_f1", 0),
        "span_f1": tiny_metrics.get("character_f1", 0),
        "answer_accuracy": tiny_metrics.get("answer_accuracy", 0),
        "train_time": tiny_result.train_time,
        "n_steps": tiny_result.final_step,
    }
    with open(output_dir / "tiny_overfit_metrics.json", "w") as f:
        json.dump(tiny_overfit_results, f, indent=2)
    
    if not tiny_passed:
        logger.warning("Tiny Overfit FAILED - check alignment and labels")
        logger.warning("Proceeding anyway for MVP...")
    
    # ==========================================================================
    # Phase 4: Quick Pilot Training
    # ==========================================================================
    logger.info("\n" + "=" * 40)
    logger.info("Phase 4: Pilot Training")
    logger.info("=" * 40)
    
    # Sample 500 for pilot
    pilot_samples = sample_balanced_subset(
        train_samples,
        n_positive=min(250, train_pos),
        n_negative=min(250, train_neg),
        seed=42,
    )
    logger.info(f"Pilot samples: {len(pilot_samples)}")
    
    # Create pilot dataset
    pilot_dataset = RAGognizeTokenDataset(
        pilot_samples,
        tokenizer,
        max_length=512,
    )
    logger.info(f"Pilot dataset: {len(pilot_dataset)} tokenized samples")
    
    # Validation dataset (smaller for speed)
    val_subset = sample_balanced_subset(
        val_samples,
        n_positive=min(100, sum(1 for s in val_samples if s.has_hallucination == 1)),
        n_negative=min(100, sum(1 for s in val_samples if s.has_hallucination == 0)),
        seed=42,
    )
    val_dataset = RAGognizeTokenDataset(
        val_subset,
        tokenizer,
        max_length=512,
    )
    logger.info(f"Val dataset (subset): {len(val_dataset)} tokenized samples")
    
    # Pilot config
    pilot_config = TrainConfig(
        model_name=model_name,
        max_length=512,
        batch_size=8,
        gradient_accumulation=2,
        learning_rate=2e-5,
        max_steps=200,
        max_epochs=1,
        eval_every=20,
        device=device,
    )
    
    # Create loaders
    pilot_loader = DataLoader(
        pilot_dataset,
        batch_size=pilot_config.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
    )
    
    # Train
    logger.info("Starting pilot training...")
    pilot_result = train_full(
        model,
        pilot_loader,
        val_loader,
        pilot_config,
        output_dir / "checkpoints",
        evaluator,
    )
    
    logger.info(f"Pilot training completed in {pilot_result.train_time:.1f}s")
    
    # ==========================================================================
    # Phase 5: Final Evaluation
    # ==========================================================================
    logger.info("\n" + "=" * 40)
    logger.info("Phase 5: Final Evaluation")
    logger.info("=" * 40)
    
    # Full validation evaluation
    val_full_dataset = RAGognizeTokenDataset(
        val_samples,
        tokenizer,
        max_length=512,
    )
    val_full_loader = DataLoader(
        val_full_dataset,
        batch_size=8,
        shuffle=False,
    )
    
    logger.info(f"Full validation: {len(val_full_dataset)} samples")
    
    # Load best checkpoint
    best_checkpoint = output_dir / "checkpoints" / "checkpoints" / "best.pt"
    if best_checkpoint.exists():
        checkpoint = torch.load(best_checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"])
        logger.info(f"Loaded best checkpoint from step {checkpoint.get('step', 'unknown')}")
    else:
        logger.warning("Best checkpoint not found, using current model")
    
    # Evaluate
    final_metrics = evaluator.evaluate(model, val_full_loader)
    
    logger.info("Final Validation Metrics:")
    logger.info(f"  Token Precision: {final_metrics.get('positive_precision', 0):.4f}")
    logger.info(f"  Token Recall: {final_metrics.get('positive_recall', 0):.4f}")
    logger.info(f"  Token F1: {final_metrics.get('positive_f1', 0):.4f}")
    logger.info(f"  Span Precision: {final_metrics.get('character_precision', 0):.4f}")
    logger.info(f"  Span Recall: {final_metrics.get('character_recall', 0):.4f}")
    logger.info(f"  Span F1: {final_metrics.get('character_f1', 0):.4f}")
    logger.info(f"  Answer Accuracy: {final_metrics.get('answer_accuracy', 0):.4f}")
    logger.info(f"  Unfaithful F1: {final_metrics.get('unfaithful_f1', 0):.4f}")
    
    # Confusion matrix
    logger.info(f"  TP: {final_metrics.get('tp', 0)}, TN: {final_metrics.get('tn', 0)}")
    logger.info(f"  FP: {final_metrics.get('fp', 0)}, FN: {final_metrics.get('fn', 0)}")
    
    # Save metrics
    with open(output_dir / "validation_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)
    
    # ==========================================================================
    # Phase 6: Generate Sample Predictions
    # ==========================================================================
    logger.info("\n" + "=" * 40)
    logger.info("Phase 6: Sample Predictions")
    logger.info("=" * 40)
    
    # Generate predictions on validation subset
    eval_subset = sample_balanced_subset(
        val_samples,
        n_positive=50,
        n_negative=50,
        seed=123,
    )
    eval_dataset = RAGognizeTokenDataset(eval_subset, tokenizer, max_length=512)
    eval_loader = DataLoader(eval_dataset, batch_size=8, shuffle=False)
    
    # Get detailed predictions
    predictions = []
    model.eval()
    
    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            probs = model.predict_proba(input_ids, attention_mask)
            probs = probs.cpu().numpy()
            
            for i in range(len(batch["case_id"])):
                pred = {
                    "case_id": batch["case_id"][i],
                    "source_model": batch["source_model"][i],
                    "question_short": batch["question"][i][:100],
                    "answer_short": batch["answer"][i][:200],
                    "gold_has_hallucination": batch["gold_has_hallucination"][i],
                    "max_prob": float(np.max(probs[i])),
                    "predicted_has_hallucination": 1 if np.max(probs[i]) >= 0.5 else 0,
                }
                predictions.append(pred)
    
    # Save predictions
    import csv
    predictions_path = output_dir / "validation_predictions.csv"
    with open(predictions_path, "w", newline="", encoding="utf-8") as f:
        if predictions:
            writer = csv.DictWriter(f, fieldnames=predictions[0].keys())
            writer.writeheader()
            writer.writerows(predictions)
    logger.info(f"Saved {len(predictions)} predictions to {predictions_path}")
    
    # ==========================================================================
    # Phase 7: Summary
    # ==========================================================================
    total_time = time.time() - start_time
    
    logger.info("\n" + "=" * 60)
    logger.info("MVP COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total time: {total_time / 60:.1f} minutes")
    logger.info(f"Model: {model_name}")
    logger.info(f"Device: {device}")
    logger.info(f"Tiny Overfit: {'PASSED' if tiny_passed else 'FAILED'}")
    logger.info(f"Pilot samples: {len(pilot_samples)}")
    logger.info(f"Final Token F1: {final_metrics.get('positive_f1', 0):.4f}")
    logger.info(f"Final Span F1: {final_metrics.get('character_f1', 0):.4f}")
    logger.info(f"Final Answer Accuracy: {final_metrics.get('answer_accuracy', 0):.4f}")
    logger.info(f"Official Test: NOT RUN")
    
    # Save summary
    summary = {
        "model_name": model_name,
        "device": device,
        "tiny_overfit_passed": tiny_passed,
        "pilot_samples": len(pilot_samples),
        "validation_samples": len(val_samples),
        "total_time_minutes": total_time / 60,
        "final_metrics": final_metrics,
        "official_test_run": False,
    }
    
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info("\nResults saved to:")
    logger.info(f"  {output_dir}")
    
    return summary


if __name__ == "__main__":
    main()
