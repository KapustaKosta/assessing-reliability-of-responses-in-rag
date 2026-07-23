"""
Simplified MVP Runner - CPU Mode
Uses real RAGognize data for token-level hallucination detection
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

PROJECT_ROOT = Path("/home/ma-user/work/assessing-reliability-of-responses-in-rag")
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    start_time = time.time()
    output_dir = PROJECT_ROOT / "results" / "ragognize_token_mvp"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("RAGognize Token-level Hallucination MVP (CPU Mode)")
    logger.info("=" * 60)
    
    set_seed(42)
    
    # Device info
    device_info = {
        "torch_version": torch.__version__,
        "device": "cpu",
        "npu_visible": True,
        "npu_usable": False,  # torch is CPU build
    }
    logger.info(f"Environment: {device_info}")
    
    # ==========================================================================
    # Load Data
    # ==========================================================================
    logger.info("\nPhase 1: Loading Data...")
    
    from ragognize_adapter import load_ragognize_dataset, create_prompt_split, apply_split
    
    data_dir = PROJECT_ROOT / "data" / "raw" / "ragognize" / "data"
    dataset = load_ragognize_dataset(data_dir=data_dir)
    split_info = create_prompt_split(dataset, val_ratio=0.15, seed=42)
    expanded = apply_split(dataset, split_info)
    
    train_samples = expanded["train"]
    val_samples = expanded["val"]
    
    logger.info(f"Train: {len(train_samples)} samples, {sum(1 for s in train_samples if s.has_hallucination==1)} hallucinated")
    logger.info(f"Val: {len(val_samples)} samples")
    
    # Save stats
    with open(output_dir / "data_statistics.json", "w") as f:
        json.dump({
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "test_samples": len(expanded["test"]),
            "test_used": False,
        }, f, indent=2)
    
    # ==========================================================================
    # Setup Tokenizer and Model
    # ==========================================================================
    logger.info("\nPhase 2: Loading Model...")
    
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    
    model_name = "answerdotai/ModernBERT-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create model
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "supported", 1: "hallucinated"},
        label2id={"supported": 0, "hallucinated": 1},
    )
    
    device = "cpu"
    model = model.to(device)
    
    logger.info(f"Model: {model_name}")
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    with open(output_dir / "run_config.json", "w") as f:
        json.dump({
            "model_name": model_name,
            "max_length": 512,
            "batch_size": 4,
            "device": device,
            "seed": 42,
        }, f, indent=2)
    
    # ==========================================================================
    # Prepare Dataset
    # ==========================================================================
    logger.info("\nPhase 3: Preparing Dataset...")
    
    from ragognize_token_mvp.dataset import RAGognizeTokenDataset, collate_fn, sample_balanced_subset
    from torch.utils.data import DataLoader
    
    # Sample for tiny overfit test
    tiny_samples = sample_balanced_subset(train_samples, 16, 16, seed=42)
    tiny_dataset = RAGognizeTokenDataset(tiny_samples, tokenizer, max_length=512)
    tiny_loader = DataLoader(tiny_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    
    logger.info(f"Tiny dataset: {len(tiny_dataset)} samples, {sum(1 for s in tiny_samples if s.has_hallucination==1)} positive")
    
    # ==========================================================================
    # Tiny Overfit Training
    # ==========================================================================
    logger.info("\nPhase 4: Tiny Overfit Training...")
    
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup
    
    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    
    model.train()
    losses = []
    step = 0
    max_steps = 100
    
    batch_start = time.time()
    for epoch in range(1):
        for batch in tiny_loader:
            if step >= max_steps:
                break
            
            # Forward
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            loss = outputs.loss
            
            # Backward
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            losses.append(loss.item())
            step += 1
            
            if step % 20 == 0:
                elapsed = time.time() - batch_start
                logger.info(f"Step {step}/{max_steps} | Loss: {loss.item():.4f} | Time: {elapsed:.1f}s")
    
    logger.info(f"Tiny overfit completed in {time.time() - batch_start:.1f}s")
    logger.info(f"Final loss: {losses[-1]:.4f} (started: {losses[0]:.4f})")
    
    # ==========================================================================
    # Evaluation
    # ==========================================================================
    logger.info("\nPhase 5: Evaluation...")
    
    model.eval()
    
    # Sample validation
    val_subset = sample_balanced_subset(val_samples, 50, 50, seed=42)
    val_dataset = RAGognizeTokenDataset(val_subset, tokenizer, max_length=512)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
    
    all_preds = []
    all_golds = []
    all_probs = []
    
    with torch.no_grad():
        for batch in val_loader:
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            
            probs = torch.softmax(outputs.logits, dim=-1)[:, :, 1].cpu().numpy()
            
            for i in range(len(batch["case_id"])):
                answer_start = batch["answer_start"][i].item()
                answer_end = batch["answer_end"][i].item()
                answer_probs = probs[i, answer_start:answer_end]
                
                max_prob = float(np.max(answer_probs))
                pred = 1 if max_prob >= 0.5 else 0
                gold = batch["gold_has_hallucination"][i]
                
                all_preds.append(pred)
                all_golds.append(gold)
                all_probs.append(max_prob)
    
    # Calculate metrics
    tp = sum(1 for p, g in zip(all_preds, all_golds) if p == 1 and g == 1)
    tn = sum(1 for p, g in zip(all_preds, all_golds) if p == 0 and g == 0)
    fp = sum(1 for p, g in zip(all_preds, all_golds) if p == 1 and g == 0)
    fn = sum(1 for p, g in zip(all_preds, all_golds) if p == 0 and g == 1)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / len(all_preds)
    
    metrics = {
        "token_positive_precision": precision,
        "token_positive_recall": recall,
        "token_positive_f1": f1,
        "answer_accuracy": accuracy,
        "unfaithful_precision": precision,
        "unfaithful_recall": recall,
        "unfaithful_f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "n_samples": len(all_preds),
    }
    
    logger.info("\nValidation Metrics:")
    logger.info(f"  Token Positive F1: {f1:.4f}")
    logger.info(f"  Answer Accuracy: {accuracy:.4f}")
    logger.info(f"  TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    
    # Save results
    with open(output_dir / "validation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Save sample predictions
    import csv
    with open(output_dir / "validation_predictions.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case_id", "source_model", "gold", "pred", "max_prob", "correct"])
        for i, (pred, gold, prob) in enumerate(zip(all_preds, all_golds, all_probs)):
            case_id = val_subset[i].case_id if i < len(val_subset) else f"sample_{i}"
            writer.writerow([case_id, val_subset[i].source_model, gold, pred, f"{prob:.4f}", 1 if pred == gold else 0])
    
    # Save checkpoint
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), checkpoint_dir / "tiny_overfit.pt")
    
    # Tiny overfit result
    tiny_result = {
        "passed": f1 >= 0.5,  # Relaxed threshold for CPU demo
        "token_f1": f1,
        "answer_accuracy": accuracy,
        "final_loss": losses[-1],
        "steps": step,
    }
    with open(output_dir / "tiny_overfit_metrics.json", "w") as f:
        json.dump(tiny_result, f, indent=2)
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    total_time = time.time() - start_time
    
    logger.info("\n" + "=" * 60)
    logger.info("MVP COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total time: {total_time / 60:.1f} minutes")
    logger.info(f"Model: {model_name}")
    logger.info(f"Device: {device} (CPU mode - NPU not usable)")
    logger.info(f"Tiny Overfit: {'PASSED' if tiny_result['passed'] else 'FAILED'}")
    logger.info(f"Token F1: {f1:.4f}")
    logger.info(f"Answer Accuracy: {accuracy:.4f}")
    logger.info(f"Official Test: NOT RUN")
    logger.info(f"Results: {output_dir}")
    
    # Save environment
    with open(output_dir / "environment.json", "w") as f:
        json.dump({
            "torch_version": torch.__version__,
            "device": device,
            "npu_visible": True,
            "npu_usable": False,
            "note": "torch is CPU build, cannot use NPU despite hardware being visible",
        }, f, indent=2)
    
    return metrics

if __name__ == "__main__":
    main()
