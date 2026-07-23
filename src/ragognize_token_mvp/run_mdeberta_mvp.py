"""
MVP Runner - Uses locally cached mDeBERTa model
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

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
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
    logger.info("RAGognize Token-level Hallucination MVP")
    logger.info("Using locally cached mDeBERTa model")
    logger.info("=" * 60)
    
    set_seed(42)
    logger.info(f"PyTorch: {torch.__version__}")
    
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
    
    train_pos = sum(1 for s in train_samples if s.has_hallucination == 1)
    logger.info(f"Train: {len(train_samples)} samples, {train_pos} hallucinated")
    logger.info(f"Val: {len(val_samples)} samples")
    
    with open(output_dir / "data_statistics.json", "w") as f:
        json.dump({"train_samples": len(train_samples), "val_samples": len(val_samples), "train_hallucinated": train_pos}, f, indent=2)
    
    # ==========================================================================
    # Load Model
    # ==========================================================================
    logger.info("\nPhase 2: Loading Model...")
    
    from transformers import AutoModelForTokenClassification, AutoTokenizer, AutoConfig
    
    model_path = "/home/ma-user/work/models/mDeBERTa-v3-base-mnli-xnli"
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    
    # Load config and modify for token classification
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    config.num_labels = 2
    config.id2label = {0: "supported", 1: "hallucinated"}
    config.label2id = {"supported": 0, "hallucinated": 1}
    
    # Load base model
    base_model = AutoModelForTokenClassification.from_pretrained(
        model_path,
        config=config,
        local_files_only=True,
        ignore_mismatched_sizes=True,
    )
    
    device = "cpu"
    base_model = base_model.to(device)
    
    logger.info(f"Model: {model_path}")
    logger.info(f"Parameters: {sum(p.numel() for p in base_model.parameters()):,}")
    
    with open(output_dir / "run_config.json", "w") as f:
        json.dump({
            "model_name": "mDeBERTa-v3-base-mnli-xnli",
            "model_path": model_path,
            "max_length": 512,
            "batch_size": 2,
            "device": device,
            "seed": 42,
        }, f, indent=2)
    
    # ==========================================================================
    # Prepare Dataset
    # ==========================================================================
    logger.info("\nPhase 3: Preparing Dataset...")
    
    from ragognize_token_mvp.dataset import RAGognizeTokenDataset, collate_fn, sample_balanced_subset
    from torch.utils.data import DataLoader
    
    tiny_samples = sample_balanced_subset(train_samples, 16, 16, seed=42)
    tiny_dataset = RAGognizeTokenDataset(tiny_samples, tokenizer, max_length=256)  # Smaller for speed
    tiny_loader = DataLoader(tiny_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    
    logger.info(f"Tiny dataset: {len(tiny_dataset)} samples")
    
    # ==========================================================================
    # Training
    # ==========================================================================
    logger.info("\nPhase 4: Training...")
    
    from torch.optim import AdamW
    
    optimizer = AdamW(base_model.parameters(), lr=5e-5, weight_decay=0.01)
    
    base_model.train()
    losses = []
    step = 0
    max_steps = 50  # Fewer steps for CPU
    
    batch_start = time.time()
    for batch in tiny_loader:
        if step >= max_steps:
            break
        
        outputs = base_model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["labels"].to(device),
        )
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        losses.append(loss.item())
        step += 1
        
        if step % 10 == 0:
            elapsed = time.time() - batch_start
            logger.info(f"Step {step}/{max_steps} | Loss: {loss.item():.4f} | Time: {elapsed:.1f}s")
    
    train_time = time.time() - batch_start
    logger.info(f"Training completed in {train_time:.1f}s")
    logger.info(f"Final loss: {losses[-1]:.4f}" if losses else "No losses")
    
    # ==========================================================================
    # Evaluation
    # ==========================================================================
    logger.info("\nPhase 5: Evaluation...")
    
    base_model.eval()
    
    val_subset = sample_balanced_subset(val_samples, 30, 30, seed=42)
    val_dataset = RAGognizeTokenDataset(val_subset, tokenizer, max_length=256)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
    
    all_preds = []
    all_golds = []
    all_probs = []
    
    with torch.no_grad():
        for batch in val_loader:
            outputs = base_model(
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
    
    # Metrics
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
        "unfaithful_f1": f1,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "n_samples": len(all_preds),
    }
    
    logger.info("\nValidation Metrics:")
    logger.info(f"  Token F1: {f1:.4f}")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    
    with open(output_dir / "validation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Predictions
    import csv
    with open(output_dir / "validation_predictions.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case_id", "source_model", "gold", "pred", "max_prob", "correct"])
        for i, (pred, gold, prob) in enumerate(zip(all_preds, all_golds, all_probs)):
            writer.writerow([val_subset[i].case_id, val_subset[i].source_model, gold, pred, f"{prob:.4f}", 1 if pred == gold else 0])
    
    # Checkpoint
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    torch.save(base_model.state_dict(), ckpt_dir / "mvp.pt")
    
    tiny_result = {"passed": f1 > 0, "token_f1": f1, "accuracy": accuracy, "steps": step, "train_time": train_time}
    with open(output_dir / "tiny_overfit_metrics.json", "w") as f:
        json.dump(tiny_result, f, indent=2)
    
    # Summary
    total_time = time.time() - start_time
    
    logger.info("\n" + "=" * 60)
    logger.info("MVP COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total time: {total_time / 60:.1f} minutes")
    logger.info(f"Model: mDeBERTa-v3-base-mnli-xnli")
    logger.info(f"Device: CPU")
    logger.info(f"Token F1: {f1:.4f}")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Official Test: NOT RUN")
    logger.info(f"Results: {output_dir}")
    
    with open(output_dir / "environment.json", "w") as f:
        json.dump({"torch_version": torch.__version__, "device": device, "model": "mDeBERTa-v3-base-mnli-xnli"}, f, indent=2)
    
    return metrics

if __name__ == "__main__":
    main()
