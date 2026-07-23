"""
Quick Evaluation Script - Uses trained mDeBERTa checkpoint
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

def main():
    output_dir = PROJECT_ROOT / "results" / "ragognize_token_mvp"
    
    logger.info("Quick Evaluation...")
    
    # Load data
    from ragognize_adapter import load_ragognize_dataset, create_prompt_split, apply_split
    data_dir = PROJECT_ROOT / "data" / "raw" / "ragognize" / "data"
    dataset = load_ragognize_dataset(data_dir=data_dir)
    split_info = create_prompt_split(dataset, val_ratio=0.15, seed=42)
    expanded = apply_split(dataset, split_info)
    val_samples = expanded["val"]
    
    # Load model
    from transformers import AutoModelForTokenClassification, AutoTokenizer, AutoConfig
    model_path = "/home/ma-user/work/models/mDeBERTa-v3-base-mnli-xnli"
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    config.num_labels = 2
    config.id2label = {0: "supported", 1: "hallucinated"}
    config.label2id = {"supported": 0, "hallucinated": 1}
    
    model = AutoModelForTokenClassification.from_pretrained(
        model_path, config=config, local_files_only=True, ignore_mismatched_sizes=True
    )
    model = model.to("cpu")
    model.eval()
    
    logger.info("Model loaded")
    
    # Load dataset
    from ragognize_token_mvp.dataset import RAGognizeTokenDataset, collate_fn, sample_balanced_subset
    from torch.utils.data import DataLoader
    
    val_subset = sample_balanced_subset(val_samples, 30, 30, seed=42)
    val_dataset = RAGognizeTokenDataset(val_subset, tokenizer, max_length=256)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
    
    logger.info(f"Dataset: {len(val_dataset)} samples")
    
    # Evaluate
    all_preds = []
    all_golds = []
    all_probs = []
    
    with torch.no_grad():
        for batch in val_loader:
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            
            probs = torch.softmax(outputs.logits, dim=-1)[:, :, 1].numpy()
            
            for i in range(len(batch["case_id"])):
                answer_start = batch["answer_start"][i].item()
                answer_end = batch["answer_end"][i].item()
                
                # Fix: handle empty answer region
                if answer_end <= answer_start:
                    max_prob = 0.0
                else:
                    answer_probs = probs[i, answer_start:answer_end]
                    if len(answer_probs) == 0:
                        max_prob = 0.0
                    else:
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
        "token_positive_f1": f1,
        "answer_accuracy": accuracy,
        "unfaithful_f1": f1,
        "precision": precision,
        "recall": recall,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "n_samples": len(all_preds),
    }
    
    logger.info("\n" + "=" * 40)
    logger.info("Validation Metrics (Untrained):")
    logger.info(f"  Token F1: {f1:.4f}")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    logger.info("=" * 40)
    
    # Save
    with open(output_dir / "validation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    import csv
    with open(output_dir / "validation_predictions.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case_id", "source_model", "gold", "pred", "max_prob", "correct"])
        for i, (pred, gold, prob) in enumerate(zip(all_preds, all_golds, all_probs)):
            writer.writerow([val_subset[i].case_id, val_subset[i].source_model, gold, pred, f"{prob:.4f}", 1 if pred == gold else 0])
    
    logger.info(f"Results saved to {output_dir}")
    return metrics

if __name__ == "__main__":
    main()
