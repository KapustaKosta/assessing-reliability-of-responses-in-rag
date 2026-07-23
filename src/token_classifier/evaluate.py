"""
Evaluation script for token-level hallucination classifier.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from token_classifier.config import TokenClassifierConfig, get_model_path
from token_classifier.dataset import load_data, create_dataloaders
from token_classifier.model import (
    TokenHallucinationClassifier,
    load_tokenizer_and_model,
    get_device,
)
from token_classifier.metrics import (
    compute_token_metrics,
    compute_span_metrics,
    compute_answer_metrics,
    compute_calibration_metrics,
)
from token_classifier.postprocess import tokens_to_spans, TokenPrediction, extract_answer_tokens_from_offsets
from token_classifier.checkpoint import CheckpointManager, load_config
from token_classifier.schema import create_grouped_split, audit_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Evaluation
# =============================================================================

@torch.no_grad()
def evaluate_model(
    model: TokenHallucinationClassifier,
    data_loader,
    device: torch.device,
    threshold: float = 0.5,
) -> dict:
    """Evaluate model on dataset."""
    model.eval()
    
    results = []
    all_labels = []
    all_preds = []
    all_probs = []
    
    for batch in tqdm(data_loader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"]
        
        outputs = model(input_ids, attention_mask)
        logits = outputs["logits"].cpu()
        
        # Get answer token predictions
        for i in range(labels.shape[0]):
            sample_id = batch["sample_ids"][i]
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
            
            results.append({
                "sample_id": sample_id,
                "labels": answer_labels.tolist(),
                "preds": answer_preds.tolist(),
                "probs": answer_probs.tolist(),
            })
    
    if not all_labels:
        return {}
    
    # Token-level metrics
    token_metrics = compute_token_metrics(all_labels, all_preds, all_probs)
    
    return {
        "token_metrics": token_metrics,
        "results": results,
    }


def search_best_threshold(
    model: TokenHallucinationClassifier,
    data_loader,
    device: torch.device,
    thresholds: list[float],
) -> tuple[float, dict]:
    """Search for best threshold on dev set."""
    best_threshold = 0.5
    best_metrics = {}
    best_score = 0.0
    
    for threshold in thresholds:
        metrics = evaluate_model(model, data_loader, device, threshold)
        if not metrics:
            continue
        
        token_metrics = metrics.get("token_metrics", {})
        score = token_metrics.get("positive_f1", 0)
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_metrics = token_metrics
    
    return best_threshold, best_metrics


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate token-level hallucination classifier")
    
    # Data
    parser.add_argument("--data_path", type=str, required=True, help="Path to data")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Path to checkpoint (or use results_dir)")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Results directory (contains checkpoint)")
    
    # Threshold
    parser.add_argument("--threshold", type=float, default=None,
                        help="Fixed threshold (if not provided, search on dev)")
    parser.add_argument("--threshold_min", type=float, default=0.05)
    parser.add_argument("--threshold_max", type=float, default=0.95)
    parser.add_argument("--threshold_step", type=float, default=0.05)
    
    # Other
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output_dir", type=str, required=True)
    
    args = parser.parse_args()
    
    # Determine checkpoint path
    if args.checkpoint_path:
        checkpoint_path = args.checkpoint_path
    elif args.results_dir:
        checkpoint_path = os.path.join(args.results_dir, "best_checkpoint.pt")
    else:
        raise ValueError("Must specify --checkpoint_path or --results_dir")
    
    # Load config
    checkpoint_manager = CheckpointManager(os.path.dirname(checkpoint_path))
    checkpoint_manager.checkpoint_path = checkpoint_path
    
    if not checkpoint_manager.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    config = checkpoint_manager.load_config()
    logger.info(f"Loaded config from checkpoint")
    
    # Override with CLI args
    if args.device:
        config.device = args.device
    if args.threshold_min:
        config.threshold_search_min = args.threshold_min
    if args.threshold_max:
        config.threshold_search_max = args.threshold_max
    if args.threshold_step:
        config.threshold_search_step = args.threshold_step
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    logger.info("Loading model...")
    tokenizer, model = load_tokenizer_and_model(config)
    device = get_device(config.device)
    model = model.to(device)
    
    # Load checkpoint
    checkpoint_manager.load(model)
    
    # Load data
    logger.info(f"Loading data from {args.data_path}")
    all_samples = load_data(args.data_path, strict=False)
    
    # Create split
    split_result = create_grouped_split(all_samples, dev_fraction=0.2, seed=config.seed)
    dev_samples = split_result["dev_samples"]
    
    # Audit split
    audit = audit_split(all_samples)
    logger.info(f"Split audit: {audit}")
    
    # Create dataloader
    _, dev_loader = create_dataloaders(
        [], dev_samples, tokenizer,
        batch_size=args.batch_size,
        max_length=config.max_length,
        context_stride=config.context_stride,
        context_max_length=config.context_max_length,
    )
    
    # Determine threshold
    if args.threshold is not None:
        threshold = args.threshold
        logger.info(f"Using fixed threshold: {threshold}")
    else:
        logger.info("Searching for best threshold...")
        thresholds = np.arange(
            config.threshold_search_min,
            config.threshold_search_max + config.threshold_search_step,
            config.threshold_search_step
        ).tolist()
        threshold, threshold_metrics = search_best_threshold(model, dev_loader, device, thresholds)
        logger.info(f"Best threshold: {threshold}")
        logger.info(f"Threshold metrics: {threshold_metrics}")
        
        # Save threshold search results
        with open(output_dir / "threshold_search.json", "w") as f:
            json.dump({
                "best_threshold": threshold,
                "best_metrics": threshold_metrics,
                "search_range": [float(config.threshold_search_min), float(config.threshold_search_max)],
                "search_step": float(config.threshold_search_step),
            }, f, indent=2)
    
    # Final evaluation
    logger.info("Final evaluation...")
    results = evaluate_model(model, dev_loader, device, threshold)
    
    token_metrics = results.get("token_metrics", {})
    
    # Compute calibration metrics
    if token_metrics:
        all_labels = []
        all_probs = []
        for r in results.get("results", []):
            all_labels.extend(r["labels"])
            all_probs.extend(r["probs"])
        
        calibration = compute_calibration_metrics(all_labels, all_probs)
    else:
        calibration = {}
    
    # Save results
    final_metrics = {
        "threshold": threshold,
        "token_metrics": token_metrics,
        "calibration": calibration,
    }
    
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)
    
    # Save predictions
    with open(output_dir / "predictions.jsonl", "w") as f:
        for r in results.get("results", []):
            f.write(json.dumps(r) + "\n")
    
    # Save split audit
    with open(output_dir / "split_audit.json", "w") as f:
        json.dump(audit, f, indent=2)
    
    logger.info(f"\nEvaluation complete. Results saved to {output_dir}")
    logger.info(f"Token F1: {token_metrics.get('positive_f1', 'N/A')}")
    logger.info(f"Accuracy: {token_metrics.get('accuracy', 'N/A')}")
    
    return final_metrics


if __name__ == "__main__":
    main()
