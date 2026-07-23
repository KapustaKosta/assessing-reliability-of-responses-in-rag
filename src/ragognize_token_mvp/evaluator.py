"""
Evaluation for RAGognize Token-level Hallucination Detection.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from ragognize_token_mvp.model import TokenClassifier
from ragognize_token_mvp.postprocess import (
    span_from_tokens,
    compute_token_metrics,
    compute_span_metrics,
    compute_answer_level_metrics,
    PredictedSpan,
)

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Evaluator for token-level hallucination detection.
    """
    
    def __init__(
        self,
        tokenizer,
        threshold: float = 0.5,
    ):
        self.tokenizer = tokenizer
        self.threshold = threshold
    
    def evaluate(
        self,
        model: TokenClassifier,
        data_loader: DataLoader,
    ) -> dict:
        """Evaluate model on dataset."""
        model.eval()
        device = model.device
        
        all_token_gold = []
        all_token_probs = []
        
        all_span_metrics = []
        all_answer_metrics = []
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"]
                
                # Predict
                probs = model.predict_proba(input_ids, attention_mask)
                probs = probs.cpu().numpy()
                labels_np = labels.numpy()
                
                # Process each sample in batch
                for i in range(len(batch["case_id"])):
                    # Get token labels and probs for this sample
                    token_labels = labels_np[i]
                    token_probs = probs[i]
                    
                    answer_start = batch["answer_start"][i].item()
                    answer_end = batch["answer_end"][i].item()
                    gold_has = batch["gold_has_hallucination"][i]
                    gold_spans = batch["gold_spans"][i]
                    answer_text = batch["answer_text"][i]
                    
                    # Token metrics (only answer tokens)
                    answer_labels = token_labels[answer_start:answer_end]
                    answer_probs = token_probs[answer_start:answer_end]
                    
                    valid_pairs = [(g, p) for g, p in zip(answer_labels, answer_probs) if g != -100]
                    if valid_pairs:
                        gold_labels, pred_probs = zip(*valid_pairs)
                        all_token_gold.extend(gold_labels)
                        all_token_probs.extend(pred_probs)
                    
                    # Span metrics
                    gold_span_list = [(s[0], s[1], s[2]) for s in gold_spans]
                    
                    # Create offset mapping for this answer
                    offset_mapping = [(j, j+1) for j in range(min(len(answer_text), 5000))]
                    
                    # Predict spans
                    try:
                        pred_spans = span_from_tokens(
                            token_probs,
                            [offset_mapping] * len(token_probs),
                            answer_start,
                            answer_end,
                            answer_text,
                            self.threshold,
                        )
                    except Exception as e:
                        logger.debug(f"Span extraction error: {e}")
                        pred_spans = []
                    
                    span_metrics = compute_span_metrics(gold_span_list, pred_spans)
                    all_span_metrics.append(span_metrics)
                    
                    # Answer level metrics
                    answer_metrics = compute_answer_level_metrics(gold_has, pred_spans, self.threshold)
                    all_answer_metrics.append(answer_metrics)
        
        # Aggregate metrics
        # Token level
        token_agg = compute_token_metrics(all_token_gold, all_token_probs, self.threshold)
        
        # Span level
        span_precision = np.mean([m["precision"] for m in all_span_metrics]) if all_span_metrics else 0
        span_recall = np.mean([m["recall"] for m in all_span_metrics]) if all_span_metrics else 0
        span_f1 = np.mean([m["f1"] for m in all_span_metrics]) if all_span_metrics else 0
        
        # Confusion matrix
        tn = sum(1 for m in all_answer_metrics if m["pred_has_hallucination"] == 0 and m["correct"] == 1)
        fp = sum(1 for m in all_answer_metrics if m["pred_has_hallucination"] == 1 and m["correct"] == 0)
        fn = sum(1 for m in all_answer_metrics if m["pred_has_hallucination"] == 0 and m["correct"] == 0 and m["n_predicted_spans"] == 0)
        tp = len(all_answer_metrics) - tn - fp - fn
        
        # Better calculation using gold_has_hallucination
        tn = fp = fn = tp = 0
        for i, m in enumerate(all_answer_metrics):
            pred = m["pred_has_hallucination"]
            gold = batch["gold_has_hallucination"][i]
            if pred == 0 and gold == 0:
                tn += 1
            elif pred == 1 and gold == 0:
                fp += 1
            elif pred == 0 and gold == 1:
                fn += 1
            else:
                tp += 1
        
        # Unfaithful class (1) metrics
        unfaithful_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        unfaithful_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        unfaithful_f1 = 2 * unfaithful_precision * unfaithful_recall / (unfaithful_precision + unfaithful_recall) if (unfaithful_precision + unfaithful_recall) > 0 else 0
        
        accuracy = (tp + tn) / len(all_answer_metrics) if all_answer_metrics else 0
        macro_f1 = unfaithful_f1 / 2  # Simplified
        
        return {
            "positive_precision": token_agg.get("precision", 0),
            "positive_recall": token_agg.get("recall", 0),
            "positive_f1": token_agg.get("f1", 0),
            "token_accuracy": token_agg.get("accuracy", 0),
            "character_precision": span_precision,
            "character_recall": span_recall,
            "character_f1": span_f1,
            "answer_accuracy": accuracy,
            "answer_macro_f1": macro_f1,
            "unfaithful_precision": unfaithful_precision,
            "unfaithful_recall": unfaithful_recall,
            "unfaithful_f1": unfaithful_f1,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "n_samples": len(all_answer_metrics),
            "threshold": self.threshold,
        }
    
    def evaluate_detailed(
        self,
        model: TokenClassifier,
        data_loader: DataLoader,
        output_dir: Path,
        max_samples: int = 100,
    ) -> dict:
        """Evaluate with detailed per-sample output."""
        metrics = self.evaluate(model, data_loader)
        
        predictions_path = output_dir / "validation_predictions.csv"
        
        model.eval()
        device = model.device
        
        predictions = []
        
        with torch.no_grad():
            for batch in data_loader:
                if len(predictions) >= max_samples:
                    break
                
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                probs = model.predict_proba(input_ids, attention_mask)
                probs = probs.cpu().numpy()
                
                for i in range(len(batch["case_id"])):
                    if len(predictions) >= max_samples:
                        break
                    
                    answer_start = batch["answer_start"][i].item()
                    answer_end = batch["answer_end"][i].item()
                    answer_probs = probs[i][answer_start:answer_end]
                    max_prob = float(np.max(answer_probs))
                    
                    sample = {
                        "case_id": batch["case_id"][i],
                        "source_model": batch["source_model"][i],
                        "question_short": batch["question"][i][:100],
                        "answer_short": batch["answer"][i][:200],
                        "gold_has_hallucination": batch["gold_has_hallucination"][i],
                        "predicted_has_hallucination": 1 if max_prob >= self.threshold else 0,
                        "max_hallucination_probability": max_prob,
                        "correct_answer_level": 1 if (1 if max_prob >= self.threshold else 0) == batch["gold_has_hallucination"][i] else 0,
                    }
                    predictions.append(sample)
        
        # Save to CSV
        with open(predictions_path, "w", newline="", encoding="utf-8") as f:
            if predictions:
                writer = csv.DictWriter(f, fieldnames=predictions[0].keys())
                writer.writeheader()
                writer.writerows(predictions)
        
        logger.info(f"Saved {len(predictions)} predictions to {predictions_path}")
        
        return metrics


def save_metrics(metrics: dict, output_path: Path):
    """Save metrics to JSON file."""
    import json
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {output_path}")
