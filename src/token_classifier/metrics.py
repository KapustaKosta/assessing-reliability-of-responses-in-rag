"""
Metrics for token-level hallucination detection.

Computes metrics at three levels:
1. Token-level: Per-token classification
2. Span-level: Character span overlap metrics
3. Answer-level: Per-answer classification
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Legacy Adapter
# =============================================================================

def adapt_legacy_samples(samples_info: list[dict], threshold: float = 0.5) -> list[dict]:
    """
    Convert historical samples that use 'answer_preds' (legacy field) to 'preds'.

    This should be called BEFORE entering any metrics function when loading
    historical output files that contain 'answer_preds' but not 'preds'.

    Args:
        samples_info: List of sample dicts from a historical run.
        threshold: Threshold to apply to probs to produce binary preds.

    Returns:
        Same list with 'preds' field filled and 'answer_preds' left unchanged
        (backward-compatible).
    """
    result = []
    for s in samples_info:
        s = dict(s)  # shallow copy to avoid mutating caller's data
        if "preds" not in s or not s["preds"]:
            # Use answer_preds if present, else fall back to empty list
            raw = s.get("answer_preds") or []
            if s.get("probs"):
                s["preds"] = [(p >= threshold) for p in s["probs"]]
            else:
                s["preds"] = list(raw) if raw else []
        result.append(s)
    return result


# =============================================================================
# Token-level Metrics
# =============================================================================

def compute_token_metrics(
    y_true: list[int],
    y_pred: list[int],
    y_prob: Optional[list[float]] = None,
) -> dict:
    """
    Compute token-level classification metrics.
    
    Args:
        y_true: True labels (0=supported, 1=hallucinated)
        y_pred: Predicted labels
        y_prob: Predicted probabilities for hallucinated class
    
    Returns:
        Dictionary with metrics:
        - accuracy
        - positive_precision
        - positive_recall
        - positive_f1
        - macro_f1
        - confusion_matrix
        - support (positive count)
        - total_count
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    accuracy = accuracy_score(y_true, y_pred)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=[0, 1], zero_division=0
    )
    
    # Macro F1
    macro_f1 = (f1[0] + f1[1]) / 2
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    result = {
        "accuracy": float(accuracy),
        "positive_precision": float(precision[1]),
        "positive_recall": float(recall[1]),
        "positive_f1": float(f1[1]),
        "supported_precision": float(precision[0]),
        "supported_recall": float(recall[0]),
        "supported_f1": float(f1[0]),
        "macro_f1": float(macro_f1),
        "confusion_matrix": cm.tolist(),
        "support_positive": int(support[1]),
        "support_negative": int(support[0]),
        "total_count": len(y_true),
    }
    
    # AUC if probabilities provided
    if y_prob is not None:
        y_prob = np.array(y_prob)
        try:
            result["roc_auc"] = float(roc_auc_score(y_true, y_prob))
            result["pr_auc"] = float(average_precision_score(y_true, y_prob))
        except ValueError as e:
            logger.warning(f"Could not compute AUC: {e}")
    
    return result


# =============================================================================
# Span-level Metrics
# =============================================================================

def span_overlap_char_level(
    pred_span: tuple[int, int],
    gold_span: tuple[int, int],
) -> float:
    """
    Compute character-level overlap between predicted and gold spans.
    
    Uses intersection over union at character level.
    
    Args:
        pred_span: (start, end) of predicted span
        gold_span: (start, end) of gold span
    
    Returns:
        IoU score [0, 1]
    """
    pred_start, pred_end = pred_span
    gold_start, gold_end = gold_span
    
    # Intersection
    inter_start = max(pred_start, gold_start)
    inter_end = min(pred_end, gold_end)
    
    if inter_start >= inter_end:
        return 0.0
    
    intersection = inter_end - inter_start
    
    # Union
    union_start = min(pred_start, gold_start)
    union_end = max(pred_end, gold_end)
    union = union_end - union_start
    
    return intersection / union if union > 0 else 0.0


def spans_to_char_set(answer: str, spans: list[dict]) -> set[int]:
    """
    Convert spans to set of character indices.
    
    Args:
        answer: The answer text
        spans: List of {"start": int, "end": int}
    
    Returns:
        Set of character indices that are in any span
    """
    char_set = set()
    for span in spans:
        start = span.get("start", 0)
        end = span.get("end", 0)
        for i in range(start, end):
            if 0 <= i < len(answer):
                char_set.add(i)
    return char_set


def compute_span_metrics(
    answer: str,
    gold_spans: list[dict],
    pred_spans: list[dict],
    threshold: float = 0.5,
) -> dict:
    """
    Compute character span-level metrics.
    
    Compares predicted hallucination spans with gold spans.
    
    Args:
        answer: The answer text
        gold_spans: Gold hallucination spans [{"start": int, "end": int}, ...]
        pred_spans: Predicted hallucination spans [{"start": int, "end": int}, ...]
        threshold: IoU threshold for matching (unused in strict version)
    
    Returns:
        Dictionary with metrics:
        - character_precision
        - character_recall
        - character_f1
        - num_gold_chars
        - num_pred_chars
        - num_overlap_chars
    """
    # Convert to character sets
    gold_chars = spans_to_char_set(answer, gold_spans)
    pred_chars = spans_to_char_set(answer, pred_spans)
    
    # Compute overlap
    overlap = gold_chars & pred_chars
    total = gold_chars | pred_chars
    
    # Precision: overlap / pred
    precision = len(overlap) / len(pred_chars) if pred_chars else 0.0
    
    # Recall: overlap / gold
    recall = len(overlap) / len(gold_chars) if gold_chars else 0.0
    
    # F1
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    
    return {
        "character_precision": float(precision),
        "character_recall": float(recall),
        "character_f1": float(f1),
        "num_gold_chars": len(gold_chars),
        "num_pred_chars": len(pred_chars),
        "num_overlap_chars": len(overlap),
    }


def compute_sample_level_span_metrics(
    samples_info: list[dict],
    threshold: float = 0.5,
) -> dict:
    """
    Compute span-level metrics aggregated across samples.

    Args:
        samples_info: List of dicts with keys:
            - answer: str
            - gold_spans: list of {"start": int, "end": int}
            - probs: list of float p_hallucination per answer token (REQUIRED)
            - preds: list of int {0,1} per answer token (REQUIRED — filled by compute_all_metrics)
            - offsets: list of (start, end) tuples per answer token (REQUIRED)
        threshold: Threshold for hallucination prediction (unused; preds already computed)

    Returns:
        Aggregated span metrics:
        - character_precision
        - character_recall
        - character_f1
        - num_gold_chars (total)
        - num_pred_chars (total)
        - num_overlap_chars (total)

    Raises:
        ValueError: if probs/preds/offsets lengths mismatch or offsets out of answer range.
    """
    total_gold_chars = 0
    total_pred_chars = 0
    total_overlap_chars = 0

    for sample_idx, sample in enumerate(samples_info):
        answer   = sample.get("answer", "")
        gold_spans = sample.get("gold_spans", [])
        offsets   = sample.get("offsets", [])
        preds     = sample.get("preds", [])
        probs     = sample.get("probs", [])

        if not answer:
            raise ValueError(f"Sample {sample_idx} has empty answer")
        if not offsets:
            raise ValueError(f"Sample {sample_idx} has no offsets — cannot recover spans")

        # ── Assertions ──────────────────────────────────────────────────
        if len(probs) != len(preds):
            raise ValueError(
                f"Sample {sample_idx} len(probs)={len(probs)} != len(preds)={len(preds)}"
            )
        if len(preds) != len(offsets):
            raise ValueError(
                f"Sample {sample_idx} len(preds)={len(preds)} != len(offsets)={len(offsets)}"
            )
        for off_idx, (start, end) in enumerate(offsets):
            if not (0 <= start <= end <= len(answer)):
                raise ValueError(
                    f"Sample {sample_idx} offset[{off_idx}]=({start},{end}) "
                    f"out of range for answer of length {len(answer)}"
                )

        # Convert token predictions to predicted spans
        pred_spans = []
        for idx, pred in enumerate(preds):
            if pred == 1:  # hallucinated
                start, end = offsets[idx]
                if start is not None and end is not None:
                    pred_spans.append({"start": start, "end": end})

        # Merge adjacent spans
        merged_pred_spans = _merge_adjacent_spans(pred_spans)

        # Compute span metrics for this sample
        metrics = compute_span_metrics(answer, gold_spans, merged_pred_spans, threshold)

        total_gold_chars   += metrics.get("num_gold_chars", 0)
        total_pred_chars   += metrics.get("num_pred_chars", 0)
        total_overlap_chars += metrics.get("num_overlap_chars", 0)

    # Compute aggregated precision/recall
    precision = total_overlap_chars / total_pred_chars if total_pred_chars > 0 else 0.0
    recall    = total_overlap_chars / total_gold_chars if total_gold_chars > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "character_precision": float(precision),
        "character_recall":    float(recall),
        "character_f1":        float(f1),
        "num_gold_chars":       total_gold_chars,
        "num_pred_chars":       total_pred_chars,
        "num_overlap_chars":    total_overlap_chars,
    }


def _merge_adjacent_spans(spans: list[dict]) -> list[dict]:
    """Merge adjacent or overlapping spans."""
    if not spans:
        return []

    # Sort by start position
    sorted_spans = sorted(spans, key=lambda x: (x["start"], x["end"]))
    merged = [sorted_spans[0]]

    for span in sorted_spans[1:]:
        last = merged[-1]
        # If adjacent or overlapping, merge
        if span["start"] <= last["end"]:
            merged[-1] = {"start": last["start"], "end": max(last["end"], span["end"])}
        else:
            merged.append(span)

    return merged


# =============================================================================
# Answer-level Metrics
# =============================================================================

def compute_answer_metrics(
    answers: list[str],
    gold_labels: list[int],
    pred_labels: list[int],
    pred_probs: Optional[list[float]] = None,
) -> dict:
    """
    Compute answer-level classification metrics.
    
    An answer is predicted as hallucinated if any token is hallucinated.
    
    Args:
        answers: List of answer texts
        gold_labels: Gold answer-level labels (0=faithful, 1=hallucinated)
        pred_labels: Predicted answer-level labels
        pred_probs: Predicted answer-level probabilities (optional)
    
    Returns:
        Dictionary with metrics:
        - precision
        - recall
        - f1
        - accuracy
        - roc_auc (if probs provided and both classes present)
        - pr_auc (if probs provided and both classes present)
    """
    gold_labels = np.array(gold_labels)
    pred_labels = np.array(pred_labels)
    
    accuracy = accuracy_score(gold_labels, pred_labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        gold_labels, pred_labels, average="binary", zero_division=0
    )
    
    result = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "support_positive": int(support) if support is not None else 0,
    }
    
    # AUC if probabilities provided
    if pred_probs is not None:
        pred_probs = np.array(pred_probs)
        try:
            result["roc_auc"] = float(roc_auc_score(gold_labels, pred_probs))
            result["pr_auc"] = float(average_precision_score(gold_labels, pred_probs))
        except ValueError as e:
            logger.warning(f"Could not compute answer-level AUC: {e}")
    
    return result


# =============================================================================
# Calibration Metrics
# =============================================================================

def compute_calibration_metrics(
    y_true: list[int],
    y_prob: list[float],
    n_bins: int = 10,
) -> dict:
    """
    Compute calibration metrics for probability predictions.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        n_bins: Number of bins for reliability diagram
    
    Returns:
        Dictionary with:
        - brier_score
        - expected_calibration_error (ECE)
        - reliability_bins: list of (avg_prob, accuracy, count) per bin
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    # Brier score
    brier = brier_score_loss(y_true, y_prob)
    
    # ECE with equal-width bins
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    reliability_bins = []
    
    for i in range(n_bins):
        bin_start = bin_edges[i]
        bin_end = bin_edges[i + 1]
        
        # Samples in this bin
        in_bin = (y_prob >= bin_start) & (y_prob < bin_end)
        # Last bin includes upper edge
        if i == n_bins - 1:
            in_bin = (y_prob >= bin_start) & (y_prob <= bin_end)
        
        count = in_bin.sum()
        if count > 0:
            avg_prob = y_prob[in_bin].mean()
            accuracy = y_true[in_bin].mean()
            ece += (count / len(y_true)) * abs(avg_prob - accuracy)
            reliability_bins.append({
                "bin_start": float(bin_start),
                "bin_end": float(bin_end),
                "avg_prob": float(avg_prob),
                "accuracy": float(accuracy),
                "count": int(count),
            })
        else:
            reliability_bins.append({
                "bin_start": float(bin_start),
                "bin_end": float(bin_end),
                "avg_prob": None,
                "accuracy": None,
                "count": 0,
            })
    
    return {
        "brier_score": float(brier),
        "expected_calibration_error": float(ece),
        "reliability_bins": reliability_bins,
    }


# =============================================================================
# Batch Metrics
# =============================================================================

def compute_batch_metrics(
    labels: torch.Tensor,
    logits: torch.Tensor,
    answer_start_indices: list[int],
    answer_token_counts: list[int],
) -> dict:
    """
    Compute metrics for a batch.
    
    Args:
        labels: [batch, seq_len] token labels
        logits: [batch, seq_len, 2] model logits
        answer_start_indices: List of start indices for answer tokens
        answer_token_counts: List of answer token counts
    
    Returns:
        Dictionary with aggregated metrics
    """
    all_true = []
    all_pred = []
    all_prob = []
    
    batch_size = labels.shape[0]
    
    for i in range(batch_size):
        start_idx = answer_start_indices[i]
        count = answer_token_counts[i]
        
        if count <= 0:
            continue
        
        # Get answer tokens
        end_idx = start_idx + count
        answer_labels = labels[i, start_idx:end_idx].cpu().numpy()
        answer_logits = logits[i, start_idx:end_idx].cpu()
        
        # Compute probabilities
        probs = torch.softmax(answer_logits, dim=-1)[:, 1].numpy()
        
        # Predictions (threshold = 0.5)
        preds = (probs >= 0.5).astype(int)
        
        all_true.extend(answer_labels.tolist())
        all_pred.extend(preds.tolist())
        all_prob.extend(probs.tolist())
    
    if not all_true:
        return {}
    
    return compute_token_metrics(all_true, all_pred, all_prob)
