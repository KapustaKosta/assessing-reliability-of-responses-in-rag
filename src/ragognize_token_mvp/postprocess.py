"""
Post-processing utilities for converting token predictions to character spans.

Key functions:
- Convert token probabilities to character probabilities
- Merge adjacent hallucinated tokens into spans
- Extract hallucination spans from token predictions
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class PredictedSpan:
    """A predicted hallucination span."""
    start: int  # Character start (inclusive)
    end: int    # Character end (exclusive)
    text: str   # The hallucinated text
    max_prob: float  # Maximum token probability in span
    mean_prob: float  # Mean token probability in span
    
    def to_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "max_prob": self.max_prob,
            "mean_prob": self.mean_prob,
        }


# =============================================================================
# Post-processing Functions
# =============================================================================

def span_from_tokens(
    token_probs: torch.Tensor,
    offset_mapping: list[list[tuple[int, int]]],
    answer_start: int,
    answer_end: int,
    answer_text: str,
    threshold: float = 0.5,
    merge_adjacent: bool = True,
) -> list[PredictedSpan]:
    """
    Convert token probabilities to character spans.
    
    Args:
        token_probs: [seq_len] tensor of hallucination probabilities
        offset_mapping: List of (char_start, char_end) for each token
        answer_start: Token index where answer starts
        answer_end: Token index where answer ends
        answer_text: Original answer text
        threshold: Probability threshold for hallucination
        merge_adjacent: Whether to merge adjacent hallucinated tokens
    
    Returns:
        List of PredictedSpan objects
    """
    spans = []
    
    if isinstance(token_probs, torch.Tensor):
        token_probs = token_probs.cpu().numpy()
    
    # Process only answer tokens
    hallucinated_token_indices = []
    hallucinated_token_probs = []
    
    for tok_idx in range(answer_start, min(answer_end, len(offset_mapping))):
        if tok_idx >= len(token_probs):
            break
        
        prob = token_probs[tok_idx]
        if prob >= threshold:
            hallucinated_token_indices.append(tok_idx)
            hallucinated_token_probs.append(prob)
    
    if not hallucinated_token_indices:
        return spans
    
    # Merge adjacent hallucinated tokens
    if merge_adjacent:
        merged_groups = _merge_adjacent_tokens(hallucinated_token_indices)
    else:
        merged_groups = [[idx] for idx in hallucinated_token_indices]
    
    # Convert token groups to character spans
    for group in merged_groups:
        if not group:
            continue
        
        # Get character bounds from token offsets
        char_starts = []
        char_ends = []
        probs = []
        
        for tok_idx in group:
            if tok_idx < len(offset_mapping):
                char_start, char_end = offset_mapping[tok_idx]
                char_starts.append(char_start)
                char_ends.append(char_end)
                probs.append(token_probs[tok_idx] if tok_idx < len(token_probs) else 0.0)
        
        if not char_starts:
            continue
        
        span_start = min(char_starts)
        span_end = max(char_ends)
        
        # Extract text
        span_text = answer_text[span_start:span_end] if span_start < len(answer_text) else ""
        
        spans.append(PredictedSpan(
            start=span_start,
            end=span_end,
            text=span_text,
            max_prob=max(probs) if probs else 0.0,
            mean_prob=np.mean(probs) if probs else 0.0,
        ))
    
    return spans


def _merge_adjacent_tokens(token_indices: list[int]) -> list[list[int]]:
    """
    Merge adjacent token indices into groups.
    
    Args:
        token_indices: List of token indices that are hallucinated
    
    Returns:
        List of groups of adjacent token indices
    """
    if not token_indices:
        return []
    
    token_indices = sorted(token_indices)
    groups = []
    current_group = [token_indices[0]]
    
    for idx in token_indices[1:]:
        if idx == current_group[-1] + 1:
            current_group.append(idx)
        else:
            groups.append(current_group)
            current_group = [idx]
    
    groups.append(current_group)
    return groups


def merge_spans(spans: list[dict], gap_threshold: int = 5) -> list[dict]:
    """
    Merge overlapping or nearby spans.
    
    Args:
        spans: List of span dicts with 'start' and 'end'
        gap_threshold: Maximum gap between spans to merge
    
    Returns:
        List of merged spans
    """
    if not spans:
        return []
    
    # Sort by start
    sorted_spans = sorted(spans, key=lambda x: x["start"])
    
    merged = [sorted_spans[0]]
    
    for span in sorted_spans[1:]:
        last = merged[-1]
        
        # Check if overlapping or within gap threshold
        if span["start"] <= last["end"] + gap_threshold:
            # Merge
            merged[-1] = {
                "start": last["start"],
                "end": max(last["end"], span["end"]),
                "text": "",  # Will be filled later
                "max_prob": max(last.get("max_prob", 1.0), span.get("max_prob", 1.0)),
            }
        else:
            merged.append(span)
    
    return merged


# =============================================================================
# Evaluation Metrics
# =============================================================================

def compute_token_metrics(
    gold_labels: list[int],
    pred_probs: list[float],
    threshold: float = 0.5,
) -> dict:
    """
    Compute token-level classification metrics.
    
    Args:
        gold_labels: Ground truth labels (0=supported, 1=hallucinated)
        pred_probs: Predicted probabilities for hallucinated class
        threshold: Classification threshold
    
    Returns:
        Dict with precision, recall, f1, accuracy
    """
    # Filter out ignored labels
    valid_pairs = [(g, p) for g, p in zip(gold_labels, pred_probs) if g != -100]
    
    if not valid_pairs:
        return {"precision": 0, "recall": 0, "f1": 0, "accuracy": 0}
    
    gold, probs = zip(*valid_pairs)
    preds = [1 if p >= threshold else 0 for p in probs]
    
    # Calculate metrics
    tp = sum(1 for g, p in zip(gold, preds) if g == 1 and p == 1)
    fp = sum(1 for g, p in zip(gold, preds) if g == 0 and p == 1)
    fn = sum(1 for g, p in zip(gold, preds) if g == 1 and p == 0)
    tn = sum(1 for g, p in zip(gold, preds) if g == 0 and p == 0)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / len(valid_pairs)
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def compute_span_metrics(
    gold_spans: list[tuple[int, int, str]],
    pred_spans: list[PredictedSpan],
) -> dict:
    """
    Compute character span-level metrics.
    
    Args:
        gold_spans: List of (start, end, text) tuples
        pred_spans: List of PredictedSpan objects
    
    Returns:
        Dict with precision, recall, f1
    """
    if not gold_spans and not pred_spans:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "char_overlap": 0}
    
    if not gold_spans:
        return {"precision": 0, "recall": 0, "f1": 0, "char_overlap": 0}
    
    if not pred_spans:
        return {"precision": 0, "recall": 0, "f1": 0, "char_overlap": 0}
    
    # Calculate character-level overlap
    gold_chars = set()
    for start, end, _ in gold_spans:
        gold_chars.update(range(start, end))
    
    pred_chars = set()
    for span in pred_spans:
        pred_chars.update(range(span.start, span.end))
    
    overlap = gold_chars & pred_chars
    
    if not gold_chars or not pred_chars:
        return {"precision": 0, "recall": 0, "f1": 0, "char_overlap": 0}
    
    precision = len(overlap) / len(pred_chars) if pred_chars else 0
    recall = len(overlap) / len(gold_chars) if gold_chars else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "char_overlap": len(overlap),
        "gold_chars": len(gold_chars),
        "pred_chars": len(pred_chars),
    }


def compute_answer_level_metrics(
    gold_has_hallucination: int,
    pred_spans: list[PredictedSpan],
    threshold: float = 0.5,
) -> dict:
    """
    Compute answer-level classification metrics.
    
    Args:
        gold_has_hallucination: 1 if answer has hallucination, 0 otherwise
        pred_spans: List of predicted spans
        threshold: Threshold for considering a span as hallucination
    
    Returns:
        Dict with prediction and correctness
    """
    # Answer is predicted as unfaithful if max probability >= threshold
    max_prob = max((s.max_prob for s in pred_spans), default=0.0)
    pred_has_hallucination = 1 if (pred_spans and max_prob >= threshold) else 0
    
    correct = 1 if pred_has_hallucination == gold_has_hallucination else 0
    
    return {
        "pred_has_hallucination": pred_has_hallucination,
        "max_prob": max_prob,
        "correct": correct,
        "n_predicted_spans": len(pred_spans),
    }
