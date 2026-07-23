"""
Postprocessing utilities for token-level predictions.

Converts token predictions to character spans and aggregates across windows.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Token to Span Conversion
# =============================================================================

@dataclass
class TokenPrediction:
    """A single token prediction."""
    text: str
    start: int  # Character position in answer
    end: int    # Character position in answer (exclusive)
    p_hallucination: float
    predicted_label: int  # 0 = supported, 1 = hallucinated


def tokens_to_spans(
    tokens: list[TokenPrediction],
    threshold: float = 0.5,
    merge_gap: int = 0,
) -> list[dict]:
    """
    Convert token predictions to hallucination spans.
    
    Adjacent hallucinated tokens are merged into continuous spans.
    
    Args:
        tokens: List of token predictions
        threshold: Probability threshold for hallucination
        merge_gap: If > 0, merge spans separated by <= merge_gap characters
    
    Returns:
        List of hallucination spans:
        [{"start": int, "end": int, "text": str, "score_max": float, "score_mean": float}, ...]
    """
    if not tokens:
        return []
    
    # Apply threshold
    hallucinated_tokens = []
    for t in tokens:
        t.predicted_label = 1 if t.p_hallucination >= threshold else 0
        if t.predicted_label == 1:
            hallucinated_tokens.append(t)
    
    if not hallucinated_tokens:
        return []
    
    # Merge adjacent hallucinated tokens
    spans = []
    current_span = None
    
    for token in hallucinated_tokens:
        if current_span is None:
            current_span = {
                "start": token.start,
                "end": token.end,
                "text": token.text,
                "scores": [token.p_hallucination],
            }
        elif token.start <= current_span["end"]:  # Adjacent or overlapping
            current_span["end"] = max(current_span["end"], token.end)
            current_span["text"] = current_span["text"] + token.text
            current_span["scores"].append(token.p_hallucination)
        else:
            # Gap detected, save current and start new
            spans.append(current_span)
            current_span = {
                "start": token.start,
                "end": token.end,
                "text": token.text,
                "scores": [token.p_hallucination],
            }
    
    # Don't forget the last span
    if current_span is not None:
        spans.append(current_span)
    
    # Merge spans separated by small gaps
    if merge_gap > 0:
        merged_spans = []
        for span in spans:
            if not merged_spans:
                merged_spans.append(span)
            else:
                last = merged_spans[-1]
                if span["start"] - last["end"] <= merge_gap:
                    # Merge
                    last["end"] = span["end"]
                    last["text"] = last["text"] + span["text"]
                    last["scores"].extend(span["scores"])
                else:
                    merged_spans.append(span)
        spans = merged_spans
    
    # Compute aggregate scores
    result = []
    for span in spans:
        result.append({
            "start": span["start"],
            "end": span["end"],
            "text": span["text"],
            "score_max": max(span["scores"]),
            "score_mean": np.mean(span["scores"]),
        })
    
    return result


def extract_answer_tokens_from_offsets(
    answer: str,
    offsets: list[tuple[int, int]],
    probs: np.ndarray,
    threshold: float = 0.5,
) -> list[TokenPrediction]:
    """
    Extract token predictions from answer offsets.
    
    Args:
        answer: The answer text
        offsets: List of (start, end) character offsets for each token
        probs: Array of hallucination probabilities for each token
        threshold: Threshold for classification
    
    Returns:
        List of TokenPrediction objects
    """
    tokens = []
    for i, (start, end) in enumerate(offsets):
        if start == 0 and end == 0:
            # Special token, skip
            continue
        
        text = answer[start:end]
        tokens.append(TokenPrediction(
            text=text,
            start=start,
            end=end,
            p_hallucination=float(probs[i]),
            predicted_label=0,  # Will be set by tokens_to_spans
        ))
    
    return tokens


# =============================================================================
# Window Aggregation
# =============================================================================

def aggregate_window_probs(
    window_probs: list[np.ndarray],
    window_offsets: list[list[tuple[int, int]]],
    answer_length: int,
    mode: str = "max",
) -> np.ndarray:
    """
    Aggregate hallucination probabilities from multiple context windows.
    
    When context is too long, we use sliding windows. The same answer tokens
    may appear in multiple windows with different context. This function
    aggregates the probabilities.
    
    Args:
        window_probs: List of probability arrays, one per window
        window_offsets: List of offset arrays, matching window_probs
        answer_length: Length of the answer in characters
        mode: "max" or "mean" aggregation
    
    Returns:
        Aggregated probabilities for each answer token position
    """
    if len(window_probs) == 1:
        return window_probs[0]
    
    # For simplicity, just use max/mean across all windows for now
    # A more sophisticated version would handle offset alignment
    
    if mode == "max":
        stacked = np.stack(window_probs, axis=0)
        return np.max(stacked, axis=0)
    elif mode == "mean":
        stacked = np.stack(window_probs, axis=0)
        return np.mean(stacked, axis=0)
    else:
        raise ValueError(f"Unknown aggregation mode: {mode}")


def align_window_token_probs(
    answer: str,
    window_results: list[dict],
    mode: str = "max",
) -> list[float]:
    """
    Align and aggregate token probabilities from multiple windows.
    
    Each window result should have:
    - answer_offsets: Character offsets for answer tokens
    - answer_probs: Hallucination probabilities
    
    Args:
        answer: The answer text
        window_results: List of window results
        mode: Aggregation mode ("max" or "mean")
    
    Returns:
        List of aggregated probabilities per answer token
    """
    if not window_results:
        return []
    
    if len(window_results) == 1:
        return window_results[0].get("answer_probs", [])
    
    # Group by character position
    char_to_probs = {}
    
    for result in window_results:
        offsets = result.get("answer_offsets", [])
        probs = result.get("answer_probs", [])
        
        for i, (start, end) in enumerate(offsets):
            if start == 0 and end == 0:
                continue
            
            for char_pos in range(start, end):
                if 0 <= char_pos < len(answer):
                    if char_pos not in char_to_probs:
                        char_to_probs[char_pos] = []
                    char_to_probs[char_pos].append(probs[i] if i < len(probs) else 0)
    
    # Aggregate
    aggregated = []
    for char_pos in range(len(answer)):
        if char_pos in char_to_probs:
            probs = char_to_probs[char_pos]
            if mode == "max":
                aggregated.append(max(probs))
            else:
                aggregated.append(sum(probs) / len(probs))
        else:
            aggregated.append(0.0)
    
    return aggregated


# =============================================================================
# Answer-level Scoring
# =============================================================================

def compute_answer_score(
    token_probs: list[float],
    mode: str = "max",
) -> float:
    """
    Compute answer-level hallucination score from token probabilities.
    
    Modes:
    - "max": Maximum token probability
    - "noisy_or": 1 - prod(1 - p) for each token
    - "ratio": Ratio of hallucinated tokens (above threshold)
    
    Args:
        token_probs: Per-token hallucination probabilities
        mode: Scoring mode
    
    Returns:
        Answer-level score [0, 1]
    """
    if not token_probs:
        return 0.0
    
    probs = np.array(token_probs)
    
    if mode == "max":
        return float(np.max(probs))
    elif mode == "noisy_or":
        return float(1 - np.prod(1 - probs))
    elif mode == "ratio":
        return float(np.mean(probs >= 0.5))
    else:
        raise ValueError(f"Unknown answer score mode: {mode}")


def predict_answer_hallucination(
    token_probs: list[float],
    threshold: float = 0.5,
    mode: str = "any",
) -> bool:
    """
    Predict whether an answer contains hallucination.
    
    Modes:
    - "any": Any token above threshold
    - "majority": Majority of tokens above threshold
    - "all": All tokens above threshold
    
    Args:
        token_probs: Per-token hallucination probabilities
        threshold: Threshold for classification
        mode: Prediction mode
    
    Returns:
        True if answer is hallucinated
    """
    if not token_probs:
        return False
    
    probs = np.array(token_probs)
    above = np.sum(probs >= threshold)
    
    if mode == "any":
        return above >= 1
    elif mode == "majority":
        return above >= len(probs) / 2
    elif mode == "all":
        return above == len(probs)
    else:
        raise ValueError(f"Unknown prediction mode: {mode}")
