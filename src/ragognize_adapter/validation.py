"""
Validation utilities for RAGognize adapter.
"""

import logging
from collections import Counter
from typing import Any

logger = logging.getLogger(__name__)


def validate_hallucination_span(
    span: dict,
    answer: str,
) -> tuple[bool, str | None]:
    """
    Validate a hallucination span against the answer.
    
    Args:
        span: Hallucination span dict with text, start, end.
        answer: The model's answer string.
    
    Returns:
        Tuple of (is_valid, error_message).
        - is_valid: True if span is valid.
        - error_message: None if valid, otherwise description of error.
    """
    if not isinstance(span, dict):
        return False, "span is not a dict"
    
    start = span.get("start")
    end = span.get("end")
    text = span.get("text")
    
    # Check required fields
    if start is None:
        return False, "missing start"
    if end is None:
        return False, "missing end"
    
    # Check types
    if not isinstance(start, int):
        return False, f"start is not int: {type(start)}"
    if not isinstance(end, int):
        return False, f"end is not int: {type(end)}"
    
    # Check range
    if start < 0:
        return False, f"start is negative: {start}"
    if end <= start:
        return False, f"end <= start: {end} <= {start}"
    if end > len(answer):
        return False, f"end > answer length: {end} > {len(answer)}"
    
    # Check text match
    if text is not None:
        extracted = answer[start:end]
        if extracted != text:
            # Try stripped match
            if extracted.strip() != text.strip():
                return False, f"text mismatch: expected '{text[:20]}...', got '{extracted[:20]}...'"
    
    return True, None


def validate_unified_sample(sample: dict) -> list[str]:
    """
    Validate a UnifiedSample dict.
    
    Args:
        sample: UnifiedSample as dictionary.
    
    Returns:
        List of validation errors (empty if valid).
    """
    errors = []
    
    # Required fields
    required_fields = [
        "case_id", "user_prompt_index", "question", "answer",
        "chunks", "hallucination_spans", "has_hallucination",
        "faithfulness_label", "source_model"
    ]
    
    for field in required_fields:
        if field not in sample:
            errors.append(f"missing required field: {field}")
    
    if errors:
        return errors
    
    # Check non-empty
    if not sample["question"].strip():
        errors.append("question is empty")
    
    if not sample["answer"].strip():
        errors.append("answer is empty")
    
    if not isinstance(sample["chunks"], list):
        errors.append("chunks is not a list")
    elif len(sample["chunks"]) == 0:
        errors.append("chunks is empty")
    
    # Check labels
    if sample["has_hallucination"] not in [0, 1]:
        errors.append(f"invalid has_hallucination: {sample['has_hallucination']}")
    
    if sample["faithfulness_label"] not in [0, 1]:
        errors.append(f"invalid faithfulness_label: {sample['faithfulness_label']}")
    
    # Check model name
    from .constants import SOURCE_MODELS
    if sample["source_model"] not in SOURCE_MODELS:
        errors.append(f"invalid source_model: {sample['source_model']}")
    
    # Validate hallucination spans
    answer = sample["answer"]
    for i, span in enumerate(sample.get("hallucination_spans", [])):
        is_valid, error = validate_hallucination_span(span, answer)
        if not is_valid:
            errors.append(f"span[{i}]: {error}")
    
    return errors


def validate_span_statistics(
    samples: list[dict],
) -> dict:
    """
    Compute statistics about hallucination spans.
    
    Args:
        samples: List of UnifiedSample dicts.
    
    Returns:
        Dictionary with span statistics.
    """
    stats = {
        "total_samples": len(samples),
        "samples_with_spans": 0,
        "total_spans": 0,
        "valid_spans": 0,
        "invalid_spans": 0,
        "span_validation": {
            "exact_match": 0,
            "stripped_match": 0,
            "mismatch": 0,
            "out_of_bounds": 0,
            "empty": 0,
        },
        "label_distribution": Counter(),
    }
    
    for sample in samples:
        answer = sample.get("answer", "")
        spans = sample.get("hallucination_spans", [])
        
        if len(spans) > 0:
            stats["samples_with_spans"] += 1
        
        stats["total_spans"] += len(spans)
        
        for span in spans:
            # Check valid field
            if span.get("valid", True):
                stats["valid_spans"] += 1
            else:
                stats["invalid_spans"] += 1
            
            # Validate against answer
            is_valid, _ = validate_hallucination_span(span, answer)
            if not is_valid:
                if "out of bounds" in str(_) or "end >" in str(_):
                    stats["span_validation"]["out_of_bounds"] += 1
                elif "mismatch" in str(_):
                    stats["span_validation"]["mismatch"] += 1
                elif "empty" in str(_):
                    stats["span_validation"]["empty"] += 1
            else:
                # Check text match
                extracted = answer[span["start"]:span["end"]]
                if span.get("text") == extracted:
                    stats["span_validation"]["exact_match"] += 1
                elif span.get("text", "").strip() == extracted.strip():
                    stats["span_validation"]["stripped_match"] += 1
        stats["label_distribution"][sample.get("has_hallucination", -1)] += 1
        stats["label_distribution"][sample.get("faithfulness_label", -1)] += 1
    
    stats["label_distribution"] = dict(stats["label_distribution"])
    
    return stats


def check_split_consistency(
    train_samples: list[dict],
    val_samples: list[dict],
    test_samples: list[dict],
) -> dict:
    """
    Check that splits don't have overlapping question indices.
    
    Args:
        train_samples: Training samples.
        val_samples: Validation samples.
        test_samples: Test samples.
    
    Returns:
        Dictionary with overlap check results.
    """
    train_indices = {s["user_prompt_index"] for s in train_samples}
    val_indices = {s["user_prompt_index"] for s in val_samples}
    test_indices = {s["user_prompt_index"] for s in test_samples}
    
    train_val_overlap = train_indices & val_indices
    train_test_overlap = train_indices & test_indices
    val_test_overlap = val_indices & test_indices
    
    return {
        "train_count": len(train_samples),
        "val_count": len(val_samples),
        "test_count": len(test_samples),
        "train_unique_prompts": len(train_indices),
        "val_unique_prompts": len(val_indices),
        "test_unique_prompts": len(test_indices),
        "train_val_overlap": len(train_val_overlap),
        "train_test_overlap": len(train_test_overlap),
        "val_test_overlap": len(val_test_overlap),
        "has_leakage": bool(train_val_overlap or train_test_overlap or val_test_overlap),
    }
