"""
Data schema for token-level hallucination detection.

Unified schema for samples with hallucination spans.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Data Schema
# =============================================================================

@dataclass
class HallucinationSpan:
    """Represents a hallucinated span in the answer."""
    start: int
    end: int
    text: Optional[str] = None  # Optional, derived from answer if not provided
    valid: bool = True
    
    def __post_init__(self):
        if self.text is None:
            # Will be filled later with answer reference
            pass
    
    def to_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "valid": self.valid,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> HallucinationSpan:
        return cls(
            start=d.get("start", 0),
            end=d.get("end", 0),
            text=d.get("text"),
            valid=d.get("valid", True),
        )


@dataclass
class TokenSample:
    """A single sample for token-level classification."""
    sample_id: str
    question_id: str
    context: str
    question: str
    answer: str
    hallucination_spans: list[HallucinationSpan] = field(default_factory=list)
    split: str = "train"  # "train", "dev", "test"
    source_model: Optional[str] = None
    
    def __post_init__(self):
        if isinstance(self.hallucination_spans, list) and len(self.hallucination_spans) > 0:
            if isinstance(self.hallucination_spans[0], dict):
                self.hallucination_spans = [
                    HallucinationSpan.from_dict(s) for s in self.hallucination_spans
                ]
    
    @property
    def has_hallucinations(self) -> bool:
        """Check if there are any valid hallucinations."""
        return any(span.valid for span in self.hallucination_spans)
    
    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "question_id": self.question_id,
            "context": self.context,
            "question": self.question,
            "answer": self.answer,
            "hallucination_spans": [s.to_dict() for s in self.hallucination_spans],
            "split": self.split,
            "source_model": self.source_model,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> TokenSample:
        spans = d.get("hallucination_spans", [])
        if spans and isinstance(spans[0], dict):
            spans = [HallucinationSpan.from_dict(s) for s in spans]
        return cls(
            sample_id=d.get("sample_id", ""),
            question_id=d.get("question_id", d.get("user_prompt_index", "")),
            context=d.get("context", ""),
            question=d.get("question", d.get("user_prompt", "")),
            answer=d.get("answer", ""),
            hallucination_spans=spans,
            split=d.get("split", "train"),
            source_model=d.get("source_model"),
        )


# =============================================================================
# Validation
# =============================================================================

class ValidationMode:
    """Validation mode for data validation."""
    STRICT = "strict"
    LENIENT = "lenient"


def validate_span(
    span: HallucinationSpan,
    answer: str,
    mode: str = ValidationMode.STRICT,
) -> bool:
    """
    Validate a hallucination span.
    
    Rules:
    - start and end must be integers
    - 0 <= start < end <= len(answer)
    - Empty list [] means no hallucination (valid)
    
    Args:
        span: The hallucination span to validate
        answer: The answer text for reference
        mode: "strict" (error on invalid) or "lenient" (skip invalid)
    
    Returns:
        True if valid, False otherwise
    
    Raises:
        ValueError: In strict mode if span is invalid
    """
    answer_len = len(answer)
    
    # Check type
    if not isinstance(span.start, int):
        msg = f"span.start must be int, got {type(span.start).__name__}"
        if mode == ValidationMode.STRICT:
            raise ValueError(msg)
        logger.warning(msg)
        return False
    
    if not isinstance(span.end, int):
        msg = f"span.end must be int, got {type(span.end).__name__}"
        if mode == ValidationMode.STRICT:
            raise ValueError(msg)
        logger.warning(msg)
        return False
    
    # Check range
    if span.start < 0:
        msg = f"span.start must be >= 0, got {span.start}"
        if mode == ValidationMode.STRICT:
            raise ValueError(msg)
        logger.warning(msg)
        return False
    
    if span.end <= span.start:
        msg = f"span.end must be > span.start, got start={span.start}, end={span.end}"
        if mode == ValidationMode.STRICT:
            raise ValueError(msg)
        logger.warning(msg)
        return False
    
    if span.end > answer_len:
        msg = f"span.end must be <= answer length ({answer_len}), got {span.end}"
        if mode == ValidationMode.STRICT:
            raise ValueError(msg)
        logger.warning(msg)
        return False
    
    return True


def validate_sample(
    sample: TokenSample,
    mode: str = ValidationMode.STRICT,
) -> bool:
    """
    Validate a sample's hallucination spans.
    
    Args:
        sample: The sample to validate
        mode: "strict" or "lenient"
    
    Returns:
        True if all valid, False otherwise
    """
    all_valid = True
    for span in sample.hallucination_spans:
        if not validate_span(span, sample.answer, mode=mode):
            all_valid = False
    
    return all_valid


# =============================================================================
# Unified Data Schema Adapter
# =============================================================================

class UnifiedDataSchema:
    """
    Adapter to convert various data formats to UnifiedDataSchema.
    
    Supports:
    - RAGognize dataset format
    - CSV processed data
    - Direct dict format
    """
    
    # Mapping of possible field names to standard names
    CONTEXT_FIELDS = ["context", "chunks", "documents", "context_chunks"]
    QUESTION_FIELDS = ["question", "user_prompt", "prompt"]
    ANSWER_FIELDS = ["answer", "output", "response", "model_output"]
    SAMPLE_ID_FIELDS = ["sample_id", "case_id", "id", "row_id"]
    QUESTION_ID_FIELDS = ["question_id", "user_prompt_index", "prompt_index", "qid"]
    SOURCE_MODEL_FIELDS = ["source_model", "model", "model_name", "model_id"]
    SPAN_FIELDS = ["hallucination_spans", "spans", "h_spans", "hallucinations"]
    SPLIT_FIELDS = ["split", "dataset_split", "source_split"]
    
    @classmethod
    def _find_field(cls, d: dict, candidates: list[str]) -> Optional[str]:
        """Find the first matching field name in the dict."""
        for name in candidates:
            if name in d:
                return name
        return None
    
    @classmethod
    def _get_field(cls, d: dict, candidates: list[str], default=None):
        """Get field value from dict using candidate names."""
        key = cls._find_field(d, candidates)
        return d.get(key, default) if key else default
    
    @classmethod
    def _parse_spans(cls, raw_spans, answer: str, strict: bool = True) -> list[HallucinationSpan]:
        """Parse hallucination spans from various formats."""
        if not raw_spans:
            return []
        
        spans = []
        for raw in raw_spans:
            if isinstance(raw, dict):
                span = HallucinationSpan(
                    start=raw.get("start", 0),
                    end=raw.get("end", 0),
                    text=raw.get("text", answer[raw.get("start", 0):raw.get("end", 0)] if "start" in raw and "end" in raw else None),
                    valid=raw.get("valid", True),
                )
            elif isinstance(raw, (list, tuple)) and len(raw) >= 2:
                span = HallucinationSpan(
                    start=raw[0],
                    end=raw[1],
                    text=answer[raw[0]:raw[1]] if raw[0] < len(answer) and raw[1] <= len(answer) else None,
                    valid=True,
                )
            else:
                if strict:
                    raise ValueError(f"Invalid span format: {raw}")
                continue
            
            spans.append(span)
        
        return spans
    
    @classmethod
    def from_dict(
        cls,
        d: dict,
        strict: bool = False,
        log_selection: bool = True,
    ) -> Optional[TokenSample]:
        """
        Convert a dict to TokenSample.
        
        Args:
            d: Input dictionary
            strict: If True, raise on missing required fields
            log_selection: If True, log selected field names
        
        Returns:
            TokenSample or None if validation fails in lenient mode
        """
        # Find field names
        context_key = cls._find_field(d, cls.CONTEXT_FIELDS)
        question_key = cls._find_field(d, cls.QUESTION_FIELDS)
        answer_key = cls._find_field(d, cls.ANSWER_FIELDS)
        sample_id_key = cls._find_field(d, cls.SAMPLE_ID_FIELDS)
        question_id_key = cls._find_field(d, cls.QUESTION_ID_FIELDS)
        source_model_key = cls._find_field(d, cls.SOURCE_MODEL_FIELDS)
        spans_key = cls._find_field(d, cls.SPAN_FIELDS)
        split_key = cls._find_field(d, cls.SPLIT_FIELDS)
        
        if log_selection:
            logger.info(
                f"Field mapping: context={context_key}, question={question_key}, "
                f"answer={answer_key}, sample_id={sample_id_key}, "
                f"question_id={question_id_key}, source_model={source_model_key}, "
                f"spans={spans_key}, split={split_key}"
            )
        
        # Get values
        context = d.get(context_key) if context_key else None
        question = d.get(question_key) if question_key else ""
        answer = d.get(answer_key) if answer_key else ""
        sample_id = d.get(sample_id_key, "") if sample_id_key else ""
        question_id = d.get(question_id_key, 0) if question_id_key else 0
        source_model = d.get(source_model_key) if source_model_key else None
        raw_spans = d.get(spans_key, []) if spans_key else []
        split = d.get(split_key, "train") if split_key else "train"
        
        # Handle context from chunks
        if context is None:
            chunks = d.get("chunks", [])
            if isinstance(chunks, list):
                context = " ".join(str(c) for c in chunks if c)
        
        # Check required fields
        if not answer and strict:
            raise ValueError("answer field is required")
        
        # Parse spans
        spans = cls._parse_spans(raw_spans, answer or "", strict=strict)
        
        # Validate spans
        if answer:
            for span in spans:
                validate_span(span, answer, mode=ValidationMode.STRICT if strict else ValidationMode.LENIENT)
        
        return TokenSample(
            sample_id=str(sample_id),
            question_id=str(question_id),
            context=str(context) if context else "",
            question=str(question),
            answer=str(answer) if answer else "",
            hallucination_spans=spans,
            split=split,
            source_model=source_model,
        )
    
    @classmethod
    def from_ragognize(cls, d: dict, source_split: str = "") -> Optional[TokenSample]:
        """Convert from RAGognize dataset format."""
        # RAGognize format has specific fields
        user_prompt = d.get("user_prompt", "")
        responses = d.get("responses", {})
        documents = d.get("documents", [])
        
        # Get context from documents/chunks
        if isinstance(documents, list):
            context = " ".join(doc.get("text", "") if isinstance(doc, dict) else str(doc) for doc in documents)
        else:
            context = str(documents)
        
        # We need to expand each model response
        # This is a simplified version - in practice, this would be called per model
        answer = d.get("answer", "")
        if not answer:
            return None
        
        # Get sample_id from case_id or generate
        sample_id = d.get("case_id", d.get("source_row", ""))
        
        return TokenSample(
            sample_id=str(sample_id),
            question_id=str(d.get("user_prompt_index", 0)),
            context=context,
            question=user_prompt,
            answer=answer,
            hallucination_spans=[],
            split=source_split,
            source_model=d.get("source_model"),
        )


# =============================================================================
# Grouped Split
# =============================================================================

def create_grouped_split(
    samples: list[TokenSample],
    dev_fraction: float = 0.2,
    test_fraction: float = 0.0,
    seed: int = 42,
    project_val_question_ids: Optional[set] = None,
) -> dict:
    """
    Create grouped train/dev/test split.
    
    Grouping ensures all samples with the same question_id stay in the same split.
    This prevents data leakage where a model could learn question-specific patterns.
    
    Args:
        samples: List of samples
        dev_fraction: Fraction of questions for dev
        test_fraction: Fraction of questions for test (currently not used, test is fixed)
        seed: Random seed for reproducibility
        project_val_question_ids: Set of question IDs to exclude from train/dev
    
    Returns:
        Dictionary with:
        - train_samples, dev_samples, test_samples
        - train_question_ids, dev_question_ids, test_question_ids
        - sample counts
    """
    random.seed(seed)
    
    # Group by question_id
    question_to_samples = {}
    for sample in samples:
        qid = str(sample.question_id)
        if qid not in question_to_samples:
            question_to_samples[qid] = []
        question_to_samples[qid].append(sample)
    
    unique_questions = list(question_to_samples.keys())
    random.shuffle(unique_questions)
    
    # Exclude project validation questions
    if project_val_question_ids:
        project_val_question_ids = {str(q) for q in project_val_question_ids}
        unique_questions = [q for q in unique_questions if q not in project_val_question_ids]
    
    n_questions = len(unique_questions)
    n_dev = max(1, int(n_questions * dev_fraction))
    
    dev_questions = set(unique_questions[:n_dev])
    train_questions = set(unique_questions[n_dev:])
    
    # Partition samples
    train_samples = []
    dev_samples = []
    
    for qid, q_samples in question_to_samples.items():
        if qid in dev_questions:
            for s in q_samples:
                s.split = "dev"
            dev_samples.extend(q_samples)
        elif qid in train_questions:
            for s in q_samples:
                s.split = "train"
            train_samples.extend(q_samples)
        else:
            # Project validation - mark but don't include
            for s in q_samples:
                s.split = "val_excluded"
    
    return {
        "train_samples": train_samples,
        "dev_samples": dev_samples,
        "train_question_ids": train_questions,
        "dev_question_ids": dev_questions,
        "train_count": len(train_samples),
        "dev_count": len(dev_samples),
        "unique_train_questions": len(train_questions),
        "unique_dev_questions": len(dev_questions),
    }


def audit_split(
    samples: list[TokenSample],
) -> dict:
    """
    Audit a split for data leakage and quality.
    
    Args:
        samples: All samples after split
    
    Returns:
        Dictionary with audit results
    """
    splits = {"train": [], "dev": [], "test": []}
    for s in samples:
        if s.split in splits:
            splits[s.split].append(s)
    
    result = {
        "train_samples": len(splits["train"]),
        "dev_samples": len(splits["dev"]),
        "test_samples": len(splits["test"]),
    }
    
    # Unique question IDs per split
    train_qids = {str(s.question_id) for s in splits["train"]}
    dev_qids = {str(s.question_id) for s in splits["dev"]}
    test_qids = {str(s.question_id) for s in splits["test"]}
    
    result["unique_train_qids"] = len(train_qids)
    result["unique_dev_qids"] = len(dev_qids)
    result["unique_test_qids"] = len(test_qids)
    
    # Check for overlaps
    result["train_dev_overlap"] = len(train_qids & dev_qids)
    result["train_test_overlap"] = len(train_qids & test_qids)
    result["dev_test_overlap"] = len(dev_qids & test_qids)
    
    result["has_leakage"] = (
        result["train_dev_overlap"] > 0 or
        result["train_test_overlap"] > 0 or
        result["dev_test_overlap"] > 0
    )
    
    # Label distribution
    for split_name, split_samples in splits.items():
        if not split_samples:
            continue
        pos_count = sum(1 for s in split_samples if s.has_hallucinations)
        result[f"{split_name}_positive_rate"] = pos_count / len(split_samples) if split_samples else 0
    
    return result
