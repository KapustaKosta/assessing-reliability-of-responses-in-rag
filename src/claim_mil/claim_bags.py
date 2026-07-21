"""
Claim bag construction for supervised MIL faithfulness training.

This module builds claim bags from the raw RAGognize dataset:
- Segments answers into claims using the existing Stage 3 segmenter
- Assigns binary labels by checking overlap with hallucination spans
- Constructs context windows for each claim
- Creates train/dev split grouped by question_id

Label convention:
    unsupported = 1  (claim overlaps with a hallucination span)
    supported   = 0  (claim has no overlap)
"""

from __future__ import annotations

import sys
from pathlib import Path
_FILE = Path(__file__).resolve()
_SRC_DIR = _FILE.parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import hashlib
import logging
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path as PathType
from typing import Optional

import numpy as np
import pandas as pd

from ragognize_adapter import (
    RAGognizeAdapter,
    UnifiedSample,
    AVAILABLE_MODELS,
    load_ragognize_dataset,
    create_train_val_split,
    apply_split,
)
from nli_faithfulness.segmentation import split_answer_into_units, ClaimUnit
from nli_faithfulness.constants import (
    DEFAULT_WINDOW_OVERLAP_TOKENS,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ContextWindow:
    """A context window for a single claim."""
    chunk_id: int
    window_id: int
    window_text: str
    token_start: int
    token_end: int
    token_count: int


@dataclass
class ClaimBag:
    """
    A training unit: one claim with its context windows and label.

    A claim is unsupported (label=1) if it overlaps with ANY hallucination span.
    A claim is supported (label=0) if no span overlaps.

    The same context windows are reused from Stage 3 (no new windowing logic).
    """
    question: str
    answer: str
    claim_text: str
    claim_char_start: int
    claim_char_end: int
    context_windows: list[ContextWindow] = field(default_factory=list)
    claim_label: int = 0  # 0=supported, 1=unsupported

    # Identifiers
    question_id: int = 0
    expanded_sample_id: str = ""
    source_model: str = ""

    # For evaluation tracking
    gold_answer_faithful: bool = True  # canonical answer-level label from adapter


# =============================================================================
# Span Overlap Logic
# =============================================================================

def _compute_claim_label(
    claim: ClaimUnit,
    hallucination_spans: list,
    answer_text: str,
) -> tuple[int, str]:
    """
    Assign a binary label to a claim based on hallucination span overlap.

    Overlap rule: max(claim_start, span_start) < min(claim_end, span_end)

    Returns:
        (label, reason)
        label: 0=supported, 1=unsupported
        reason: human-readable reason for the label
    """
    if not hallucination_spans:
        return 0, "no_hallucination_spans"

    # Strip whitespace for offset comparison
    # Claims use non-whitespace character intervals
    claim_text_stripped = claim.text.strip()
    if not claim_text_stripped:
        return 0, "empty_claim"

    # Compute non-whitespace-relative offsets
    # Find where stripped text starts within the original claim text
    stripped_start_in_claim = len(claim.text) - len(claim.text.lstrip())
    stripped_end_in_claim = stripped_start_in_claim + len(claim_text_stripped)

    # Convert to answer-relative coordinates
    claim_ns_start = claim.char_start + stripped_start_in_claim
    claim_ns_end = claim.char_start + stripped_end_in_claim

    unsupported_count = 0
    overlap_details = []

    for span in hallucination_spans:
        span_start = span.get("start")
        span_end = span.get("end")
        span_text = span.get("text", "")
        span_valid = span.get("valid", True)

        # Skip invalid spans
        if not span_valid:
            continue

        # Validate span coordinates
        if not isinstance(span_start, (int, float)) or not isinstance(span_end, (int, float)):
            continue

        span_start = int(span_start)
        span_end = int(span_end)

        if span_start < 0 or span_end <= span_start:
            continue

        if span_end > len(answer_text):
            # Span extends beyond answer text - clamp or skip
            # Use the span's own text length as reference
            if span_text:
                expected_end = span_start + len(span_text)
                if span_end > expected_end:
                    span_end = expected_end

        # Check non-whitespace overlap
        # Overlap: max(ns_start, span_start) < min(ns_end, span_end)
        overlap_start = max(claim_ns_start, span_start)
        overlap_end = min(claim_ns_end, span_end)

        if overlap_start < overlap_end:
            # Non-whitespace character overlap exists
            unsupported_count += 1
            overlap_char_count = overlap_end - overlap_start
            overlap_details.append(
                f"overlap({overlap_char_count}chars) with span({span_start}-{span_end})"
            )

    if unsupported_count > 0:
        return 1, f"unsupported:{unsupported_count}spans({';'.join(overlap_details[:2])})"

    return 0, "no_overlap"


# =============================================================================
# Claim Bag Builder
# =============================================================================

class ClaimBagBuilder:
    """
    Builds claim bags from raw RAGognize data.

    Reuses Stage 3 segmentation and windowing logic.
    """

    def __init__(
        self,
        adapter: RAGognizeAdapter,
        tokenizer: Optional[object] = None,
        max_length: int = 512,
        overlap_tokens: int = DEFAULT_WINDOW_OVERLAP_TOKENS,
    ):
        self.adapter = adapter
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.overlap_tokens = overlap_tokens

    def build_windows(
        self,
        claim: ClaimUnit,
        chunks: list[str],
        max_windows_per_claim: int = 4,
    ) -> list[ContextWindow]:
        """
        Build context windows for a claim using the claim's text as hypothesis.

        This replicates Stage 3 windowing logic without running NLI inference.
        Returns window texts and token positions for MIL encoding.
        """
        if not self.tokenizer or not chunks:
            return []

        windows = []
        hyp_tokens = len(self.tokenizer.encode(
            claim.text, add_special_tokens=False
        )) + 10
        budget = self.max_length - hyp_tokens - 3

        if budget <= 0:
            return []

        for chunk_id, chunk_text in enumerate(chunks, start=1):
            tokens = self.tokenizer.encode(
                chunk_text,
                add_special_tokens=False,
                return_tensors="pt",
            ).squeeze()

            total_tokens = len(tokens)

            if total_tokens <= budget:
                windows.append(ContextWindow(
                    chunk_id=chunk_id,
                    window_id=0,
                    window_text=chunk_text,
                    token_start=0,
                    token_end=int(total_tokens),
                    token_count=int(total_tokens),
                ))
            else:
                stride = budget - self.overlap_tokens
                if stride <= 0:
                    stride = 32

                window_id = 0
                start = 0

                while start < total_tokens and len(windows) < max_windows_per_claim:
                    end = min(start + budget, total_tokens)
                    window_tokens = tokens[start:end]
                    window_text = self.tokenizer.decode(
                        window_tokens, skip_special_tokens=True
                    )

                    windows.append(ContextWindow(
                        chunk_id=chunk_id,
                        window_id=window_id,
                        window_text=window_text,
                        token_start=int(start),
                        token_end=int(end),
                        token_count=len(window_tokens),
                    ))

                    start += stride
                    window_id += 1

        return windows

    def sample_to_claim_bags(
        self,
        sample: UnifiedSample,
    ) -> tuple[list[ClaimBag], list[dict]]:
        """
        Convert a single UnifiedSample into claim bags.

        Returns:
            (claim_bags, skipped_reasons)
        """
        bags = []
        skipped = []

        if not sample.answer or not sample.answer.strip():
            skipped.append({
                "expanded_sample_id": sample.case_id,
                "source_model": sample.source_model,
                "question_id": sample.user_prompt_index,
                "reason": "empty_answer",
                "detail": "",
            })
            return bags, skipped

        # Segment answer into claims
        seg = split_answer_into_units(
            case_id=sample.case_id,
            answer=sample.answer,
        )

        hallucination_spans = sample.hallucination_spans

        # Convert hallucination_spans to dict format for overlap check
        hal_dicts = []
        for hs in hallucination_spans:
            if isinstance(hs, dict):
                hal_dicts.append(hs)
            else:
                hal_dicts.append(asdict(hs))

        for claim in seg.claims:
            label, reason = _compute_claim_label(
                claim=claim,
                hallucination_spans=hal_dicts,
                answer_text=sample.answer,
            )

            # Build context windows
            windows = self.build_windows(claim, sample.chunks)

            if not windows:
                skipped.append({
                    "expanded_sample_id": sample.case_id,
                    "source_model": sample.source_model,
                    "question_id": sample.user_prompt_index,
                    "claim_id": claim.claim_id,
                    "claim_text": claim.text[:50],
                    "reason": "no_context_windows",
                    "detail": f"chunks={len(sample.chunks)}",
                })

            bag = ClaimBag(
                question=sample.question,
                answer=sample.answer,
                claim_text=claim.text,
                claim_char_start=claim.char_start,
                claim_char_end=claim.char_end,
                context_windows=windows,
                claim_label=label,
                question_id=sample.user_prompt_index,
                expanded_sample_id=sample.case_id,
                source_model=sample.source_model,
                gold_answer_faithful=sample.faithfulness_label,
            )
            bags.append(bag)

        return bags, skipped


# =============================================================================
# Grouped Train/Dev Split
# =============================================================================

def create_grouped_split(
    samples: list[UnifiedSample],
    dev_fraction: float = 0.10,
    seed: int = 42,
    project_val_question_ids: Optional[set[int]] = None,
) -> dict:
    """
    Create a grouped internal train/dev split.

    All samples for the same question_id go to the same partition.
    Project validation questions are excluded from both partitions.

    Returns:
        dict with:
            - train_samples, dev_samples (list of UnifiedSample)
            - train_question_ids, dev_question_ids (set)
            - leakage_check dict
            - manifest dict
    """
    rng = random.Random(seed)

    # Collect all question_ids
    question_to_samples: dict[int, list[UnifiedSample]] = {}
    for s in samples:
        qid = s.user_prompt_index
        if qid not in question_to_samples:
            question_to_samples[qid] = []
        question_to_samples[qid].append(s)

    all_question_ids = sorted(question_to_samples.keys())

    # Exclude project validation questions
    if project_val_question_ids:
        available_qids = [q for q in all_question_ids if q not in project_val_question_ids]
        excluded_count = len(all_question_ids) - len(available_qids)
        logger.info(f"Excluded {excluded_count} project-validation questions")
    else:
        available_qids = all_question_ids

    rng.shuffle(available_qids)

    # Compute dev size
    n_dev = max(1, int(len(available_qids) * dev_fraction))
    dev_question_ids = set(available_qids[:n_dev])
    train_question_ids = set(available_qids[n_dev:])

    # Assemble samples
    train_samples = []
    dev_samples = []

    for s in samples:
        qid = s.user_prompt_index
        if qid in train_question_ids:
            train_samples.append(s)
        elif qid in dev_question_ids:
            dev_samples.append(s)
        # else: project validation, excluded

    # Count claim-level labels
    def count_claims(samples_list):
        supported = 0
        unsupported = 0
        skipped_empty = 0
        for s in samples_list:
            if not s.answer or not s.answer.strip():
                skipped_empty += 1
                continue
            # Quick estimate: count hallucinations vs total
            n_halls = len([h for h in s.hallucination_spans
                          if (isinstance(h, dict) and h.get('valid', True)) or
                             (not isinstance(h, dict) and getattr(h, 'valid', True))])
            n_claims_est = max(1, len(s.answer) // 50)  # rough estimate
            unsupported += min(n_halls, n_claims_est)
            supported += max(0, n_claims_est - min(n_halls, n_claims_est))
        return supported, unsupported, skipped_empty

    train_supported, train_unsupported, train_skipped = count_claims(train_samples)
    dev_supported, dev_unsupported, dev_skipped = count_claims(dev_samples)

    # Leakage checks
    leakage = {
        "train_dev_intersection": len(train_question_ids & dev_question_ids),
        "train_val_intersection": (
            len(train_question_ids & project_val_question_ids)
            if project_val_question_ids else 0
        ),
        "dev_val_intersection": (
            len(dev_question_ids & project_val_question_ids)
            if project_val_question_ids else 0
        ),
        "passed": (
            len(train_question_ids & dev_question_ids) == 0 and
            (not project_val_question_ids or
             len(train_question_ids & project_val_question_ids) == 0) and
            (not project_val_question_ids or
             len(dev_question_ids & project_val_question_ids) == 0)
        ),
    }

    # Manifest
    manifest = {
        "dev_fraction": dev_fraction,
        "seed": seed,
        "total_questions": len(all_question_ids),
        "available_questions": len(available_qids),
        "train_questions": len(train_question_ids),
        "dev_questions": len(dev_question_ids),
        "excluded_questions": len(all_question_ids) - len(available_qids),
        "train_samples": len(train_samples),
        "dev_samples": len(dev_samples),
        "train_supported_estimate": train_supported,
        "train_unsupported_estimate": train_unsupported,
        "dev_supported_estimate": dev_supported,
        "dev_unsupported_estimate": dev_unsupported,
        "leakage": leakage,
    }

    return {
        "train_samples": train_samples,
        "dev_samples": dev_samples,
        "train_question_ids": train_question_ids,
        "dev_question_ids": dev_question_ids,
        "leakage": leakage,
        "manifest": manifest,
    }


# =============================================================================
# Manifest Generation
# =============================================================================

def generate_split_manifest(
    samples: list[UnifiedSample],
    split_result: dict,
    output_path: PathType,
) -> pd.DataFrame:
    """
    Generate a split manifest CSV tracking claim-level and sample-level info.
    """
    records = []

    for s in samples:
        n_halls = len([h for h in s.hallucination_spans
                      if (isinstance(h, dict) and h.get('valid', True)) or
                         (not isinstance(h, dict) and getattr(h, 'valid', True))])
        seg = split_answer_into_units(case_id=s.case_id, answer=s.answer)
        n_claims = len(seg.claims)

        # Determine which partition
        partition = "unknown"
        if s.user_prompt_index in split_result["train_question_ids"]:
            partition = "train"
        elif s.user_prompt_index in split_result["dev_question_ids"]:
            partition = "dev"

        records.append({
            "question_id": s.user_prompt_index,
            "expanded_sample_id": s.case_id,
            "source_model": s.source_model,
            "source_split": s.source_split,
            "source_row_index": s.source_row_index,
            "partition": partition,
            "n_claims": n_claims,
            "n_hallucinations": n_halls,
            "gold_faithful": s.faithfulness_label,
            "answer_length": len(s.answer),
            "n_chunks": len(s.chunks),
        })

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    return df
