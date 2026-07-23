"""
Dataset for RAGognize Token-level Hallucination Detection.

Handles:
- Loading RAGognize data via adapter
- Tokenization with offset mapping
- Label alignment (character spans -> token labels)
- Context truncation while preserving Answer
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Literal, NamedTuple

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from ragognize_adapter import (
    load_ragognize_dataset,
    create_prompt_split,
    apply_split,
    UnifiedSample,
    HallucinationSpan,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

SEPARATOR = " "

# Token indices
LABEL_IGNORE = -100
LABEL_SUPPORTED = 0
LABEL_HALLUCINATED = 1


# =============================================================================
# Tokenization Result
# =============================================================================

class TokenizationResult(NamedTuple):
    """Result of tokenizing a sample."""
    input_ids: list[int]
    attention_mask: list[int]
    offset_mapping: list[tuple[int, int]]
    labels: list[int]
    answer_offset_start: int  # Token index where answer starts
    answer_offset_end: int    # Token index where answer ends
    context_tokens: int
    question_tokens: int
    answer_tokens: int


# =============================================================================
# Dataset
# =============================================================================

class RAGognizeTokenDataset(Dataset):
    """
    PyTorch Dataset for RAGognize token-level hallucination detection.
    
    Input format:
        [CLS] context [SEP] question [SEP] answer [SEP]
    
    Labels:
        - Answer tokens: 0 (supported) or 1 (hallucinated)
        - Other tokens: -100 (ignored in loss)
    
    Truncation strategy:
        - Preserve full Answer and Question
        - Truncate Context from the left if needed
    """
    
    def __init__(
        self,
        samples: list[UnifiedSample],
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        answer_window_chars: int = 2000,
    ):
        """
        Initialize dataset.
        
        Args:
            samples: List of UnifiedSample objects
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            answer_window_chars: Max characters to process for answer
        """
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Pre-process all samples
        self.tokenized = []
        self.stats = {
            "total": 0,
            "truncated_context": 0,
            "truncated_answer": 0,
            "has_hallucination": 0,
            "no_hallucination": 0,
            "total_tokens": 0,
            "total_labels_1": 0,
        }
        
        for i, sample in enumerate(samples):
            try:
                result = self._tokenize_sample(sample, answer_window_chars)
                self.tokenized.append({
                    "sample": sample,
                    "result": result,
                    "sample_idx": i,
                })
                self.stats["total"] += 1
                
                if result.labels.count(LABEL_HALLUCINATED) > 0:
                    self.stats["has_hallucination"] += 1
                else:
                    self.stats["no_hallucination"] += 1
                
                self.stats["total_tokens"] += len(result.labels)
                self.stats["total_labels_1"] += result.labels.count(LABEL_HALLUCINATED)
                
            except Exception as e:
                logger.warning(f"Failed to tokenize sample {sample.case_id}: {e}")
        
        logger.info(f"Dataset: {self.stats}")
    
    def _tokenize_sample(
        self,
        sample: UnifiedSample,
        answer_window_chars: int,
    ) -> TokenizationResult:
        """
        Tokenize a single sample with label alignment.
        
        Args:
            sample: UnifiedSample with hallucination spans
            answer_window_chars: Max chars from answer to process
        
        Returns:
            TokenizationResult with tokenized data and labels
        """
        # Prepare text parts
        context_text = SEPARATOR.join(sample.chunks) if sample.chunks else ""
        question_text = sample.question
        answer_text = sample.answer
        
        # Truncate answer if too long
        original_answer_len = len(answer_text)
        if len(answer_text) > answer_window_chars:
            answer_text = answer_text[:answer_window_chars]
        
        # Adjust hallucination spans to truncated answer
        adjusted_spans = []
        for span in sample.hallucination_spans:
            if span.start < answer_window_chars:
                end = min(span.end, answer_window_chars)
                adjusted_spans.append({
                    "start": span.start,
                    "end": end,
                })
        
        # Tokenize using the correct API
        # Encode each part separately to get offset mappings
        context_enc = self.tokenizer(
            context_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        question_enc = self.tokenizer(
            question_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        answer_enc = self.tokenizer(
            answer_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        
        context_ids = context_enc["input_ids"]
        context_offsets = context_enc["offset_mapping"]
        
        question_ids = question_enc["input_ids"]
        question_offsets = question_enc["offset_mapping"]
        
        answer_ids = answer_enc["input_ids"]
        answer_offsets = answer_enc["offset_mapping"]
        
        # Build full sequence: [CLS] context [SEP] question [SEP] answer [SEP]
        full_ids = [self.tokenizer.cls_token_id]
        full_mask = [1]
        full_offsets = [(0, 0)]  # CLS token
        
        # Context
        full_ids.extend(context_ids)
        full_mask.extend([1] * len(context_ids))
        full_offsets.extend(context_offsets)
        
        # [SEP]
        full_ids.append(self.tokenizer.sep_token_id)
        full_mask.append(1)
        full_offsets.append((0, 0))
        
        # Question
        question_start_token = len(full_ids)
        full_ids.extend(question_ids)
        full_mask.extend([1] * len(question_ids))
        full_offsets.extend(question_offsets)
        
        # [SEP]
        full_ids.append(self.tokenizer.sep_token_id)
        full_mask.append(1)
        full_offsets.append((0, 0))
        
        # Answer
        answer_start_token = len(full_ids)
        answer_offset_start = answer_start_token
        full_ids.extend(answer_ids)
        full_mask.extend([1] * len(answer_ids))
        full_offsets.extend(answer_offsets)
        answer_offset_end = len(full_ids)
        
        # [SEP]
        full_ids.append(self.tokenizer.sep_token_id)
        full_mask.append(1)
        full_offsets.append((0, 0))
        
        # Create labels (only for answer tokens)
        labels = [-100] * len(full_ids)
        
        for tok_idx in range(answer_offset_start, answer_offset_end):
            if tok_idx >= len(full_offsets):
                continue
                
            char_start, char_end = full_offsets[tok_idx]
            
            # Check overlap with any hallucination span
            is_hallucinated = False
            for span in adjusted_spans:
                # Character overlap: max(start1, start2) < min(end1, end2)
                if max(char_start, span["start"]) < min(char_end if char_end > 0 else char_start + 1, span["end"]):
                    is_hallucinated = True
                    break
            
            labels[tok_idx] = LABEL_HALLUCINATED if is_hallucinated else LABEL_SUPPORTED
        
        return TokenizationResult(
            input_ids=full_ids,
            attention_mask=full_mask,
            offset_mapping=full_offsets,
            labels=labels,
            answer_offset_start=answer_offset_start,
            answer_offset_end=answer_offset_end,
            context_tokens=len(context_ids),
            question_tokens=len(question_ids),
            answer_tokens=len(answer_ids),
        )
    
    def __len__(self) -> int:
        return len(self.tokenized)
    
    def __getitem__(self, idx: int) -> dict:
        """Get a single item."""
        item = self.tokenized[idx]
        result = item["result"]
        sample = item["sample"]
        
        # Truncate if needed
        input_ids = result.input_ids[:self.max_length]
        attention_mask = result.attention_mask[:self.max_length]
        labels = result.labels[:self.max_length]
        offset_mapping = result.offset_mapping[:self.max_length]
        
        # Pad if needed
        pad_len = self.max_length - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
            attention_mask = attention_mask + [0] * pad_len
            labels = labels + [-100] * pad_len
            offset_mapping = offset_mapping + [(0, 0)] * pad_len
        
        # Adjust answer positions for truncation
        answer_start = min(result.answer_offset_start, self.max_length)
        answer_end = min(result.answer_offset_end, self.max_length)
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "sample_idx": torch.tensor(item["sample_idx"], dtype=torch.long),
            "answer_start": torch.tensor(answer_start, dtype=torch.long),
            "answer_end": torch.tensor(answer_end, dtype=torch.long),
            "offset_mapping": offset_mapping,
            # Metadata for evaluation
            "case_id": sample.case_id,
            "source_model": sample.source_model,
            "question": sample.question[:200],
            "answer": sample.answer,
            "gold_spans": [(s.start, s.end, s.text) for s in sample.hallucination_spans],
            "gold_has_hallucination": sample.has_hallucination,
            "gold_faithfulness": sample.faithfulness_label,
            "answer_text": sample.answer[:2000],
        }


def collate_fn(batch: list[dict]) -> dict:
    """Collate function for DataLoader."""
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "sample_idx": torch.stack([b["sample_idx"] for b in batch]),
        "answer_start": torch.stack([b["answer_start"] for b in batch]),
        "answer_end": torch.stack([b["answer_end"] for b in batch]),
        # Metadata (keep as lists)
        "case_id": [b["case_id"] for b in batch],
        "source_model": [b["source_model"] for b in batch],
        "question": [b["question"] for b in batch],
        "answer": [b["answer"] for b in batch],
        "gold_spans": [b["gold_spans"] for b in batch],
        "gold_has_hallucination": [b["gold_has_hallucination"] for b in batch],
        "gold_faithfulness": [b["gold_faithfulness"] for b in batch],
        "answer_text": [b["answer_text"] for b in batch],
    }


def load_ragognize_token_data(
    data_dir: Optional[Path] = None,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, list[UnifiedSample]]:
    """
    Load and split RAGognize data for token-level training.
    """
    random.seed(seed)
    
    # Load dataset
    dataset = load_ragognize_dataset(data_dir=data_dir)
    
    # Create split
    split_info = create_prompt_split(dataset, val_ratio=val_ratio, seed=seed)
    
    # Apply split
    expanded = apply_split(dataset, split_info)
    
    return expanded


def sample_balanced_subset(
    samples: list[UnifiedSample],
    n_positive: int,
    n_negative: int,
    seed: int = 42,
) -> list[UnifiedSample]:
    """
    Sample a balanced subset for quick experiments.
    """
    random.seed(seed)
    
    positive = [s for s in samples if s.has_hallucination == 1]
    negative = [s for s in samples if s.has_hallucination == 0]
    
    random.shuffle(positive)
    random.shuffle(negative)
    
    selected = positive[:n_positive] + negative[:n_negative]
    random.shuffle(selected)
    
    return selected
