"""
Dataset utilities for token-level hallucination detection.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional, Literal

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from .schema import TokenSample, UnifiedDataSchema, create_grouped_split, audit_split, HallucinationSpan
from .labeling import AnswerTokenizer, compute_token_labels

logger = logging.getLogger(__name__)


# =============================================================================
# Dataset
# =============================================================================

class TokenClassificationDataset(Dataset):
    """
    PyTorch Dataset for token-level hallucination classification.
    
    Each sample is tokenized with three-part encoding:
        [CLS] Context [SEP] Question [SEP] Answer [SEP]
    
    Labels are only assigned to Answer tokens (other tokens have label=-100).
    """
    
    def __init__(
        self,
        samples: list[TokenSample],
        tokenizer,
        max_length: int = 512,
        context_stride: int = 128,
        context_max_length: int = 400,
        max_samples: Optional[int] = None,
    ):
        """
        Initialize dataset.
        
        Args:
            samples: List of TokenSample objects
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            context_stride: Stride for context windowing
            context_max_length: Maximum tokens for context
            max_samples: Maximum number of samples to use (for debugging)
        """
        self.samples = samples[:max_samples] if max_samples else samples
        self.tokenizer = tokenizer
        self.answer_tokenizer = AnswerTokenizer(
            tokenizer,
            max_length=max_length,
            context_max_length=context_max_length,
            context_stride=context_stride,
        )
        self.max_length = max_length
        
        # Pre-tokenize all samples
        self.tokenized_data = []
        for sample in self.samples:
            try:
                windows = self.answer_tokenizer.tokenize_sample_with_labels(sample)
                for window in windows:
                    self.tokenized_data.append({
                        "sample": sample,
                        "window": window,
                        "window_id": window["window_id"],
                    })
            except Exception as e:
                logger.warning(f"Failed to tokenize sample {sample.sample_id}: {e}")
    
    def __len__(self) -> int:
        return len(self.tokenized_data)
    
    def __getitem__(self, idx: int) -> dict:
        """Get a single item."""
        item = self.tokenized_data[idx]
        window = item["window"]
        
        # Pad or truncate input_ids
        input_ids = window["input_ids"]
        labels = window["labels"]
        
        # Truncate if necessary
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
        
        # Pad if necessary
        padding_length = self.max_length - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
            labels = labels + [-100] * padding_length
        
        attention_mask = [1 if tid != self.tokenizer.pad_token_id else 0 for tid in input_ids]
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "sample_id": item["sample"].sample_id,
            "window_id": window["window_id"],
            "answer_token_count": window["answer_token_count"],
            "answer_start_idx": window["answer_start_idx"],
            "answer_text": item["sample"].answer,  # For span metrics
            "gold_spans": [{"start": s.start, "end": s.end} for s in item["sample"].hallucination_spans if s.valid],  # Convert to dict list
            "answer_offsets": window.get("answer_offsets", []),  # For span metrics
        }
    
    def get_sample(self, idx: int) -> TokenSample:
        """Get the original sample for an index."""
        return self.tokenized_data[idx]["sample"]


# =============================================================================
# Data Loading
# =============================================================================

def load_data(
    data_path: str,
    strict: bool = False,
) -> list[TokenSample]:
    """
    Load data from various formats.
    
    Supported formats:
    - JSONL: One JSON object per line with TokenSample fields
    - JSON: Single JSON array with TokenSample objects
    - CSV: Rows with required columns
    
    Args:
        data_path: Path to data file
        strict: If True, raise on validation errors
    
    Returns:
        List of TokenSample objects
    """
    path = Path(data_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    samples = []
    schema = UnifiedDataSchema()
    
    if path.suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    sample = schema.from_dict(d, strict=strict)
                    if sample:
                        samples.append(sample)
                except Exception as e:
                    msg = f"Error parsing line {line_num}: {e}"
                    if strict:
                        raise ValueError(msg) from e
                    logger.warning(msg)
    
    elif path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if isinstance(data, list):
            for d in data:
                try:
                    sample = schema.from_dict(d, strict=strict)
                    if sample:
                        samples.append(sample)
                except Exception as e:
                    msg = f"Error parsing JSON object: {e}"
                    if strict:
                        raise ValueError(msg) from e
                    logger.warning(msg)
        else:
            sample = schema.from_dict(data, strict=strict)
            if sample:
                samples.append(sample)
    
    else:
        # Assume CSV
        import pandas as pd
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            try:
                d = row.to_dict()
                sample = schema.from_dict(d, strict=strict)
                if sample:
                    samples.append(sample)
            except Exception as e:
                msg = f"Error parsing CSV row: {e}"
                if strict:
                    raise ValueError(msg) from e
                logger.warning(msg)
    
    logger.info(f"Loaded {len(samples)} samples from {data_path}")
    return samples


def create_dataloaders(
    train_samples: list[TokenSample],
    dev_samples: list[TokenSample],
    tokenizer,
    batch_size: int = 8,
    max_length: int = 512,
    context_stride: int = 128,
    context_max_length: int = 400,
    num_workers: int = 0,
    max_train_samples: Optional[int] = None,
    max_dev_samples: Optional[int] = None,
) -> tuple[DataLoader, DataLoader]:
    """
    Create train and dev dataloaders.
    
    Args:
        train_samples: Training samples
        dev_samples: Development samples
        tokenizer: HuggingFace tokenizer
        batch_size: Batch size
        max_length: Maximum sequence length
        context_stride: Context window stride
        context_max_length: Maximum context length
        num_workers: Number of dataloader workers
        max_train_samples: Limit training samples
        max_dev_samples: Limit dev samples
    
    Returns:
        (train_loader, dev_loader)
    """
    train_dataset = TokenClassificationDataset(
        train_samples,
        tokenizer,
        max_length=max_length,
        context_stride=context_stride,
        context_max_length=context_max_length,
        max_samples=max_train_samples,
    )
    
    dev_dataset = TokenClassificationDataset(
        dev_samples,
        tokenizer,
        max_length=max_length,
        context_stride=context_stride,
        context_max_length=context_max_length,
        max_samples=max_dev_samples,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=_collate_fn,
    )
    
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_fn,
    )
    
    return train_loader, dev_loader


def _collate_fn(batch: list[dict]) -> dict:
    """Collate function for DataLoader."""
    result = {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
        "sample_ids": [item["sample_id"] for item in batch],
        "window_ids": [item["window_id"] for item in batch],
        "answer_token_counts": [item["answer_token_count"] for item in batch],
        "answer_start_indices": [item["answer_start_idx"] for item in batch],
        "answer_text": [item.get("answer_text", "") for item in batch],
        "gold_spans": [item.get("gold_spans", []) for item in batch],
        "answer_offsets": [item.get("answer_offsets", []) for item in batch],
    }
    return result
