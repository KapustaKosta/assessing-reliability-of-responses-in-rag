"""
Data loading utilities for NLI Faithfulness baseline.

Loads processed train/val/test splits and prepares them for NLI inference.
Does not modify the original DataFrames or CSV files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from .constants import (
    PROCESSED_DIR,
    CHUNK_COLUMNS,
    REQUIRED_COLUMNS,
)


@dataclass
class ChunkData:
    """Represents a single chunk with its metadata."""
    case_id: str
    chunk_id: int  # 1-8
    chunk_rank: int  # Same as chunk_id (preserves retrieval order)
    chunk_text: str
    retrieval_config: str  # "top_5" or "top_8"
    is_available: bool = True


@dataclass
class SampleData:
    """Represents a single sample with all its chunks."""
    case_id: str
    answer: str
    binary_faithfulness: bool
    binary_relevancy: bool
    question: str = ""  # User's question
    joint_label: str = ""  # "{relevancy}_{faithfulness}"
    chunks: list[ChunkData] = field(default_factory=list)
    
    @property
    def retrieval_config(self) -> str:
        """Get retrieval config from chunks."""
        available_chunks = [c for c in self.chunks if c.is_available]
        if available_chunks:
            return available_chunks[0].retrieval_config
        return "unknown"
    
    @property
    def chunk_count(self) -> int:
        """Number of available chunks."""
        return len([c for c in self.chunks if c.is_available])


@dataclass
class Dataset:
    """Container for a dataset split."""
    name: str  # "train", "val", "test"
    samples: list[SampleData]
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __iter__(self):
        return iter(self.samples)


def _extract_chunks(row: pd.Series) -> list[ChunkData]:
    """Extract available chunks from a DataFrame row."""
    chunks = []
    case_id = row["case_id"]
    retrieval_config = row["retrieval_config"]
    
    for i, col in enumerate(CHUNK_COLUMNS, start=1):
        chunk_text = row[col]
        
        # Skip missing chunks
        if pd.isna(chunk_text) or str(chunk_text).strip() == "":
            continue
            
        chunks.append(ChunkData(
            case_id=str(case_id),
            chunk_id=i,
            chunk_rank=i,
            chunk_text=str(chunk_text),
            retrieval_config=str(retrieval_config),
            is_available=True,
        ))
    
    return chunks


def load_split(split_name: str) -> Dataset:
    """
    Load a single data split.
    
    Args:
        split_name: One of "train", "val", "test"
        
    Returns:
        Dataset object containing all samples with their chunks
    """
    if split_name not in ("train", "val", "test"):
        raise ValueError(f"Invalid split name: {split_name}")
    
    csv_path = PROCESSED_DIR / f"{split_name}.csv"
    
    # Load only required columns
    df = pd.read_csv(csv_path, usecols=REQUIRED_COLUMNS)
    
    samples = []
    for _, row in df.iterrows():
        chunks = _extract_chunks(row)
        
        # Skip samples with no chunks (shouldn't happen but be safe)
        if not chunks:
            continue
        
        sample = SampleData(
            case_id=str(row["case_id"]),
            answer=str(row["answer"]) if pd.notna(row["answer"]) else "",
            binary_faithfulness=bool(row["binary_faithfulness"]),
            binary_relevancy=bool(row["binary_relevancy"]),
            joint_label=str(row["joint_label"]),
            chunks=chunks,
        )
        samples.append(sample)
    
    return Dataset(name=split_name, samples=samples)


def load_dataset() -> tuple[Dataset, Dataset, Dataset]:
    """
    Load all three data splits.
    
    Returns:
        Tuple of (train, val, test) datasets
    """
    train = load_split("train")
    val = load_split("val")
    test = load_split("test")
    
    return train, val, test


def get_stratified_sample(
    dataset: Dataset,
    n_per_class: int = 10,
    seed: int = 42,
) -> Dataset:
    """
    Create a stratified sample from a dataset for smoke testing.
    
    Ensures balanced representation of faithfulness=0 and faithfulness=1,
    and tries to include both top_5 and top_8 retrieval configs.
    """
    import random
    random.seed(seed)
    
    # Group by faithfulness
    faithful_samples = [s for s in dataset if s.binary_faithfulness]
    unfaithful_samples = [s for s in dataset if not s.binary_faithfulness]
    
    selected = []
    
    # Select n_per_class from each class
    for samples, target_label in [(unfaithful_samples, False), (faithful_samples, True)]:
        shuffled = samples.copy()
        random.shuffle(shuffled)
        
        # First, select up to n/2 from each of top_5 and top_8 within this class
        top_5 = [s for s in shuffled if s.retrieval_config == "top_5"]
        top_8 = [s for s in shuffled if s.retrieval_config == "top_8"]
        
        # Take half from each config
        n_half = n_per_class // 2
        
        selected_from_class = []
        selected_from_class.extend(top_5[:n_half])
        
        # Fill remaining slots from top_8 if needed
        remaining = n_per_class - len(selected_from_class)
        selected_from_class.extend(top_8[:remaining])
        
        # If still not enough, get more from remaining top_5
        if len(selected_from_class) < n_per_class:
            more_needed = n_per_class - len(selected_from_class)
            more_top5 = [s for s in top_5[n_half:] if s not in selected_from_class]
            selected_from_class.extend(more_top5[:more_needed])
        
        selected.extend(selected_from_class)
    
    return Dataset(name=f"{dataset.name}_sample", samples=selected)


def dataset_summary(dataset: Dataset) -> dict:
    """Generate a summary of a dataset."""
    total = len(dataset)
    if total == 0:
        return {"total": 0}
    
    faithful_count = sum(1 for s in dataset if s.binary_faithfulness)
    top_5_count = sum(1 for s in dataset if s.retrieval_config == "top_5")
    top_8_count = sum(1 for s in dataset if s.retrieval_config == "top_8")
    
    # Joint label distribution
    joint_labels = {}
    for s in dataset:
        joint_labels[s.joint_label] = joint_labels.get(s.joint_label, 0) + 1
    
    # Chunk count distribution
    chunk_counts = {}
    for s in dataset:
        n = s.chunk_count
        chunk_counts[n] = chunk_counts.get(n, 0) + 1
    
    return {
        "total": total,
        "faithful": faithful_count,
        "unfaithful": total - faithful_count,
        "faithfulness_positive_rate": faithful_count / total,
        "top_5": top_5_count,
        "top_8": top_8_count,
        "joint_labels": joint_labels,
        "chunk_counts": chunk_counts,
    }
