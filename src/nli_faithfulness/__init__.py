"""
NLI-based Faithfulness and Relevance Detection for RAG Reliability Assessment.

This package implements claim-level encoder/NLI methods to detect:
- Faithfulness: Whether claims are supported by retrieved context
- Relevance: Whether claims address the user's question
- Reliability: faithful AND relevant
"""

from .constants import (
    CHUNK_COLUMNS,
    DEFAULT_MODEL_NAME,
    LARGE_MODEL_NAME,
    DEVICE_PREFERENCE,
    CACHE_DIR,
    RESULTS_DIR,
    DEFAULT_WINDOW_OVERLAP_TOKENS,
    MIN_WINDOW_TOKENS,
    MAX_CHUNK_WINDOWS_PER_CHUNK,
)

from .segmentation import split_answer_into_units, segment_dataset, AnswerSegments, ClaimUnit
from .inference import NLIModel, batch_inference, ChunkWindow, NLIScore
from .aggregation import (
    apply_faithfulness_strategy,
    apply_relevance_strategy,
    compute_reliability,
    FAITHFULNESS_STRATEGIES,
    RELEVANCE_STRATEGIES,
)

__all__ = [
    # Constants
    "CHUNK_COLUMNS",
    "DEFAULT_MODEL_NAME",
    "LARGE_MODEL_NAME",
    "DEVICE_PREFERENCE",
    "CACHE_DIR",
    "RESULTS_DIR",
    "DEFAULT_WINDOW_OVERLAP_TOKENS",
    "MIN_WINDOW_TOKENS",
    "MAX_CHUNK_WINDOWS_PER_CHUNK",
    # Segmentation
    "split_answer_into_units",
    "segment_dataset",
    "AnswerSegments",
    "ClaimUnit",
    # Inference
    "NLIModel",
    "batch_inference",
    "ChunkWindow",
    "NLIScore",
    # Aggregation
    "apply_faithfulness_strategy",
    "apply_relevance_strategy",
    "compute_reliability",
    "FAITHFULNESS_STRATEGIES",
    "RELEVANCE_STRATEGIES",
]
