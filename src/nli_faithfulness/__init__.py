"""
NLI-based Faithfulness Detection for RAG Reliability Assessment.

This package implements sentence-level evidence-aware NLI methods to detect
whether RAG-generated answers are faithful to the retrieved context chunks.
"""

from .constants import (
    CHUNK_COLUMNS,
    DEFAULT_MODEL_NAME,
    LARGE_MODEL_NAME,
    DEVICE_PREFERENCE,
    CACHE_DIR,
    RESULTS_DIR,
)
from .data import load_dataset, load_split
from .segmentation import split_answer_into_units
from .inference import NLIModel, batch_inference
from .aggregation import (
    aggregate_whole_answer_max_entail,
    aggregate_whole_answer_entail_minus_contrad,
    aggregate_sentence_min_support,
    aggregate_sentence_fraction_supported,
    aggregate_sentence_support_with_contradiction_penalty,
    AGGREGATION_STRATEGIES,
)
from .evaluation import (
    compute_metrics,
    find_best_threshold,
    evaluate_subgroups,
    compare_with_tfidf,
)

__all__ = [
    "CHUNK_COLUMNS",
    "DEFAULT_MODEL_NAME",
    "LARGE_MODEL_NAME",
    "DEVICE_PREFERENCE",
    "CACHE_DIR",
    "RESULTS_DIR",
    "load_dataset",
    "load_split",
    "split_answer_into_units",
    "NLIModel",
    "batch_inference",
    "aggregate_whole_answer_max_entail",
    "aggregate_whole_answer_entail_minus_contrad",
    "aggregate_sentence_min_support",
    "aggregate_sentence_fraction_supported",
    "aggregate_sentence_support_with_contradiction_penalty",
    "AGGREGATION_STRATEGIES",
    "compute_metrics",
    "find_best_threshold",
    "evaluate_subgroups",
    "compare_with_tfidf",
]
