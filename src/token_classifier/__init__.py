"""
Token-level hallucination detection package (Scheme 3).

This package implements token-level classification for hallucination detection
in RAG systems. It is independent from Claim-MIL (Scheme 1/2).

Key differences from Claim-MIL:
- Operates at token level, not claim level
- Uses character spans for training supervision
- Outputs hallucination probability per token
- Supports context windowing with aggregation
"""

from .config import TokenClassifierConfig, get_model_path
from .schema import (
    TokenSample,
    UnifiedDataSchema,
    validate_span,
    create_grouped_split,
)
from .labeling import (
    compute_token_labels,
    span_overlaps_token,
)
from .model import TokenHallucinationClassifier
from .metrics import (
    compute_token_metrics,
    compute_span_metrics,
    compute_answer_metrics,
    compute_calibration_metrics,
)
from .postprocess import (
    tokens_to_spans,
    aggregate_window_probs,
)

__version__ = "0.1.0"

__all__ = [
    "TokenClassifierConfig",
    "get_model_path",
    "TokenSample",
    "UnifiedDataSchema",
    "validate_span",
    "create_grouped_split",
    "compute_token_labels",
    "span_overlaps_token",
    "TokenHallucinationClassifier",
    "compute_token_metrics",
    "compute_span_metrics",
    "compute_answer_metrics",
    "compute_calibration_metrics",
    "tokens_to_spans",
    "aggregate_window_probs",
]
