"""
RAGognize Dataset Adapter

Provides unified interface for:
- Loading RAGognize data from local Parquet files
- Transforming to UnifiedSample format
- Token-level hallucination detection
- NLI-based faithfulness detection
"""

from .constants import (
    SOURCE_MODELS,
    GOLDEN_ANSWER_MODEL,
    ALL_MODELS,
    DEFAULT_SEED,
    DEFAULT_VAL_RATIO,
    PROJECT_ROOT,
    DATA_RAW_DIR,
    RESULTS_DIR,
)
from .loader import (
    load_ragognize_dataset,
    get_dataset_info,
    verify_required_fields,
)
from .inspection import (
    inspect_hallucinations_structure,
    comprehensive_response_inspection,
    analyze_span_validity,
)
from .adapter import (
    HallucinationSpan,
    UnifiedSample,
    RAGognizeAdapter,
    create_unified_dataset,
)
from .validation import (
    validate_hallucination_span,
    validate_unified_sample,
    validate_span_statistics,
    check_split_consistency,
)
from .splitting import (
    create_prompt_split,
    apply_split,
    save_split_manifest,
    save_split_summary,
    verify_no_overlap,
)

__all__ = [
    # Constants
    "SOURCE_MODELS",
    "GOLDEN_ANSWER_MODEL", 
    "ALL_MODELS",
    "DEFAULT_SEED",
    "DEFAULT_VAL_RATIO",
    "PROJECT_ROOT",
    "DATA_RAW_DIR",
    "RESULTS_DIR",
    # Loader
    "load_ragognize_dataset",
    "get_dataset_info",
    "verify_required_fields",
    # Inspection
    "inspect_hallucinations_structure",
    "comprehensive_response_inspection",
    "analyze_span_validity",
    # Adapter
    "HallucinationSpan",
    "UnifiedSample",
    "RAGognizeAdapter",
    "create_unified_dataset",
    # Validation
    "validate_hallucination_span",
    "validate_unified_sample",
    "validate_span_statistics",
    "check_split_consistency",
    # Splitting
    "create_prompt_split",
    "apply_split",
    "save_split_manifest",
    "save_split_summary",
    "verify_no_overlap",
]
