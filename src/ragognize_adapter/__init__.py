"""
RAGognize Dataset Adapter.

Transforms F4biian/RAGognize dataset into unified format for faithfulness detection.
"""

from .adapter import (
    RAGognizeAdapter,
    RAGognizeSample,
    UnifiedSample,
    ModelResponse,
    HallucinationSpan,
    AVAILABLE_MODELS,
    load_ragognize_dataset,
    get_unified_dataset,
    get_dataset_stats,
    get_comprehensive_stats,
    run_adapter_tests,
    test_reproducibility,
    create_train_val_split,
    apply_split,
)

__all__ = [
    "RAGognizeAdapter",
    "RAGognizeSample",
    "UnifiedSample",
    "ModelResponse",
    "HallucinationSpan",
    "AVAILABLE_MODELS",
    "load_ragognize_dataset",
    "get_unified_dataset",
    "get_dataset_stats",
    "get_comprehensive_stats",
    "run_adapter_tests",
    "test_reproducibility",
    "create_train_val_split",
    "apply_split",
]
