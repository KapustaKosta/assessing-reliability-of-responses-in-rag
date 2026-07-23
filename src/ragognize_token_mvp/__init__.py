"""
RAGognize Token-level Hallucination Detection MVP
Day 1 - Real data, real results

Key features:
- Uses real RAGognize hallucination spans
- Encoder-based token classification
- Character span to token label alignment
- Post-processing to recover character spans
"""

from .model import TokenClassifier, get_device_info
from .dataset import RAGognizeTokenDataset, load_ragognize_token_data, sample_balanced_subset, collate_fn
from .postprocess import span_from_tokens, merge_spans, PredictedSpan
from .trainer import TrainConfig, train_tiny_overfit, train_full
from .evaluator import Evaluator

__all__ = [
    "TokenClassifier",
    "get_device_info",
    "RAGognizeTokenDataset",
    "load_ragognize_token_data",
    "sample_balanced_subset",
    "collate_fn",
    "span_from_tokens",
    "merge_spans",
    "PredictedSpan",
    "Trainer",
    "TrainConfig",
    "Evaluator",
]
