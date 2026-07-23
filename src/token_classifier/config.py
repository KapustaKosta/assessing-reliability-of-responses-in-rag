"""
Configuration for Token-level hallucination classifier.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Literal


def get_model_path(model_path: Optional[str] = None) -> str:
    """
    Get model path with priority:
    1. Explicit model_path parameter
    2. TOKEN_CLASSIFIER_MODEL_PATH environment variable
    3. CLAIM_MIL_MODEL_PATH environment variable (fallback)
    4. Raise error if none configured
    """
    if model_path and Path(model_path).is_dir():
        return model_path
    
    env_path = os.environ.get("TOKEN_CLASSIFIER_MODEL_PATH", "")
    if env_path and Path(env_path).is_dir():
        return env_path
    
    fallback_path = os.environ.get("CLAIM_MIL_MODEL_PATH", "")
    if fallback_path and Path(fallback_path).is_dir():
        return fallback_path
    
    raise ValueError(
        "Model path not configured. Please set one of:\n"
        "  1. --model_path argument\n"
        "  2. TOKEN_CLASSIFIER_MODEL_PATH environment variable\n"
        "  3. CLAIM_MIL_MODEL_PATH environment variable"
    )


@dataclass
class TokenClassifierConfig:
    """Configuration for token-level classifier."""
    
    # Model
    model_path: Optional[str] = None
    encoder_name: str = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
    dropout: float = 0.1
    hidden_size: Optional[int] = None
    
    # Tokenization
    max_length: int = 512
    context_stride: int = 128
    context_max_length: int = 400
    
    # Training
    device: Literal["auto", "cpu", "npu", "cuda"] = "auto"
    seed: int = 42
    epochs: int = 10
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Class weights
    positive_class_weight: Optional[float] = None  # None = auto, 0 = disabled
    
    # Aggregation
    context_aggregation: Literal["max", "mean"] = "max"
    
    # Early stopping
    patience: int = 5
    
    # Threshold tuning
    threshold_metric: Literal["token_f1", "span_f1", "answer_f1"] = "span_f1"
    threshold_search_min: float = 0.05
    threshold_search_max: float = 0.95
    threshold_search_step: float = 0.05
    
    def get_resolved_model_path(self) -> str:
        """Resolve model path with environment variables."""
        if self.model_path and Path(self.model_path).is_dir():
            return self.model_path
        
        env_path = os.environ.get("TOKEN_CLASSIFIER_MODEL_PATH", "")
        if env_path and Path(env_path).is_dir():
            return env_path
        
        fallback_path = os.environ.get("CLAIM_MIL_MODEL_PATH", "")
        if fallback_path and Path(fallback_path).is_dir():
            return fallback_path
        
        raise ValueError(
            "Model path not configured. Please set one of:\n"
            "  1. --model_path argument\n"
            "  2. TOKEN_CLASSIFIER_MODEL_PATH environment variable\n"
            "  3. CLAIM_MIL_MODEL_PATH environment variable"
        )
    
    def to_dict(self) -> dict:
        """Serialize to dict for checkpoint."""
        return {
            "model_path": self.model_path,
            "encoder_name": self.encoder_name,
            "dropout": self.dropout,
            "hidden_size": self.hidden_size,
            "max_length": self.max_length,
            "context_stride": self.context_stride,
            "context_max_length": self.context_max_length,
            "device": self.device,
            "seed": self.seed,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "max_grad_norm": self.max_grad_norm,
            "positive_class_weight": self.positive_class_weight,
            "context_aggregation": self.context_aggregation,
            "patience": self.patience,
            "threshold_metric": self.threshold_metric,
            "threshold_search_min": self.threshold_search_min,
            "threshold_search_max": self.threshold_search_max,
            "threshold_search_step": self.threshold_search_step,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> TokenClassifierConfig:
        """Deserialize from dict."""
        known_fields = {
            "model_path", "encoder_name", "dropout", "hidden_size",
            "max_length", "context_stride", "context_max_length", "device",
            "seed", "epochs", "batch_size", "gradient_accumulation_steps",
            "learning_rate", "weight_decay", "max_grad_norm",
            "positive_class_weight", "context_aggregation", "patience",
            "threshold_metric", "threshold_search_min", "threshold_search_max",
            "threshold_search_step",
        }
        config_dict = {k: v for k, v in d.items() if k in known_fields}
        return cls(**config_dict)
