"""
Token-level hallucination classifier model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Literal

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig

from .config import TokenClassifierConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Model
# =============================================================================

class TokenHallucinationClassifier(nn.Module):
    """
    Token-level hallucination classifier.
    
    Architecture:
        Encoder (pretrained) -> Dropout -> Linear(hidden_size, 2)
    
    Returns logits for each token, with:
        - logits[..., 0] = supported score
        - logits[..., 1] = hallucinated score
    
    Training:
        - CrossEntropyLoss with ignore_index=-100
        - Only answer tokens contribute to loss
        - Optional positive class weight for imbalanced data
    """
    
    def __init__(
        self,
        config: TokenClassifierConfig,
        tokenizer: Optional[AutoTokenizer] = None,
    ):
        """
        Initialize model.
        
        Args:
            config: Model configuration
            tokenizer: HuggingFace tokenizer (optional, stored for inference)
        """
        super().__init__()
        self.config = config
        
        # Resolve model path
        model_path = config.get_resolved_model_path()
        
        # Load encoder config
        self.encoder_config = AutoConfig.from_pretrained(
            model_path,
            local_files_only=True,
        )
        
        hidden_size = (
            config.hidden_size
            if config.hidden_size is not None
            else self.encoder_config.hidden_size
        )
        
        # Load encoder with eager attention for stability
        self.encoder = AutoModel.from_pretrained(
            model_path,
            attn_implementation="eager",
            local_files_only=True,
        )
        
        # Classifier head
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(hidden_size, 2)  # 2 classes: supported, hallucinated
        
        self._tokenizer = tokenizer
        
        # Count parameters
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        classifier_params = sum(p.numel() for p in self.classifier.parameters())
        logger.info(
            f"TokenHallucinationClassifier initialized: "
            f"encoder_params={encoder_params:,}, classifier_params={classifier_params:,}"
        )
    
    @property
    def tokenizer(self) -> Optional[AutoTokenizer]:
        return self._tokenizer
    
    @tokenizer.setter
    def tokenizer(self, value: AutoTokenizer):
        self._tokenizer = value
    
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
    
    def get_encoder_params(self) -> list[torch.nn.Parameter]:
        """Get encoder parameters for optional freezing."""
        return list(self.encoder.parameters())
    
    def get_classifier_params(self) -> list[torch.nn.Parameter]:
        """Get classifier parameters."""
        return list(self.classifier.parameters())
    
    def freeze_encoder(self):
        """Freeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        logger.info("Encoder frozen")
    
    def unfreeze_encoder(self):
        """Unfreeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        logger.info("Encoder unfrozen")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
    ) -> dict:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            labels: Token labels [batch, seq_len], -100 for ignored tokens
            return_hidden: If True, return hidden states
        
        Returns:
            dict with:
            - logits: [batch, seq_len, 2] raw logits
            - loss: Scalar loss (if labels provided)
            - valid_token_count: Number of valid (non-ignored) tokens
            - positive_token_count: Number of positive (hallucinated) tokens
            - hidden_states: [batch, seq_len, hidden] (if return_hidden)
        """
        # Encode
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden]
        
        # Apply dropout
        dropped = self.dropout(hidden_states)
        
        # Classify
        logits = self.classifier(dropped)  # [batch, seq_len, 2]
        
        result = {"logits": logits}
        
        # Compute loss if labels provided
        if labels is not None:
            # Reshape for cross-entropy: [batch * seq_len, 2] and [batch * seq_len]
            batch_size, seq_len, num_classes = logits.shape
            
            # Flatten
            logits_flat = logits.view(-1, num_classes)  # [batch * seq_len, 2]
            labels_flat = labels.view(-1)  # [batch * seq_len]
            
            # Compute loss with ignore_index=-100
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits_flat, labels_flat)
            
            # Count valid tokens
            valid_mask = labels_flat != -100
            valid_token_count = valid_mask.sum().item()
            
            # Count positive tokens (hallucinated = label 1)
            positive_mask = valid_mask & (labels_flat == 1)
            positive_token_count = positive_mask.sum().item()
            
            result["loss"] = loss
            result["valid_token_count"] = valid_token_count
            result["positive_token_count"] = positive_token_count
        
        if return_hidden:
            result["hidden_states"] = hidden_states
        
        return result
    
    def predict_proba(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict hallucination probabilities.
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
        
        Returns:
            p_hallucination: [batch, seq_len] probabilities for hallucinated class
        """
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            logits = outputs["logits"]
            
            # Softmax over classes, take class 1 (hallucinated)
            probs = torch.softmax(logits, dim=-1)
            return probs[..., 1]  # [batch, seq_len]


# =============================================================================
# Device Management
# =============================================================================

def get_device(device: str = "auto") -> torch.device:
    """
    Get torch device with auto-detection.
    
    Args:
        device: "auto", "cpu", "npu", or "cuda"
    
    Returns:
        torch.device
    """
    if device == "auto":
        # Try NPU first
        if torch.npu.is_available():
            return torch.device("npu:0")
        # Fall back to CPU
        return torch.device("cpu")
    
    if device == "npu":
        if torch.npu.is_available():
            return torch.device("npu:0")
        logger.warning("NPU not available, falling back to CPU")
        return torch.device("cpu")
    
    if device == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        logger.warning("CUDA not available, falling back to CPU")
        return torch.device("cpu")
    
    return torch.device(device)


def load_tokenizer_and_model(
    config: TokenClassifierConfig,
) -> tuple[AutoTokenizer, TokenHallucinationClassifier]:
    """
    Load tokenizer and model.
    
    Args:
        config: Model configuration
    
    Returns:
        (tokenizer, model)
    """
    model_path = config.get_resolved_model_path()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        use_safetensors=True,
    )
    
    # Create model
    model = TokenHallucinationClassifier(config, tokenizer=tokenizer)
    
    # Move to device
    device = get_device(config.device)
    model = model.to(device)
    
    return tokenizer, model


# =============================================================================
# Batch Training Helpers
# =============================================================================

def compute_loss_with_class_weights(
    logits: torch.Tensor,
    labels: torch.Tensor,
    positive_class_weight: Optional[float] = None,
) -> torch.Tensor:
    """
    Compute cross-entropy loss with optional class weights.
    
    Args:
        logits: [batch, seq_len, 2]
        labels: [batch, seq_len]
        positive_class_weight: Weight for positive (hallucinated) class
    
    Returns:
        Scalar loss
    """
    batch_size, seq_len, num_classes = logits.shape
    logits_flat = logits.view(-1, num_classes)
    labels_flat = labels.view(-1)
    
    if positive_class_weight and positive_class_weight > 0:
        # Weight for class 1 (hallucinated)
        weight = torch.tensor([1.0, positive_class_weight], device=logits.device)
        loss_fct = nn.CrossEntropyLoss(weight=weight, ignore_index=-100)
    else:
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    
    return loss_fct(logits_flat, labels_flat)


def check_nan_inf(tensor: torch.Tensor, name: str = "tensor") -> bool:
    """Check if tensor has NaN or Inf values."""
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    
    if has_nan or has_inf:
        logger.warning(f"{name} has NaN={has_nan}, Inf={has_inf}")
        return False
    return True


def compute_grad_norm(model: nn.Module, max_norm: float = 1.0) -> float:
    """Compute gradient norm and optionally clip."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    
    # Clip if needed
    if total_norm > max_norm:
        logger.info(f"Gradient norm {total_norm:.4f} > {max_norm}, clipping")
    
    return total_norm
