"""
Token Classification Model for RAGognize Hallucination Detection.

Architecture: Encoder -> Dropout -> Linear(hidden_size, 2)
- Uses pretrained encoder (e.g., distilbert-base-uncased, ModernBERT)
- Returns logits for each token
- Class 0: Supported, Class 1: Hallucinated
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Model
# =============================================================================

class TokenClassifier(nn.Module):
    """
    Token-level hallucination classifier.
    
    Architecture:
        Encoder (pretrained) -> Dropout -> Linear(hidden_size, 2)
    
    Training:
        - CrossEntropyLoss with ignore_index=-100
        - Only answer tokens contribute to loss
    """
    
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        dropout: float = 0.1,
        device: Optional[str] = None,
    ):
        """
        Initialize model.
        
        Args:
            model_name: HuggingFace model name
            dropout: Dropout rate
            device: Device to use (auto-detect if None)
        """
        super().__init__()
        self.model_name = model_name
        self.dropout_rate = dropout
        
        # Load encoder
        logger.info(f"Loading model: {model_name}")
        
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        
        # Classifier head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 2)
        
        # Detect device
        if device is None:
            device = self._detect_device()
        self._device = torch.device(device)
        
        logger.info(f"Model initialized. Hidden size: {hidden_size}, Device: {self._device}")
        
        # Count parameters
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        classifier_params = sum(p.numel() for p in self.classifier.parameters())
        logger.info(f"Parameters: encoder={encoder_params:,}, classifier={classifier_params:,}")
    
    def _detect_device(self) -> str:
        """Auto-detect the best available device."""
        # Check for NPU
        try:
            import torch_npu
            if torch_npu.is_available():
                return "npu:0"
        except ImportError:
            pass
        
        # Check for CUDA
        if torch.cuda.is_available():
            return "cuda:0"
        
        return "cpu"
    
    @property
    def device(self) -> torch.device:
        return self._device
    
    def to(self, device) -> "TokenClassifier":
        """Move model to device."""
        self._device = torch.device(device)
        return super().to(device)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            labels: Token labels [batch, seq_len], -100 for ignored
        
        Returns:
            dict with logits, loss, valid_token_count, positive_token_count
        """
        # Ensure inputs are on correct device
        input_ids = input_ids.to(self._device)
        attention_mask = attention_mask.to(self._device)
        
        # Encode
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        # Apply dropout
        dropped = self.dropout(hidden_states)
        
        # Classify
        logits = self.classifier(dropped)
        
        result = {"logits": logits}
        
        # Compute loss if labels provided
        if labels is not None:
            labels = labels.to(self._device)
            
            # Flatten
            logits_flat = logits.view(-1, 2)
            labels_flat = labels.view(-1)
            
            # Cross-entropy loss
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits_flat, labels_flat)
            
            # Count valid and positive tokens
            valid_mask = labels_flat != -100
            valid_count = valid_mask.sum().item()
            
            positive_mask = valid_mask & (labels_flat == 1)
            positive_count = positive_mask.sum().item()
            
            result["loss"] = loss
            result["valid_token_count"] = valid_count
            result["positive_token_count"] = positive_count
        
        return result
    
    def predict_proba(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Predict hallucination probabilities."""
        with torch.no_grad():
            input_ids = input_ids.to(self._device)
            attention_mask = attention_mask.to(self._device)
            
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state
            dropped = self.dropout(hidden_states)
            logits = self.classifier(dropped)
            
            probs = torch.softmax(logits, dim=-1)
            return probs[..., 1]  # [batch, seq_len]
    
    def save_pretrained(self, path: Path):
        """Save model checkpoint."""
        path.mkdir(parents=True, exist_ok=True)
        
        self.encoder.save_pretrained(path / "encoder")
        self.classifier.state_dict()
        torch.save({
            "model_name": self.model_name,
            "dropout": self.dropout_rate,
            "classifier_state": self.classifier.state_dict(),
        }, path / "classifier.pt")
        
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def from_pretrained(cls, path: Path, device: Optional[str] = None) -> "TokenClassifier":
        """Load model from checkpoint."""
        checkpoint = torch.load(path / "classifier.pt", map_location="cpu")
        
        model = cls(
            model_name=checkpoint["model_name"],
            dropout=checkpoint["dropout"],
            device=device,
        )
        
        model.classifier.load_state_dict(checkpoint["classifier_state"])
        
        logger.info(f"Model loaded from {path}")
        return model


# =============================================================================
# Device Management
# =============================================================================

def get_device_info() -> dict:
    """Get information about available devices."""
    info = {
        "torch_version": torch.__version__,
        "device": "cpu",  # Default
        "gpu_available": torch.cuda.is_available(),
        "npu_available": False,
    }
    
    if torch.cuda.is_available():
        info["device"] = "cuda:0"
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory"] = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    # Check NPU
    try:
        import torch_npu
        if torch_npu.is_available():
            info["device"] = "npu:0"
            info["npu_available"] = True
    except ImportError:
        pass
    
    return info
