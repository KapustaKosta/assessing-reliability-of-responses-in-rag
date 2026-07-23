"""
Multiple Instance Learning (MIL) model for claim-level faithfulness.

Architecture:
- Encoder: pretrained NLI encoder (mDeBERTa-v3-base-mnli-xnli)
- Bag aggregation: max-pooling over context windows
- Output: binary classification (supported vs unsupported)

Label/Logit/Loss Convention (Phase 2 - Unified):
    label 0 = supported  (claim does NOT overlap hallucination span)
    label 1 = unsupported (claim overlaps hallucination span)

    The classifier outputs "unsupported_logit" directly:
        unsupported_logit > 0  -> model thinks "unsupported"  -> p_unsupported = sigmoid(unsupported_logit) > 0.5
        unsupported_logit < 0  -> model thinks "supported"    -> p_unsupported = sigmoid(unsupported_logit) < 0.5

    This is consistent across:
        - model.forward(): returns p_unsupported = sigmoid(unsupported_logit)
        - mil_forward_batch(): uses BCEWithLogitsLoss(unsupported_logit, unsupported_label)
        - evaluate.py: threshold applied to p_unsupported
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig

from claim_mil.claim_bags import ClaimBag

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class MILConfig:
    """Configuration for the MIL model."""
    encoder_name: str = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
    pooling_mode: str = "max"  # "max" or "log_sum_exp"
    log_sum_exp_temperature: float = 1.0
    dropout: float = 0.1
    hidden_size: Optional[int] = None

    def get_resolved_encoder_name(self) -> str:
        """
        Resolve encoder name to local path if CLAIM_MIL_MODEL_PATH is set.

        Priority:
        1. Use encoder_name if it's a valid directory
        2. Fall back to CLAIM_MIL_MODEL_PATH environment variable
        3. Return original encoder_name otherwise
        """
        import os
        from pathlib import Path

        # If encoder_name is a valid directory, use it
        if Path(self.encoder_name).is_dir():
            return self.encoder_name

        # Check CLAIM_MIL_MODEL_PATH environment variable
        env_path = os.environ.get("CLAIM_MIL_MODEL_PATH", "")
        if env_path and Path(env_path).is_dir():
            return env_path

        # Fall back to original name
        return self.encoder_name


# =============================================================================
# MIL Model
# =============================================================================

class ClaimMILModel(nn.Module):
    """
    MIL model for claim-level faithfulness.

    For each claim bag:
        1. Encode each (context_window, claim) pair with the encoder
        2. Aggregate window representations using max-pooling
        3. Predict unsupported probability via sigmoid

    Training: binary cross-entropy on claim bags
    """

    def __init__(self, config: MILConfig, tokenizer: Optional[AutoTokenizer] = None):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer  # stored separately

        # Resolve encoder name to local path if available
        resolved_encoder_name = config.get_resolved_encoder_name()

        # Load encoder config
        self.encoder_config = AutoConfig.from_pretrained(
            resolved_encoder_name,
            local_files_only=True,
        )
        hidden_size = (
            config.hidden_size
            if config.hidden_size is not None
            else self.encoder_config.hidden_size
        )

        # Use eager attention universally - stable across MPS/CPU/CUDA/NPU
        self.encoder = AutoModel.from_pretrained(
            resolved_encoder_name,
            attn_implementation="eager",
            local_files_only=True,
        )

        # Classification head
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(hidden_size, 1)

        self.pooling_mode = config.pooling_mode
        self.lse_temp = config.log_sum_exp_temperature

    def _encode_pairs(
        self,
        context_windows: list[str],
        claim_text: str,
    ) -> torch.Tensor:
        """
        Encode (context_window, claim_text) pairs.

        Returns:
            Tensor of shape (num_windows, hidden_size)
        """
        if not context_windows:
            raise ValueError("context_windows cannot be empty")

        tok = self.tokenizer
        if tok is None:
            raise ValueError("tokenizer must be provided to encode pairs")

        # Tokenize
        inputs = tok(
            context_windows,
            [claim_text] * len(context_windows),
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.encoder.device) for k, v in inputs.items()}

        # Encode
        outputs = self.encoder(**inputs)
        # Use [CLS] token representation
        hidden_states = outputs.last_hidden_state[:, 0, :]  # (batch, hidden)

        # Ensure float32 for classifier compatibility
        return hidden_states.to(torch.float32)

    def _aggregate_bag(self, window_repr: torch.Tensor) -> torch.Tensor:
        """
        Aggregate window representations into a bag representation.

        Args:
            window_repr: Tensor of shape (num_windows, hidden_size)

        Returns:
            Bag representation: (hidden_size,)
        """
        if self.pooling_mode == "max":
            return window_repr.max(dim=0).values
        elif self.pooling_mode == "log_sum_exp":
            # Smooth approximation to max: LSE(x) = (1/t) * log(sum(exp(t*x)))
            t = self.lse_temp
            return (1.0 / t) * torch.logsumexp(t * window_repr, dim=0)
        else:
            raise ValueError(f"Unknown pooling mode: {self.pooling_mode}")

    def forward(
        self,
        context_windows: list[str],
        claim_text: str,
        return_probs: bool = True,
    ) -> dict:
        """
        Forward pass for a single claim bag.

        Semantics (Phase 2 - Unified):
            unsupported_logit > 0  -> model thinks "unsupported"  -> p_unsupported > 0.5
            unsupported_logit < 0  -> model thinks "supported"    -> p_unsupported < 0.5

            p_unsupported = sigmoid(unsupported_logit)
            p_supported   = 1 - p_unsupported

        Args:
            context_windows: List of context window texts
            claim_text: The claim text (hypothesis)
            return_probs: If True, return probabilities

        Returns:
            dict with:
                - bag_repr: aggregated bag representation
                - unsupported_logit: raw logit for unsupported class (positive = unsupported)
                - p_unsupported: probability of unsupported (1)
                - p_supported: probability of supported (0)
        """
        window_repr = self._encode_pairs(context_windows, claim_text)
        bag_repr = self._aggregate_bag(window_repr)
        bag_repr = self.dropout(bag_repr)

        unsupported_logit = self.classifier(bag_repr.unsqueeze(0)).squeeze(-1)

        unsupported_logit_val = unsupported_logit.item() if unsupported_logit.numel() == 1 else unsupported_logit[0].item()
        p_unsupported = torch.sigmoid(unsupported_logit)
        p_supported = 1.0 - p_unsupported

        return {
            "bag_repr": bag_repr,
            "unsupported_logit": unsupported_logit_val,
            "p_supported": p_supported.mean().item() if p_supported.numel() > 1 else p_supported.item(),
            "p_unsupported": p_unsupported.mean().item() if p_unsupported.numel() > 1 else p_unsupported.item(),
        }

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def encode_windows(self, windows: list[str], claim: str) -> torch.Tensor:
        """Encode (context_window, claim) pairs, return hidden states."""
        return self._encode_pairs(windows, claim)

    def predict_bag(self, windows: list[str], claim: str) -> float:
        """Predict unsupported probability for a claim bag."""
        with torch.no_grad():
            result = self.forward(windows, claim, return_probs=True)
            return result["p_unsupported"]


# =============================================================================
# Batch inference helpers
# =============================================================================

@torch.no_grad()
def batch_predict(
    model: ClaimMILModel,
    bags: list[ClaimBag],
    batch_size: int = 8,
) -> list[dict]:
    """
    Run batch inference on a list of claim bags.

    Returns:
        List of dicts with p_unsupported, p_supported, support_logit for each bag
    """
    results = []
    for i in range(0, len(bags), batch_size):
        batch = bags[i:i + batch_size]
        batch_results = []

        for bag in batch:
            if not bag.context_windows:
                batch_results.append({
                    "p_unsupported": 0.5,
                    "p_supported": 0.5,
                    "unsupported_logit": 0.0,
                    "bag_repr": None,
                })
                continue

            window_texts = [w.window_text for w in bag.context_windows]
            result = model.forward(window_texts, bag.claim_text)
            batch_results.append(result)

        results.extend(batch_results)

    return results
