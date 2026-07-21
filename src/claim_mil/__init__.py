"""
Phase 2: Supervised Claim-level Faithfulness with Multiple-Instance Learning.

Components:
- claim_bags: Build claim bags from hallucination spans
- model: MIL model with max-pooling classifier
- train: Training CLI
- evaluate: Validation evaluation
"""

from .claim_bags import (
    ClaimBag,
    ClaimBagBuilder,
    ContextWindow,
    create_grouped_split,
    generate_split_manifest,
    _compute_claim_label,
)
from .model import ClaimMILModel, MILConfig, batch_predict

__all__ = [
    "ClaimBag",
    "ClaimBagBuilder",
    "ContextWindow",
    "create_grouped_split",
    "generate_split_manifest",
    "_compute_claim_label",
    "ClaimMILModel",
    "MILConfig",
    "batch_predict",
]
