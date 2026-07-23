"""
Checkpoint management for token classifier.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import torch

from .config import TokenClassifierConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Checkpoint
# =============================================================================

class CheckpointManager:
    """
    Manages model checkpoints.
    
    Saves:
    - encoder and classifier parameters
    - optimizer state
    - scheduler state
    - epoch/step
    - configuration
    - tokenizer/model path
    - labels semantics
    - thresholds
    - git commit hash
    """
    
    CHECKPOINT_KEYS = [
        "encoder_state_dict",
        "classifier_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "epoch",
        "step",
        "best_metric",
        "config",
    ]
    
    def __init__(self, results_dir: str, checkpoint_name: str = "best_checkpoint.pt"):
        """
        Initialize checkpoint manager.
        
        Args:
            results_dir: Directory to save checkpoints
            checkpoint_name: Name of checkpoint file
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.results_dir / checkpoint_name
        self.best_metric = None
    
    def save(
        self,
        model,
        optimizer,
        scheduler,
        epoch: int,
        step: int,
        best_metric: float,
        config: TokenClassifierConfig,
        metrics: Optional[dict] = None,
    ):
        """
        Save checkpoint.
        
        Args:
            model: TokenHallucinationClassifier
            optimizer: Optimizer
            scheduler: Learning rate scheduler (optional)
            epoch: Current epoch
            step: Current step
            best_metric: Best metric value
            config: Model configuration
            metrics: Additional metrics to save
        """
        # Get git commit hash if available
        git_hash = self._get_git_hash()
        
        checkpoint = {
            "encoder_state_dict": model.encoder.state_dict(),
            "classifier_state_dict": model.classifier.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "epoch": epoch,
            "step": step,
            "best_metric": best_metric,
            "config": config.to_dict(),
            "git_hash": git_hash,
            "metrics": metrics or {},
        }
        
        # Save
        torch.save(checkpoint, self.checkpoint_path)
        logger.info(f"Checkpoint saved to {self.checkpoint_path}")
        
        self.best_metric = best_metric
    
    def load(
        self,
        model,
        optimizer=None,
        scheduler=None,
    ) -> dict:
        """
        Load checkpoint.
        
        Args:
            model: Model to load weights into
            optimizer: Optimizer to load state into (optional)
            scheduler: Scheduler to load state into (optional)
        
        Returns:
            Checkpoint metadata (epoch, step, etc.)
        """
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        
        # Load model weights
        model.encoder.load_state_dict(checkpoint["encoder_state_dict"])
        model.classifier.load_state_dict(checkpoint["classifier_state_dict"])
        
        # Load optimizer state
        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Load scheduler state
        if scheduler and checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        self.best_metric = checkpoint.get("best_metric")
        
        logger.info(
            f"Checkpoint loaded: epoch={checkpoint['epoch']}, "
            f"step={checkpoint['step']}, best_metric={self.best_metric}"
        )
        
        return {
            "epoch": checkpoint["epoch"],
            "step": checkpoint["step"],
            "best_metric": checkpoint.get("best_metric"),
            "config": checkpoint.get("config"),
            "git_hash": checkpoint.get("git_hash"),
            "metrics": checkpoint.get("metrics", {}),
        }
    
    def load_config(self) -> TokenClassifierConfig:
        """Load configuration from checkpoint."""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        config_dict = checkpoint.get("config", {})
        
        return TokenClassifierConfig.from_dict(config_dict)
    
    def exists(self) -> bool:
        """Check if checkpoint exists."""
        return self.checkpoint_path.exists()
    
    @staticmethod
    def _get_git_hash() -> Optional[str]:
        """Get current git commit hash."""
        try:
            import subprocess
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent.parent,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None


# =============================================================================
# Config Management
# =============================================================================

def save_config(config: TokenClassifierConfig, results_dir: str):
    """Save configuration to JSON file."""
    config_path = Path(results_dir) / "config.json"
    
    with open(config_path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    
    logger.info(f"Config saved to {config_path}")


def load_config(results_dir: str) -> TokenClassifierConfig:
    """Load configuration from JSON file."""
    config_path = Path(results_dir) / "config.json"
    
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    
    return TokenClassifierConfig.from_dict(config_dict)
