"""
Training loop for RAGognize Token-level Hallucination Detection.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from ragognize_token_mvp.model import TokenClassifier, get_device_info
from ragognize_token_mvp.dataset import RAGognizeTokenDataset, collate_fn

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Training configuration."""
    model_name: str = "distilbert-base-uncased"
    max_length: int = 512
    batch_size: int = 8
    gradient_accumulation: int = 2
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_epochs: int = 1
    max_steps: int = 150  # For tiny overfit
    gradient_clip: float = 1.0
    seed: int = 42
    device: str = "auto"
    eval_every: int = 10  # Steps


@dataclass
class TrainingResult:
    """Result of training."""
    best_step: int
    best_token_f1: float
    best_span_f1: float
    final_step: int
    final_loss: float
    train_time: float
    checkpoint_path: Path
    history: list[dict] = field(default_factory=list)


def train_tiny_overfit(
    model: TokenClassifier,
    train_loader: DataLoader,
    config: TrainConfig,
    output_dir: Path,
) -> TrainingResult:
    """
    Train model on small dataset (tiny overfit test).
    
    Args:
        model: Model to train
        train_loader: Training data loader
        config: Training configuration
        output_dir: Directory to save checkpoints
    
    Returns:
        TrainingResult with metrics
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    device = model.device
    logger.info(f"Training on device: {device}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    
    # Learning rate scheduler
    total_steps = config.max_steps
    warmup_steps = int(total_steps * config.warmup_ratio)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.1, 1 - (step - warmup_steps) / max(1, total_steps - warmup_steps))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training loop
    model.train()
    history = []
    best_token_f1 = 0
    best_span_f1 = 0
    best_step = 0
    
    start_time = time.time()
    step = 0
    
    logger.info(f"Starting tiny overfit training: {config.max_steps} steps")
    
    while step < config.max_steps:
        for batch in train_loader:
            if step >= config.max_steps:
                break
            
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward
            outputs = model(input_ids, attention_mask, labels)
            loss = outputs["loss"]
            
            # Scale loss for gradient accumulation
            loss = loss / config.gradient_accumulation
            
            # Backward
            loss.backward()
            
            # Gradient accumulation
            if (step + 1) % config.gradient_accumulation == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                step += 1
                
                # Log
                current_lr = scheduler.get_last_lr()[0]
                logger.info(
                    f"Step {step}/{config.max_steps} | "
                    f"Loss: {loss.item() * config.gradient_accumulation:.4f} | "
                    f"LR: {current_lr:.2e}"
                )
                
                # Record history
                history.append({
                    "step": step,
                    "loss": loss.item() * config.gradient_accumulation,
                    "lr": current_lr,
                    "valid_tokens": outputs.get("valid_token_count", 0),
                    "positive_tokens": outputs.get("positive_token_count", 0),
                })
                
                # Save checkpoint
                if step % 50 == 0 or step == config.max_steps:
                    checkpoint_path = checkpoint_dir / f"step_{step}.pt"
                    torch.save({
                        "step": step,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                    }, checkpoint_path)
                    logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    train_time = time.time() - start_time
    
    # Save final model
    final_checkpoint = checkpoint_dir / "tiny_overfit_final.pt"
    torch.save({
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }, final_checkpoint)
    
    return TrainingResult(
        best_step=best_step,
        best_token_f1=best_token_f1,  # Will be filled by evaluation
        best_span_f1=best_span_f1,  # Will be filled by evaluation
        final_step=step,
        final_loss=loss.item() * config.gradient_accumulation,
        train_time=train_time,
        checkpoint_path=final_checkpoint,
        history=history,
    )


def train_full(
    model: TokenClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainConfig,
    output_dir: Path,
    evaluator,  # Will be passed from main
) -> TrainingResult:
    """
    Full training loop with validation.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration
        output_dir: Directory to save checkpoints
        evaluator: Evaluator instance
    
    Returns:
        TrainingResult with metrics
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    device = model.device
    logger.info(f"Training on device: {device}")
    
    # Calculate total steps
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * config.max_epochs
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    
    # Scheduler
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=warmup_steps
    )
    
    model.train()
    history = []
    best_span_f1 = 0
    best_step = 0
    step = 0
    
    start_time = time.time()
    
    logger.info(f"Starting full training: {total_steps} steps ({config.max_epochs} epochs)")
    
    for epoch in range(config.max_epochs):
        logger.info(f"Epoch {epoch + 1}/{config.max_epochs}")
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward
            outputs = model(input_ids, attention_mask, labels)
            loss = outputs["loss"]
            
            # Scale loss
            loss = loss / config.gradient_accumulation
            loss.backward()
            
            if (batch_idx + 1) % config.gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                
                step += 1
                
                # Log
                if step % 10 == 0:
                    current_lr = optimizer.param_groups[0]["lr"]
                    logger.info(
                        f"Step {step} | Loss: {loss.item() * config.gradient_accumulation:.4f} | "
                        f"LR: {current_lr:.2e}"
                    )
                
                # Evaluate periodically
                if step % config.eval_every == 0:
                    eval_metrics = evaluator.evaluate(model, val_loader)
                    model.train()
                    
                    span_f1 = eval_metrics.get("character_f1", 0)
                    token_f1 = eval_metrics.get("positive_f1", 0)
                    
                    logger.info(
                        f"Eval @ Step {step} | "
                        f"Token F1: {token_f1:.4f} | "
                        f"Span F1: {span_f1:.4f}"
                    )
                    
                    history.append({
                        "step": step,
                        "loss": loss.item() * config.gradient_accumulation,
                        "token_f1": token_f1,
                        "span_f1": span_f1,
                    })
                    
                    # Save best
                    if span_f1 > best_span_f1:
                        best_span_f1 = span_f1
                        best_token_f1 = token_f1
                        best_step = step
                        
                        best_path = checkpoint_dir / "best.pt"
                        torch.save({
                            "step": step,
                            "model_state": model.state_dict(),
                            "metrics": eval_metrics,
                        }, best_path)
                        logger.info(f"New best: Span F1 = {span_f1:.4f}")
    
    train_time = time.time() - start_time
    
    # Save last
    last_path = checkpoint_dir / "last.pt"
    torch.save({
        "step": step,
        "model_state": model.state_dict(),
    }, last_path)
    
    return TrainingResult(
        best_step=best_step,
        best_token_f1=best_token_f1,
        best_span_f1=best_span_f1,
        final_step=step,
        final_loss=loss.item() * config.gradient_accumulation,
        train_time=train_time,
        checkpoint_path=last_path,
        history=history,
    )
