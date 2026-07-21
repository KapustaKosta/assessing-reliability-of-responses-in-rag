"""
Training CLI for supervised MIL faithfulness model.

Usage:
    python -m claim_mil.train --epochs 3 --batch_size 4 --lr 2e-5
    python -m claim_mil.train --smoke_test --max_bags 20
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# Setup paths
_SRC_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(_SRC_DIR))

from claim_mil.claim_bags import (
    ClaimBagBuilder,
    ClaimBag,
    create_grouped_split,
    generate_split_manifest,
)
from claim_mil.model import ClaimMILModel, MILConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Dataset
# =============================================================================

class ClaimBagDataset(Dataset):
    def __init__(self, bags: list[ClaimBag]):
        self.bags = bags

    def __len__(self) -> int:
        return len(self.bags)

    def __getitem__(self, idx: int) -> tuple:
        bag = self.bags[idx]
        windows = [w.window_text for w in bag.context_windows]
        return windows, bag.claim_text, bag.claim_label


def collate_bags(batch):
    windows_batch = [b[0] for b in batch]
    claims_batch = [b[1] for b in batch]
    labels_batch = [b[2] for b in batch]
    return windows_batch, claims_batch, labels_batch


# =============================================================================
# MIL Forward
# =============================================================================

def mil_forward_batch(
    model: ClaimMILModel,
    batch_windows: list[list[str]],
    batch_claims: list[str],
    batch_labels: list[int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Optimized MIL forward: encodes all (window, claim) pairs in one batched call
    then performs per-bag max-pooling.

    Handles empty-window bags by outputting a scalar 0.0 logit (neutral).

    Returns:
        (loss, logits_tensor)
        logits: raw logits for "supported" class (positive = supported)
    """
    # Flatten all (window, claim) pairs across the batch
    flat_text_pairs = []
    flat_claims = []
    bag_to_window_indices = []

    for windows, claim in zip(batch_windows, batch_claims):
        if not windows:
            bag_to_window_indices.append(None)  # empty bag marker
            continue
        indices = []
        for w in windows:
            flat_text_pairs.append(w)
            flat_claims.append(claim)
            indices.append(len(flat_text_pairs) - 1)
        bag_to_window_indices.append(indices)

    all_support_logits: list[torch.Tensor] = []

    if flat_text_pairs:
        tok = model.tokenizer
        inputs = tok(
            flat_text_pairs,
            flat_claims,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model.encoder(**inputs)
        window_repr_all = outputs.last_hidden_state[:, 0, :].to(torch.float32)

        for windows, indices in zip(batch_windows, bag_to_window_indices):
            if indices is None:
                # Empty bag: output neutral 0.0 for shape (1,)
                all_support_logits.append(torch.zeros(1, device=device, requires_grad=True))
                continue

            bag_repr = window_repr_all[indices].max(dim=0).values
            bag_repr = model.dropout(bag_repr)
            logit = model.classifier(bag_repr.unsqueeze(0)).view(-1)
            all_support_logits.append(logit)
    else:
        # All bags empty: all-neutral
        for _ in batch_windows:
            all_support_logits.append(torch.zeros(1, device=device, requires_grad=True))

    logits = torch.stack(all_support_logits)
    logits = logits.view(-1)
    labels = torch.tensor(batch_labels, dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(logits, labels)

    return loss, logits


# =============================================================================
# Trainer
# =============================================================================

class MILTrainer:
    def __init__(
        self,
        model: ClaimMILModel,
        config: MILConfig,
        train_bags: list[ClaimBag],
        dev_bags: list[ClaimBag],
        args: argparse.Namespace,
        results_dir: Path,
    ):
        self.model = model
        self.config = config
        self.args = args
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Note: MPS has scatter_index -1 errors with this MIL forward; CPU used for full training
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        self.model.to(self.device)

        # Compute class weights
        train_labels = np.array([b.claim_label for b in train_bags])
        n_pos = int(train_labels.sum())
        n_neg = len(train_labels) - n_pos
        pos_weight_val = n_neg / max(n_pos, 1)
        self.pos_weight = torch.tensor([pos_weight_val]).to(self.device)
        logger.info(
            f"Class distribution: neg={n_neg} ({n_neg/len(train_labels)*100:.1f}%), "
            f"pos={n_pos} ({n_pos/len(train_labels)*100:.1f}%), "
            f"pos_weight={pos_weight_val:.4f}"
        )

        # Criterion with pos_weight for BCE
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.epochs, eta_min=args.lr * 0.1,
        )

        # Mixed precision
        self.use_amp = args.use_amp and self.device.type in ("cuda", "mps")
        self.scaler = GradScaler('cuda') if self.use_amp else None

        # Dataloaders
        self.train_loader = DataLoader(
            ClaimBagDataset(train_bags),
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_bags,
            num_workers=0,
            drop_last=False,
        )
        self.dev_loader = DataLoader(
            ClaimBagDataset(dev_bags),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_bags,
            num_workers=0,
        )

        self.train_bags = train_bags
        self.dev_bags = dev_bags

        self.best_dev_f1 = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        self.history = []

    def _run_epoch(self, epoch: int, loader: DataLoader, train: bool) -> dict:
        """Run one epoch of training or evaluation."""
        self.model.train() if train else self.model.eval()
        total_loss = 0.0
        n_batches = 0
        all_labels = []
        all_preds = []
        all_probs_unsupported = []

        iterator = tqdm(loader, desc=f"Epoch {epoch+1} [{'Train' if train else 'Dev'}]")

        for batch_windows, batch_claims, batch_labels in iterator:
            labels_np = np.array(batch_labels)

            if train:
                self.optimizer.zero_grad()

                if self.use_amp:
                    with torch.amp.autocast(device_type=self.device.type, enabled=True):
                        loss, logits = mil_forward_batch(
                            self.model, batch_windows, batch_claims, batch_labels, self.device
                        )
                else:
                    loss, logits = mil_forward_batch(
                        self.model, batch_windows, batch_claims, batch_labels, self.device
                    )

                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    self.optimizer.step()

                total_loss += loss.item()

            else:
                with torch.no_grad():
                    loss, logits = mil_forward_batch(
                        self.model, batch_windows, batch_claims, batch_labels, self.device
                    )
                total_loss += loss.item()

            n_batches += 1

            # Convert support logits to unsupported probabilities
            # logit > 0 -> supported (label=0), logit < 0 -> unsupported (label=1)
            # p_supported = sigmoid(logit), p_unsupported = sigmoid(-logit)
            p_supported = torch.sigmoid(logits.detach()).cpu().numpy()
            p_unsupported = 1.0 - p_supported

            preds = (p_unsupported >= 0.5).astype(int)

            all_labels.extend(labels_np.tolist())
            all_preds.extend(preds.tolist())
            all_probs_unsupported.extend(p_unsupported.tolist())

            if train:
                iterator.set_postfix({"loss": f"{loss.item():.4f}"})

        self.scheduler.step()

        # Compute metrics
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs_unsupported)

        metrics = _compute_metrics(all_labels, all_preds, all_probs)
        metrics["loss"] = total_loss / n_batches

        return metrics

    def train(self) -> dict:
        logger.info(f"Training for {self.args.epochs} epochs")
        start_time = time.time()

        for epoch in range(self.args.epochs):
            # Train
            train_m = self._run_epoch(epoch, self.train_loader, train=True)

            # Eval
            dev_m = self._run_epoch(epoch, self.dev_loader, train=False)
            dev_f1 = dev_m.get("f1", 0)

            logger.info(
                f"Epoch {epoch+1} | "
                f"Train loss={train_m['loss']:.4f} F1={train_m.get('f1',0):.4f} Acc={train_m.get('accuracy',0):.4f} | "
                f"Dev F1={dev_f1:.4f} Acc={dev_m.get('accuracy',0):.4f} BA={dev_m.get('balanced_accuracy',0):.4f} "
                f"AUROC={dev_m.get('auroc', 'N/A')} AUPRC={dev_m.get('auprc', 'N/A')}"
            )

            self.history.append({
                "epoch": epoch + 1,
                "train_loss": train_m["loss"],
                "train_f1": train_m.get("f1", 0),
                "dev_f1": dev_f1,
                "dev_ba": dev_m.get("balanced_accuracy", 0),
                "dev_auroc": dev_m.get("auroc"),
                "dev_auprc": dev_m.get("auprc"),
                "dev_precision": dev_m.get("precision_unsupported_class", 0),
                "dev_recall": dev_m.get("recall_unsupported_class", 0),
                "lr": self.scheduler.get_last_lr()[0],
            })

            if dev_f1 > self.best_dev_f1:
                self.best_dev_f1 = dev_f1
                self.best_epoch = epoch + 1
                self.patience_counter = 0
                self._save_checkpoint(self.results_dir / "best_checkpoint.pt")
                logger.info(f"  -> New best! Dev F1: {dev_f1:.4f}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.args.patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        elapsed = time.time() - start_time

        pd.DataFrame(self.history).to_csv(
            self.results_dir / "training_history.csv", index=False
        )
        logger.info(
            f"Training done in {elapsed:.1f}s. Best epoch: {self.best_epoch} "
            f"(dev F1={self.best_dev_f1:.4f})"
        )

        return {
            "best_epoch": self.best_epoch,
            "best_dev_f1": self.best_dev_f1,
            "elapsed_seconds": elapsed,
            "history": self.history,
        }

    def _save_checkpoint(self, path: Path):
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": asdict(self.config),
            "best_dev_f1": self.best_dev_f1,
            "best_epoch": self.best_epoch,
        }, path)


# =============================================================================
# Metrics
# =============================================================================

def _compute_metrics(labels: np.ndarray, preds: np.ndarray, probs: np.ndarray) -> dict:
    """
    Compute classification metrics for the unsupported class (label=1).

    Label convention:
        0 = supported
        1 = unsupported
    """
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        balanced_accuracy_score, confusion_matrix, roc_auc_score,
        average_precision_score,
    )

    UNSUPPORTED = 1
    SUPPORTED = 0

    n = len(labels)
    if n == 0:
        return {}

    metrics = {
        "n": n,
        "accuracy": accuracy_score(labels, preds),
        "balanced_accuracy": balanced_accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="binary", pos_label=UNSUPPORTED),
        "precision": precision_score(labels, preds, average="binary", pos_label=UNSUPPORTED),
        "recall": recall_score(labels, preds, average="binary", pos_label=UNSUPPORTED),
        "precision_unsupported_class": precision_score(labels, preds, average="binary", pos_label=UNSUPPORTED),
        "recall_unsupported_class": recall_score(labels, preds, average="binary", pos_label=UNSUPPORTED),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }

    # AUROC / AUPRC
    try:
        metrics["auroc"] = roc_auc_score(labels, probs)
    except Exception:
        metrics["auroc"] = None

    try:
        metrics["auprc"] = average_precision_score(labels, probs)
    except Exception:
        metrics["auprc"] = None

    # Confusion matrix with explicit order [Unfaithful(1), Faithful(0)]
    cm = confusion_matrix(labels, preds, labels=[UNSUPPORTED, SUPPORTED])
    metrics["confusion_matrix"] = cm.tolist()
    metrics["tn"] = int(cm[1, 1])  # True Faithful (both 0)
    metrics["fp"] = int(cm[0, 1])  # Predicted Faithful but was Unfaithful
    metrics["fn"] = int(cm[1, 0])  # Predicted Unfaithful but was Faithful
    metrics["tp"] = int(cm[0, 0])  # True Unfaithful

    return metrics


# =============================================================================
# Threshold Selection
# =============================================================================

def select_threshold(
    model: ClaimMILModel,
    dev_bags: list[ClaimBag],
    device: torch.device,
) -> tuple[float, dict]:
    """Select best threshold on dev using macro-F1."""
    model.eval()
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for bag in tqdm(dev_bags, desc="Threshold scan"):
            if not bag.context_windows:
                p_unsupported = 0.5
            else:
                windows = [w.window_text for w in bag.context_windows]
                result = model.forward(windows, bag.claim_text)
                p_unsupported = result["p_unsupported"]
            all_labels.append(bag.claim_label)
            all_probs.append(p_unsupported)

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    best_thresh = 0.5
    best_f1 = 0.0
    scan_results = []

    for thresh in np.arange(0.10, 0.91, 0.01):
        preds = (all_probs >= thresh).astype(int)
        m = _compute_metrics(all_labels, preds, all_probs)
        scan_results.append({"threshold": round(thresh, 2), "f1": m.get("f1_macro", 0)})
        if m.get("f1_macro", 0) > best_f1:
            best_f1 = m.get("f1_macro", 0)
            best_thresh = thresh

    logger.info(f"Best threshold={best_thresh:.2f} (macro-F1={best_f1:.4f})")
    return float(best_thresh), {"threshold_scan": scan_results, "best_threshold": best_thresh}


# =============================================================================
# Argument Parser
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MIL faithfulness model")

    p.add_argument("--models", nargs="+",
                   default=["Llama-2-7b-chat-hf", "Mistral-7B-Instruct-v0.3"])
    p.add_argument("--dev_fraction", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_bags", type=int, default=None)

    p.add_argument("--encoder", type=str,
                   default="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
    p.add_argument("--pooling_mode", type=str, default="max",
                   choices=["max", "log_sum_exp"])
    p.add_argument("--lse_temp", type=float, default=1.0)
    p.add_argument("--dropout", type=float, default=0.1)

    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--use_amp", action="store_true", default=True)
    p.add_argument("--no_amp", dest="use_amp", action="store_false")

    p.add_argument("--results_dir", type=str,
                   default="results/phase2_mil_faithfulness")
    p.add_argument("--smoke_test", action="store_true")
    p.add_argument("--overfit_diagnostic", action="store_true")
    p.add_argument("--val_only", action="store_true")

    return p.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    args = parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    logger.info("=" * 60)
    logger.info("PHASE 2: SUPERVISED MIL FAITHFULNESS")
    logger.info("=" * 60)

    mil_config = MILConfig(
        encoder_name=args.encoder,
        pooling_mode=args.pooling_mode,
        log_sum_exp_temperature=args.lse_temp,
        dropout=args.dropout,
    )

    # Git info
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()
        git_branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
        git_dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], text=True).strip())
    except Exception:
        git_commit = git_branch = "unknown"
        git_dirty = False

    logger.info(f"Git: {git_commit[:8]} {git_branch} dirty={git_dirty}")

    # Tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.encoder)

    # Load data
    from ragognize_adapter import (
        RAGognizeAdapter, load_ragognize_dataset,
        create_train_val_split,
    )

    logger.info("Loading RAGognize dataset...")
    raw = load_ragognize_dataset()

    split_info = create_train_val_split(raw, val_size=0.15, seed=42)
    project_val_qids = {
        raw["train"][i]["user_prompt_index"]
        for i in split_info["val_indices"]
    }
    logger.info(f"Project val questions: {len(project_val_qids)}")

    adapter = RAGognizeAdapter(models=args.models)

    # Build train samples from project train (exclude val questions)
    train_items = []
    for row_idx, item in enumerate(raw["train"]):
        qid = item["user_prompt_index"]
        if qid in project_val_qids:
            continue
        item_copy = dict(item)
        item_copy["_source_row_index"] = row_idx
        item_copy["_source_split"] = "train"
        train_items.append(item_copy)

    for row_idx, item in enumerate(raw["test"]):
        item_copy = dict(item)
        item_copy["_source_row_index"] = row_idx
        item_copy["_source_split"] = "test"
        train_items.append(item_copy)

    logger.info(f"Project train items: {len(train_items)}")

    # Expand to UnifiedSamples
    unified = []
    for item in train_items:
        ssplit = item.get("_source_split", "train")
        sidx = item.get("_source_row_index", 0)
        samples = adapter.parse_sample(item, ssplit, sidx)
        unified.extend(samples)

    logger.info(f"Expanded unified samples: {len(unified)}")

    # Grouped split
    logger.info("Creating grouped train/dev split...")
    split_result = create_grouped_split(
        samples=unified,
        dev_fraction=args.dev_fraction,
        seed=args.seed,
        project_val_question_ids=project_val_qids,
    )

    logger.info(f"Train: {len(split_result['train_samples'])} samples, "
                f"Dev: {len(split_result['dev_samples'])} samples, "
                f"Leakage: {split_result['leakage']}")

    # Save split manifest
    manifest_path = results_dir / "supervised_split_manifest.csv"
    generate_split_manifest(unified, split_result, manifest_path)

    # Build claim bags
    logger.info("Building claim bags...")
    builder = ClaimBagBuilder(adapter=adapter, tokenizer=tokenizer, max_length=512)

    train_bags = []
    dev_bags = []
    skipped = []

    for sample in tqdm(split_result["train_samples"], desc="Train bags"):
        bags, sk = builder.sample_to_claim_bags(sample)
        train_bags.extend(bags)
        skipped.extend(sk)

    for sample in tqdm(split_result["dev_samples"], desc="Dev bags"):
        bags, sk = builder.sample_to_claim_bags(sample)
        dev_bags.extend(bags)
        skipped.extend(sk)

    if args.max_bags:
        train_bags = train_bags[: args.max_bags]
        dev_bags = dev_bags[: args.max_bags]

    n_train_pos = sum(b.claim_label for b in train_bags)
    n_dev_pos = sum(b.claim_label for b in dev_bags)
    logger.info(
        f"Bags: train={len(train_bags)} (pos={n_train_pos}, {n_train_pos/len(train_bags)*100:.1f}%), "
        f"dev={len(dev_bags)} (pos={n_dev_pos}, {n_dev_pos/len(dev_bags)*100:.1f}%), "
        f"skipped={len(skipped)}"
    )

    # Save claim bags
    def save_bags(bags, path):
        records = []
        for b in bags:
            records.append({
                "question": b.question,
                "answer": b.answer,
                "claim_text": b.claim_text,
                "claim_char_start": b.claim_char_start,
                "claim_char_end": b.claim_char_end,
                "context_windows": [
                    asdict(w) for w in b.context_windows
                ],
                "claim_label": b.claim_label,
                "question_id": b.question_id,
                "expanded_sample_id": b.expanded_sample_id,
                "source_model": b.source_model,
                "gold_answer_faithful": b.gold_answer_faithful,
            })
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return len(records)

    train_bags_path = results_dir / "claim_bags_train.jsonl"
    dev_bags_path = results_dir / "claim_bags_dev.jsonl"
    save_bags(train_bags, train_bags_path)
    save_bags(dev_bags, dev_bags_path)

    if skipped:
        with open(results_dir / "skipped_samples.jsonl", "w") as f:
            for s in skipped:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    logger.info(f"Saved: train={train_bags_path}, dev={dev_bags_path}")

    # ---- Smoke Test ----
    if args.smoke_test:
        logger.info("=== SMOKE TEST ===")
        torch.manual_seed(args.seed)
        device = torch.device("cpu")
        model = ClaimMILModel(mil_config, tokenizer=tokenizer)
        model.to(device)

        if train_bags:
            bag = train_bags[0]
            windows = [w.window_text for w in bag.context_windows]
            result = model.forward(windows, bag.claim_text)
            logger.info(f"Forward OK: p_unsupported={result['p_unsupported']:.4f}")

            # Backward
            model.train()
            loss, logits = mil_forward_batch(
                model,
                [windows],
                [bag.claim_text],
                [bag.claim_label],
                device,
            )
            loss.backward()
            logger.info(f"Backward OK: loss={loss.item():.4f}")

            # Checkpoint
            ckpt = results_dir / "smoke_checkpoint.pt"
            torch.save({"model_state_dict": model.state_dict()}, ckpt)
            loaded = torch.load(ckpt, weights_only=False)
            model.load_state_dict(loaded["model_state_dict"])
            logger.info("Checkpoint save/load OK")

            # Verify IDs
            assert bag.claim_label in (0, 1)
            assert len(bag.expanded_sample_id) > 0
            logger.info("ID verification OK")

        logger.info("=== SMOKE TEST PASSED ===")
        return

    if args.overfit_diagnostic:
        logger.info("=== OVERFIT DIAGNOSTIC ===")
        tiny_bags = train_bags[: min(20, len(train_bags))]
        device = torch.device("cpu")
        model = ClaimMILModel(mil_config, tokenizer=tokenizer)
        model.to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

        init_losses = []
        final_losses = []

        for epoch in range(10):
            model.train()
            epoch_losses = []
            for i in range(0, len(tiny_bags), 4):
                batch = tiny_bags[i : i + 4]
                windows = [[w.window_text for w in b.context_windows] for b in batch]
                claims = [b.claim_text for b in batch]
                labels = [b.claim_label for b in batch]

                loss, _ = mil_forward_batch(model, windows, claims, labels, device)
                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch_losses.append(loss.item())

            avg_loss = np.mean(epoch_losses)
            if epoch == 0:
                init_losses.append(avg_loss)
            if epoch == 9:
                final_losses.append(avg_loss)
            logger.info(f"  Epoch {epoch+1}: loss={avg_loss:.4f}")

        if init_losses and final_losses:
            decrease = init_losses[0] - final_losses[0]
            logger.info(
                f"Overfit diag: init={init_losses[0]:.4f} final={final_losses[0]:.4f} "
                f"decrease={decrease:.4f}"
            )
            assert decrease > 0.1, "Loss should decrease substantially on tiny set"
        logger.info("=== OVERFIT DIAGNOSTIC PASSED ===")
        return

    if args.val_only:
        logger.info("--val_only: skipping training")
        return

    # ---- Full Training ----
    logger.info("=== TRAINING ===")
    model = ClaimMILModel(mil_config, tokenizer=tokenizer)
    trainer = MILTrainer(
        model=model, config=mil_config,
        train_bags=train_bags, dev_bags=dev_bags,
        args=args, results_dir=results_dir,
    )

    train_result = trainer.train()

    # Threshold selection
    logger.info("=== THRESHOLD SELECTION ===")
    best_thresh, thresh_info = select_threshold(model, dev_bags, trainer.device)

    # Save config
    best_config = {
        "threshold": best_thresh,
        "pooling_mode": args.pooling_mode,
        "encoder": args.encoder,
        "best_epoch": train_result["best_epoch"],
        "best_dev_f1": train_result["best_dev_f1"],
    }
    with open(results_dir / "best_config.json", "w") as f:
        json.dump(best_config, f, indent=2)

    # Run manifest
    manifest = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "git_commit": git_commit,
        "git_branch": git_branch,
        "git_dirty": git_dirty,
        "project_train_samples": len(unified),
        "train_samples": len(split_result["train_samples"]),
        "dev_samples": len(split_result["dev_samples"]),
        "train_claim_bags": len(train_bags),
        "dev_claim_bags": len(dev_bags),
        "train_pos_rate": n_train_pos / len(train_bags) if train_bags else 0,
        "dev_pos_rate": n_dev_pos / len(dev_bags) if dev_bags else 0,
        "skipped_count": len(skipped),
        "best_threshold": best_thresh,
        "best_dev_f1": train_result["best_dev_f1"],
        "best_epoch": train_result["best_epoch"],
        "training_elapsed_seconds": train_result["elapsed_seconds"],
        "split": split_result["manifest"],
        "leakage": split_result["leakage"],
        "pooling_mode": args.pooling_mode,
        "encoder": args.encoder,
    }
    with open(results_dir / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Done. Manifest: {results_dir / 'run_manifest.json'}")


if __name__ == "__main__":
    main()
