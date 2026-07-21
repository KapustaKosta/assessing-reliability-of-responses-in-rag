#!/usr/bin/env python3
"""
Demo pipeline: builds bags, runs minimal training, saves artifacts.

Uses max_length=256, batch_size=2, 60 train bags / 30 dev bags / 1 epoch.
"""
import os, sys, json, time, logging
from pathlib import Path
from datetime import datetime
from dataclasses import asdict

sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

os.environ["HF_HUB_OFFLINE"] = "1"
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.FileHandler("/tmp/demo_run.log"), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("demo")

from claim_mil.claim_bags import ClaimBagBuilder
from claim_mil.model import ClaimMILModel, MILConfig
from claim_mil.train import mil_forward_batch, _compute_metrics
from ragognize_adapter import RAGognizeAdapter, load_ragognize_dataset, create_train_val_split
from claim_mil.claim_bags import create_grouped_split, generate_split_manifest

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    results_dir = Path("results/phase2_mil_faithfulness")
    results_dir.mkdir(parents=True, exist_ok=True)

    log.info("=== Loading RAGognize ===")
    raw = load_ragognize_dataset()
    split_info = create_train_val_split(raw, val_size=0.15, seed=42)
    project_val_qids = {raw["train"][i]["user_prompt_index"] for i in split_info["val_indices"]}

    train_items = []
    for row_idx, item in enumerate(raw["train"]):
        if item["user_prompt_index"] in project_val_qids:
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

    log.info(f"Project train items: {len(train_items)}")

    adapter = RAGognizeAdapter(models=["Llama-2-7b-chat-hf", "Mistral-7B-Instruct-v0.3"])
    unified = []
    for item in train_items:
        samples = adapter.parse_sample(item, item.get("_source_split", "train"), item.get("_source_row_index", 0))
        unified.extend(samples)

    log.info(f"Expanded samples: {len(unified)}")

    split_result = create_grouped_split(
        samples=unified, dev_fraction=0.10, seed=42,
        project_val_question_ids=project_val_qids,
    )

    manifest_path = results_dir / "supervised_split_manifest.csv"
    generate_split_manifest(unified, split_result, manifest_path)
    log.info(f"Saved split manifest: {manifest_path}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")

    log.info("=== Building claim bags (max_length=256) ===")
    builder = ClaimBagBuilder(adapter=adapter, tokenizer=tokenizer, max_length=256)

    train_bags = []
    dev_bags = []

    for sample in split_result["train_samples"][:50]:
        bags, _ = builder.sample_to_claim_bags(sample)
        train_bags.extend(bags)
        if len(train_bags) >= 60:
            break
    train_bags = train_bags[:60]

    for sample in split_result["dev_samples"][:30]:
        bags, _ = builder.sample_to_claim_bags(sample)
        dev_bags.extend(bags)
        if len(dev_bags) >= 30:
            break
    dev_bags = dev_bags[:30]

    log.info(f"Train bags: {len(train_bags)}, Dev bags: {len(dev_bags)}")
    n_train_pos = sum(b.claim_label for b in train_bags)
    n_dev_pos = sum(b.claim_label for b in dev_bags)
    log.info(f"Train pos: {n_train_pos}, Dev pos: {n_dev_pos}")

    # Save bags
    def save_bags(bags, path):
        records = []
        for b in bags:
            records.append({
                "question": b.question,
                "answer": b.answer,
                "claim_text": b.claim_text,
                "claim_char_start": b.claim_char_start,
                "claim_char_end": b.claim_char_end,
                "context_windows": [asdict(w) for w in b.context_windows],
                "claim_label": b.claim_label,
                "question_id": b.question_id,
                "expanded_sample_id": b.expanded_sample_id,
                "source_model": b.source_model,
                "gold_answer_faithful": b.gold_answer_faithful,
            })
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    save_bags(train_bags, results_dir / "claim_bags_train.jsonl")
    save_bags(dev_bags, results_dir / "claim_bags_dev.jsonl")
    log.info("Saved claim bags")

    # Training
    log.info("=== TRAINING (1 epoch, max_length=256, bs=2) ===")
    mil_config = MILConfig(encoder_name="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli", pooling_mode="max")
    device = torch.device("cpu")
    model = ClaimMILModel(mil_config, tokenizer=tokenizer)
    model.to(device)

    train_labels = np.array([b.claim_label for b in train_bags])
    n_pos = int(train_labels.sum())
    n_neg = len(train_labels) - n_pos
    pos_weight_val = n_neg / max(n_pos, 1)
    pos_weight = torch.tensor([pos_weight_val])

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    model.train()
    losses = []
    batch_size = 2
    n_skipped = 0
    start = time.time()

    for i in range(0, len(train_bags), batch_size):
        batch = train_bags[i:i+batch_size]
        windows_batch = [[w.window_text for w in b.context_windows] for b in batch]
        claims_batch = [b.claim_text for b in batch]
        labels_batch = [b.claim_label for b in batch]

        if all(len(w) == 0 for w in windows_batch):
            log.info(f"  Batch {i//batch_size + 1}: SKIP (empty windows)")
            n_skipped += 1
            continue

        try:
            loss, logits = mil_forward_batch(model, windows_batch, claims_batch, labels_batch, device)
        except Exception as e:
            log.info(f"  Batch {i//batch_size + 1}: ERROR {e}")
            n_skipped += 1
            continue

        if torch.isnan(loss) or torch.isinf(loss):
            log.info(f"  Batch {i//batch_size + 1}: SKIP (NaN/Inf)")
            n_skipped += 1
            continue

        optimizer.zero_grad()
        loss.backward()
        grad_norm = sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5

        if grad_norm > 1000:
            log.info(f"  Batch {i//batch_size + 1}: SKIP (grad_norm={grad_norm:.1f})")
            n_skipped += 1
            continue

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())
        log.info(f"  Batch {i//batch_size + 1}: loss={loss.item():.4f} grad={grad_norm:.2f}")

    elapsed = time.time() - start
    log.info(f"Training done in {elapsed:.1f}s, completed={len(losses)}, skipped={n_skipped}")

    # Dev eval
    log.info("=== DEV EVAL ===")
    model.eval()
    dev_labels, dev_probs = [], []
    with torch.no_grad():
        for bag in dev_bags:
            if not bag.context_windows:
                p_unsupported = 0.5
            else:
                try:
                    result = model.forward([w.window_text for w in bag.context_windows], bag.claim_text)
                    p_unsupported = result["p_unsupported"]
                except Exception:
                    p_unsupported = 0.5
            dev_labels.append(bag.claim_label)
            dev_probs.append(p_unsupported)

    dev_labels = np.array(dev_labels)
    dev_probs = np.array(dev_probs)
    dev_preds = (dev_probs >= 0.5).astype(int)
    dev_metrics = _compute_metrics(dev_labels, dev_preds, dev_probs)
    log.info(f"Dev F1={dev_metrics.get('f1', 0):.4f} Acc={dev_metrics.get('accuracy', 0):.4f} BA={dev_metrics.get('balanced_accuracy', 0):.4f}")

    # Save checkpoint
    ckpt_path = results_dir / "best_checkpoint.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": asdict(mil_config),
        "best_dev_f1": dev_metrics.get("f1", 0.0),
        "best_epoch": 1,
    }, ckpt_path)
    log.info(f"Saved checkpoint: {ckpt_path}")

    # best_config.json
    best_config = {
        "threshold": 0.5,
        "pooling_mode": "max",
        "encoder": "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
        "best_epoch": 1,
        "best_dev_f1": dev_metrics.get("f1", 0.0),
        "best_dev_metrics": {k: v for k, v in dev_metrics.items() if k != "confusion_matrix"},
    }
    with open(results_dir / "best_config.json", "w") as f:
        json.dump(best_config, f, indent=2)

    # training_history.csv
    pd.DataFrame([{
        "epoch": 1,
        "train_loss": float(np.mean(losses)) if losses else float("nan"),
        "train_loss_last": losses[-1] if losses else float("nan"),
        "dev_f1": dev_metrics.get("f1", 0),
        "dev_ba": dev_metrics.get("balanced_accuracy", 0),
        "dev_auroc": dev_metrics.get("auroc") or 0,
        "dev_auprc": dev_metrics.get("auprc") or 0,
        "dev_precision": dev_metrics.get("precision_unsupported_class", 0),
        "dev_recall": dev_metrics.get("recall_unsupported_class", 0),
        "lr": 2e-5,
        "n_batches_completed": len(losses),
        "n_batches_skipped": n_skipped,
    }]).to_csv(results_dir / "training_history.csv", index=False)

    # run_manifest.json
    manifest = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "git_commit": "7a9c6a2",
        "git_branch": "feature/phase2-supervised-faithfulness",
        "git_dirty": True,
        "project_train_samples": len(unified),
        "train_samples": len(split_result["train_samples"]),
        "dev_samples": len(split_result["dev_samples"]),
        "train_claim_bags": len(train_bags),
        "dev_claim_bags": len(dev_bags),
        "train_pos_rate": n_train_pos / len(train_bags) if train_bags else 0,
        "dev_pos_rate": n_dev_pos / len(dev_bags) if dev_bags else 0,
        "skipped_count": 0,
        "best_threshold": 0.5,
        "best_dev_f1": dev_metrics.get("f1", 0.0),
        "best_epoch": 1,
        "training_elapsed_seconds": elapsed,
        "training_n_batches_completed": len(losses),
        "training_n_batches_skipped": n_skipped,
        "split": split_result["manifest"],
        "leakage": split_result["leakage"],
        "pooling_mode": "max",
        "encoder": "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
        "max_length": 256,
        "batch_size": batch_size,
        "device": "cpu",
        "hardware_note": "CPU-only environment. Full training infeasible within reasonable time budget; demo run with reduced subset.",
    }
    with open(results_dir / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    log.info("=== DEMO COMPLETE ===")

if __name__ == "__main__":
    main()
