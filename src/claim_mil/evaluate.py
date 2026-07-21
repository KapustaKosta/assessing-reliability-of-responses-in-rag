"""
Evaluation on fixed 1100-response project validation set.

This script:
1. Loads the best checkpoint from training
2. Runs inference on project validation samples
3. Aggregates claim-level predictions to answer-level
4. Computes and saves all metrics
5. Generates error analysis
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from dataclasses import asdict
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

_SRC_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(_SRC_DIR))

from claim_mil.claim_bags import ClaimBagBuilder
from claim_mil.model import ClaimMILModel, MILConfig
from ragognize_adapter import (
    RAGognizeAdapter, load_ragognize_dataset,
    create_train_val_split,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Metric helpers (inline to avoid circular imports)
# =============================================================================

def compute_metrics(labels: np.ndarray, preds: np.ndarray) -> dict:
    """Compute classification metrics."""
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        balanced_accuracy_score, confusion_matrix, roc_auc_score,
        average_precision_score,
    )

    n = len(labels)
    if n == 0:
        return {}

    UNSUPPORTED = 1
    SUPPORTED = 0

    metrics = {
        "n": n,
        "accuracy": accuracy_score(labels, preds),
        "balanced_accuracy": balanced_accuracy_score(labels, preds),
        "f1_binary": f1_score(labels, preds, average="binary", pos_label=UNSUPPORTED),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "precision_unsupported": precision_score(labels, preds, average="binary", pos_label=UNSUPPORTED),
        "recall_unsupported": recall_score(labels, preds, average="binary", pos_label=UNSUPPORTED),
    }

    try:
        probs_for_auroc = np.array([0.5] * n)  # placeholder, filled by caller
        metrics["auroc"] = None  # caller fills this
        metrics["auprc"] = None
    except Exception:
        metrics["auroc"] = None
        metrics["auprc"] = None

    cm = confusion_matrix(labels, preds, labels=[UNSUPPORTED, SUPPORTED])
    metrics["confusion_matrix"] = cm.tolist()
    metrics["tn"] = int(cm[1, 1])
    metrics["fp"] = int(cm[0, 1])
    metrics["fn"] = int(cm[1, 0])
    metrics["tp"] = int(cm[0, 0])

    return metrics


# =============================================================================
# Main evaluation
# =============================================================================

def evaluate(
    results_dir: Path,
    checkpoint_path: Path,
    config_path: Path,
    output_dir: Path,
    models: list[str],
) -> dict:
    """Run full evaluation on project validation set."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    with open(config_path) as f:
        config = json.load(f)

    mil_config = MILConfig(
        encoder_name=config.get("encoder", "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"),
        pooling_mode=config.get("pooling_mode", "max"),
    )

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(mil_config.encoder_name)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ClaimMILModel(mil_config, tokenizer=tokenizer)
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    logger.info(f"Loaded checkpoint from {checkpoint_path}")

    # Load project validation samples
    logger.info("Loading RAGognize dataset...")
    raw = load_ragognize_dataset()

    split_info = create_train_val_split(raw, val_size=0.15, seed=42)
    val_qids = {raw["train"][i]["user_prompt_index"] for i in split_info["val_indices"]}
    logger.info(f"Project validation questions: {len(val_qids)}")

    adapter = RAGognizeAdapter(models=models)

    # Build validation samples
    val_items = []
    for row_idx, item in enumerate(raw["train"]):
        if item["user_prompt_index"] in val_qids:
            item_copy = dict(item)
            item_copy["_source_row_index"] = row_idx
            item_copy["_source_split"] = "train"
            val_items.append(item_copy)

    logger.info(f"Project val items: {len(val_items)}")

    unified = []
    for item in val_items:
        ssplit = item.get("_source_split", "train")
        sidx = item.get("_source_row_index", 0)
        samples = adapter.parse_sample(item, ssplit, sidx)
        unified.extend(samples)

    logger.info(f"Expanded val samples: {len(unified)}")

    # Build claim bags
    builder = ClaimBagBuilder(adapter=adapter, tokenizer=tokenizer, max_length=512)

    all_bags = []
    all_skipped = []

    for sample in tqdm(unified, desc="Building val bags"):
        bags, skipped = builder.sample_to_claim_bags(sample)
        all_bags.extend(bags)
        all_skipped.extend(skipped)

    logger.info(f"Val bags: {len(all_bags)}, Skipped: {len(all_skipped)}")

    # Save skipped
    if all_skipped:
        with open(output_dir / "skipped_samples.jsonl", "w") as f:
            for s in all_skipped:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Run inference
    logger.info("Running inference...")
    bag_predictions = []

    with torch.no_grad():
        for bag in tqdm(all_bags, desc="Inference"):
            if not bag.context_windows:
                p_unsupported = 0.5
                support_logit = 0.0
            else:
                windows = [w.window_text for w in bag.context_windows]
                result = model.forward(windows, bag.claim_text)
                p_unsupported = result["p_unsupported"]
                support_logit = result["support_logit"]

            bag_predictions.append({
                "expanded_sample_id": bag.expanded_sample_id,
                "question_id": bag.question_id,
                "source_model": bag.source_model,
                "claim_text": bag.claim_text,
                "claim_char_start": bag.claim_char_start,
                "claim_char_end": bag.claim_char_end,
                "claim_label_gold": bag.claim_label,
                "p_unsupported": float(p_unsupported),
                "support_logit": float(support_logit),
                "gold_answer_faithful": bag.gold_answer_faithful,
                "answer": bag.answer,
                "question": bag.question,
            })

    # Save claim predictions
    claim_preds_path = output_dir / "validation_claim_predictions.jsonl"
    with open(claim_preds_path, "w") as f:
        for pred in bag_predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")
    logger.info(f"Saved claim predictions: {claim_preds_path}")

    # ---- Answer-level aggregation ----
    df = pd.DataFrame(bag_predictions)

    # Answer-level score: max(p_unsupported) per answer
    answer_scores = df.groupby("expanded_sample_id").agg(
        answer_unfaithfulness_score=("p_unsupported", "max"),
        n_claims=("p_unsupported", "count"),
        gold_answer_faithful=("gold_answer_faithful", "first"),
        gold_claim_unsupported=("claim_label_gold", "max"),  # any unsupported claim
        source_model=("source_model", "first"),
        question_id=("question_id", "first"),
        answer=("answer", "first"),
        question=("question", "first"),
    ).reset_index()

    # Apply threshold from config
    threshold = config.get("threshold", 0.5)
    answer_scores["answer_pred_unfaithful"] = (
        answer_scores["answer_unfaithfulness_score"] >= threshold
    ).astype(int)

    # Gold: answer is faithful if no unsupported claims
    answer_scores["gold_answer_label"] = (
        answer_scores["gold_answer_faithful"] & (answer_scores["gold_claim_unsupported"] == 0)
    ).astype(int)

    # Save answer-level predictions
    val_pred_path = output_dir / "validation_predictions.csv"
    answer_scores.to_csv(val_pred_path, index=False)
    logger.info(f"Saved answer predictions: {val_pred_path}")

    # ---- Metrics ----
    labels = answer_scores["gold_answer_label"].values
    preds = answer_scores["answer_pred_unfaithful"].values
    probs = answer_scores["answer_unfaithfulness_score"].values

    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        balanced_accuracy_score, confusion_matrix, roc_auc_score,
        average_precision_score,
    )

    UNSUPPORTED = 1
    SUPPORTED = 0

    metrics = {
        "n_responses": len(answer_scores),
        "n_claims": len(df),
        "threshold": threshold,
        "accuracy": accuracy_score(labels, preds),
        "balanced_accuracy": balanced_accuracy_score(labels, preds),
        "f1_binary": f1_score(labels, preds, average="binary", pos_label=UNSUPPORTED),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "precision_unsupported": precision_score(labels, preds, average="binary", pos_label=UNSUPPORTED),
        "recall_unsupported": recall_score(labels, preds, average="binary", pos_label=UNSUPPORTED),
        "auroc": roc_auc_score(labels, probs),
        "auprc": average_precision_score(labels, probs),
    }

    cm = confusion_matrix(labels, preds, labels=[UNSUPPORTED, SUPPORTED])
    metrics["confusion_matrix"] = cm.tolist()
    metrics["confusion_matrix_labels"] = ["Unfaithful", "Faithful"]

    # Per-source-model metrics
    model_metrics = {}
    for model in answer_scores["source_model"].unique():
        subset = answer_scores[answer_scores["source_model"] == model]
        if len(subset) > 0:
            m_labels = subset["gold_answer_label"].values
            m_preds = subset["answer_pred_unfaithful"].values
            m_probs = subset["answer_unfaithfulness_score"].values
            model_metrics[model] = {
                "n": len(subset),
                "accuracy": float(accuracy_score(m_labels, m_preds)),
                "balanced_accuracy": float(balanced_accuracy_score(m_labels, m_preds)),
                "f1_binary": float(f1_score(m_labels, m_preds, average="binary", pos_label=UNSUPPORTED)),
                "f1_macro": float(f1_score(m_labels, m_preds, average="macro")),
                "auroc": float(roc_auc_score(m_labels, m_probs)) if len(set(m_labels)) > 1 else None,
            }

    # Claim-level metrics
    claim_labels = df["claim_label_gold"].values
    claim_preds = (df["p_unsupported"] >= threshold).astype(int).values
    claim_probs = df["p_unsupported"].values

    claim_metrics = {
        "n_claims": len(df),
        "n_supported": int((claim_labels == 0).sum()),
        "n_unsupported": int((claim_labels == 1).sum()),
        "accuracy": float(accuracy_score(claim_labels, claim_preds)),
        "balanced_accuracy": float(balanced_accuracy_score(claim_labels, claim_preds)),
        "f1_binary": float(f1_score(claim_labels, claim_preds, average="binary", pos_label=UNSUPPORTED)),
        "f1_macro": float(f1_score(claim_labels, claim_preds, average="macro")),
        "auroc": float(roc_auc_score(claim_labels, claim_probs)) if len(set(claim_labels)) > 1 else None,
    }

    # Gold label consistency audit
    canonical_faithful = answer_scores["gold_answer_faithful"].values.astype(int)
    derived_faithful = (answer_scores["gold_claim_unsupported"] == 0).astype(int)
    consistency = (canonical_faithful == derived_faithful).mean()
    logger.info(f"Gold label consistency: {consistency:.2%} (canonical vs derived)")

    # Save metrics
    metrics["by_source_model"] = model_metrics
    metrics["claim_level"] = claim_metrics
    metrics["gold_label_consistency"] = consistency

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics: {metrics_path}")

    # Error analysis
    errors = answer_scores[
        (answer_scores["gold_answer_label"] != answer_scores["answer_pred_unfaithful"])
    ].copy()
    errors["error_type"] = errors.apply(
        lambda r: "FP" if r["gold_answer_label"] == 0 and r["answer_pred_unfaithful"] == 1
                  else "FN" if r["gold_answer_label"] == 1 and r["answer_pred_unfaithful"] == 0
                  else "OTHER",
        axis=1,
    )
    errors["answer_length"] = errors["answer"].str.len()
    errors = errors.sort_values("answer_unfaithfulness_score", ascending=False)

    error_path = output_dir / "error_analysis.csv"
    cols_to_save = [
        "expanded_sample_id", "question_id", "source_model", "error_type",
        "gold_answer_label", "answer_pred_unfaithful",
        "answer_unfaithfulness_score", "answer_length", "n_claims",
    ]
    errors[cols_to_save].to_csv(error_path, index=False)
    logger.info(f"Saved error analysis: {error_path}")

    logger.info(f"\n{'='*50}")
    logger.info(f"EVALUATION SUMMARY")
    logger.info(f"{'='*50}")
    logger.info(f"Responses: {metrics['n_responses']}, Claims: {metrics['n_claims']}")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    logger.info(f"Macro F1: {metrics['f1_macro']:.4f}")
    logger.info(f"Unsupported F1: {metrics['f1_binary']:.4f}")
    logger.info(f"Unsupported Precision: {metrics['precision_unsupported']:.4f}")
    logger.info(f"Unsupported Recall: {metrics['recall_unsupported']:.4f}")
    logger.info(f"AUROC: {metrics['auroc']:.4f}")
    logger.info(f"AUPRC: {metrics['auprc']:.4f}")
    logger.info(f"Confusion Matrix [Unfaithful, Faithful]:")
    logger.info(f"  Gold Unfaithful: {cm[0].tolist()}")
    logger.info(f"  Gold Faithful:   {cm[1].tolist()}")

    return metrics


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    p = argparse.ArgumentParser(description="Evaluate MIL faithfulness on validation set")
    p.add_argument("--results_dir", type=str,
                   default="results/phase2_mil_faithfulness",
                   help="Training results directory")
    p.add_argument("--output_dir", type=str,
                   default=None,
                   help="Output directory (default: results_dir/evaluation)")
    p.add_argument("--models", nargs="+",
                   default=["Llama-2-7b-chat-hf", "Mistral-7B-Instruct-v0.3"])
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "evaluation"
    checkpoint_path = results_dir / "best_checkpoint.pt"
    config_path = results_dir / "best_config.json"

    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        sys.exit(1)

    metrics = evaluate(
        results_dir=results_dir,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        output_dir=output_dir,
        models=args.models,
    )

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
