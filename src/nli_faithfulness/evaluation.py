"""
Evaluation utilities for NLI Faithfulness baseline.

Includes threshold selection, metrics computation, subgroup evaluation,
and TF-IDF comparison.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)

from .constants import (
    THRESHOLD_RANGE_START,
    THRESHOLD_RANGE_END,
    THRESHOLD_RANGE_STEP,
    TFIDF_RESULTS_DIR,
)


logger = logging.getLogger(__name__)


# =============================================================================
# Metrics computation
# =============================================================================

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str = "",
) -> dict:
    """
    Compute comprehensive metrics for binary classification.
    
    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted labels (0 or 1)
        prefix: Optional prefix for metric names
        
    Returns:
        Dictionary with all metrics
    """
    metrics = {}
    
    # Basic metrics
    if prefix:
        prefix = f"{prefix}_"
    
    metrics[f"{prefix}accuracy"] = accuracy_score(y_true, y_pred)
    metrics[f"{prefix}f1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    metrics[f"{prefix}f1_positive"] = f1_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)
    metrics[f"{prefix}f1_negative"] = f1_score(y_true, y_pred, average="binary", pos_label=0, zero_division=0)
    
    metrics[f"{prefix}precision_positive"] = precision_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)
    metrics[f"{prefix}recall_positive"] = recall_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)
    
    metrics[f"{prefix}precision_negative"] = precision_score(y_true, y_pred, average="binary", pos_label=0, zero_division=0)
    metrics[f"{prefix}recall_negative"] = recall_score(y_true, y_pred, average="binary", pos_label=0, zero_division=0)
    
    return metrics


def compute_metrics_from_scores(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict:
    """
    Compute metrics from probability scores and a threshold.
    
    Args:
        y_true: Ground truth labels
        scores: Predicted probabilities
        threshold: Decision threshold
        
    Returns:
        Dictionary with all metrics
    """
    y_pred = (scores >= threshold).astype(int)
    return compute_metrics(y_true, y_pred)


# =============================================================================
# Threshold selection
# =============================================================================

def find_best_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
    metric: str = "f1_macro",
) -> tuple[float, dict]:
    """
    Find the best threshold based on a specific metric.
    
    Args:
        y_true: Ground truth labels
        scores: Predicted probabilities
        thresholds: Array of thresholds to try (default: 0.10 to 0.90 step 0.01)
        metric: Metric to optimize ("f1_macro", "f1_positive", "f1_negative")
        
    Returns:
        Tuple of (best_threshold, all_metrics_dict)
    """
    if thresholds is None:
        thresholds = np.round(
            np.arange(THRESHOLD_RANGE_START, THRESHOLD_RANGE_END, THRESHOLD_RANGE_STEP),
            2
        )
    
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    
    results = []
    
    for threshold in thresholds:
        y_pred = (scores >= threshold).astype(int)
        metrics = compute_metrics(y_true, y_pred)
        results.append({
            "threshold": threshold,
            **metrics,
        })
    
    results_df = pd.DataFrame(results)
    
    # Find best threshold
    best_idx = results_df[metric].idxmax()
    best_threshold = results_df.loc[best_idx, "threshold"]
    best_metrics = results_df.loc[best_idx].to_dict()
    
    # Check for ties
    tied = results_df[results_df[metric] == results_df[metric].max()]
    if len(tied) > 1:
        logger.info(f"Tied at {metric}={results_df[metric].max():.4f} for thresholds: {tied['threshold'].tolist()}")
        # Pick the middle threshold among ties
        best_threshold = tied["threshold"].median()
        best_metrics = results_df[results_df["threshold"] == best_threshold].iloc[0].to_dict()
    
    return float(best_threshold), best_metrics


def find_best_threshold_with_constraint(
    y_true: np.ndarray,
    scores: np.ndarray,
    min_f1_negative: float = 0.0,
    min_f1_positive: float = 0.0,
) -> tuple[float, dict]:
    """
    Find best threshold subject to minimum F1 constraints on both classes.
    
    This ensures balanced performance rather than optimizing only macro-F1.
    
    Args:
        y_true: Ground truth labels
        scores: Predicted probabilities
        min_f1_negative: Minimum acceptable F1 for negative class
        min_f1_positive: Minimum acceptable F1 for positive class
        
    Returns:
        Tuple of (best_threshold, metrics)
    """
    thresholds = np.round(
        np.arange(THRESHOLD_RANGE_START, THRESHOLD_RANGE_END, THRESHOLD_RANGE_STEP),
        2
    )
    
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    
    results = []
    
    for threshold in thresholds:
        y_pred = (scores >= threshold).astype(int)
        metrics = compute_metrics(y_true, y_pred)
        
        # Check constraints
        if (metrics["f1_negative"] >= min_f1_negative and 
            metrics["f1_positive"] >= min_f1_positive):
            results.append({
                "threshold": threshold,
                **metrics,
            })
    
    if not results:
        logger.warning("No thresholds satisfy constraints, falling back to best macro-F1")
        return find_best_threshold(y_true, scores)
    
    results_df = pd.DataFrame(results)
    best_idx = results_df["f1_macro"].idxmax()
    
    return float(results_df.loc[best_idx, "threshold"]), results_df.loc[best_idx].to_dict()


# =============================================================================
# Subgroup evaluation
# =============================================================================

def evaluate_subgroups(
    predictions_df: pd.DataFrame,
    samples_df: pd.DataFrame,
    group_column: str = "retrieval_config",
    label_column: str = "binary_faithfulness",
) -> pd.DataFrame:
    """
    Evaluate predictions across subgroups.
    
    Args:
        predictions_df: DataFrame with predictions (must include case_id and faithfulness_pred)
        samples_df: DataFrame with sample info (case_id, labels, subgroup columns)
        group_column: Column to group by (e.g., "retrieval_config", "joint_label")
        label_column: Name of the true label column in samples_df
        
    Returns:
        DataFrame with metrics for each subgroup
    """
    # Merge predictions with sample info
    merged = predictions_df.merge(
        samples_df[["case_id", label_column, group_column]],
        on="case_id",
        how="left",
    )
    
    # Group and compute metrics
    results = []
    
    for group_name, group_data in merged.groupby(group_column):
        y_true = group_data[label_column].values
        y_pred = group_data["faithfulness_pred"].values
        
        if len(y_true) == 0:
            continue
            
        metrics = compute_metrics(y_true, y_pred)
        metrics["subgroup"] = group_name
        metrics["n_samples"] = len(y_true)
        results.append(metrics)
    
    return pd.DataFrame(results)


def evaluate_all_subgroups(
    predictions_df: pd.DataFrame,
    samples_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Evaluate across all important subgroups.
    
    Args:
        predictions_df: DataFrame with predictions
        samples_df: DataFrame with sample info
        
    Returns:
        Dictionary mapping subgroup name to results DataFrame
    """
    subgroups = {}
    
    # By retrieval config
    if "retrieval_config" in samples_df.columns:
        subgroups["retrieval_config"] = evaluate_subgroups(
            predictions_df, samples_df, "retrieval_config"
        )
    
    # By joint label
    if "joint_label" in samples_df.columns:
        subgroups["joint_label"] = evaluate_subgroups(
            predictions_df, samples_df, "joint_label"
        )
    
    # By relevant but unfaithful (1_0)
    if "joint_label" in samples_df.columns:
        samples_df = samples_df.copy()
        samples_df["relevant_unfaithful"] = (
            samples_df["joint_label"] == "1_0"
        ).astype(int)
        subgroups["relevant_unfaithful"] = evaluate_subgroups(
            predictions_df, samples_df, "relevant_unfaithful"
        )
    
    return subgroups


# =============================================================================
# TF-IDF comparison
# =============================================================================

def load_tfidf_results() -> dict:
    """
    Load TF-IDF baseline results for comparison.
    
    Returns:
        Dictionary with TF-IDF metrics
    """
    tfidf_results = {}
    
    # Load selected model results
    selected_path = TFIDF_RESULTS_DIR / "selected_model_results.csv"
    if selected_path.exists():
        df = pd.read_csv(selected_path)
        
        # Get faithfulness results
        faith_df = df[df["target"] == "faithfulness"]
        if len(faith_df) > 0:
            # Use test results
            test_row = faith_df[faith_df["split"] == "test"]
            if len(test_row) > 0:
                row = test_row.iloc[0]
                tfidf_results["faithfulness_test"] = {
                    "accuracy": float(row["accuracy"]),
                    "f1_macro": float(row["f1_macro"]),
                    "f1_positive": float(row["f1_positive"]),
                    "f1_negative": float(row["f1_negative"]),
                    "precision_positive": float(row["precision_positive"]),
                    "recall_positive": float(row["recall_positive"]),
                    "precision_negative": float(row["precision_negative"]),
                    "recall_negative": float(row["recall_negative"]),
                }
    
    # Load stage2 summary
    summary_path = TFIDF_RESULTS_DIR / "stage2_summary.json"
    if summary_path.exists():
        with open(summary_path, "r") as f:
            summary = json.load(f)
            tfidf_results["summary"] = summary
    
    return tfidf_results


def compare_with_tfidf(
    nli_metrics: dict,
    split: str = "test",
) -> pd.DataFrame:
    """
    Compare NLI metrics with TF-IDF baseline.
    
    Args:
        nli_metrics: Dictionary with NLI metrics
        split: Which split to compare ("test" or "validation")
        
    Returns:
        DataFrame with comparison
    """
    tfidf_results = load_tfidf_results()
    
    rows = []
    
    # TF-IDF results
    if "faithfulness_test" in tfidf_results and split == "test":
        tfidf = tfidf_results["faithfulness_test"]
        rows.append({
            "method": "tfidf_baseline",
            "split": "test",
            "accuracy": tfidf["accuracy"],
            "f1_macro": tfidf["f1_macro"],
            "f1_positive": tfidf["f1_positive"],
            "f1_negative": tfidf["f1_negative"],
        })
    
    # NLI results
    nli_key = f"faithfulness_{split}"
    if nli_key in nli_metrics:
        nli = nli_metrics[nli_key]
        rows.append({
            "method": "nli_faithfulness",
            "split": split,
            "accuracy": nli.get("accuracy", 0),
            "f1_macro": nli.get("f1_macro", 0),
            "f1_positive": nli.get("f1_positive", 0),
            "f1_negative": nli.get("f1_negative", 0),
        })
    
    return pd.DataFrame(rows)


# =============================================================================
# Confusion matrix
# =============================================================================

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Confusion Matrix",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot a confusion matrix.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        title: Plot title
        save_path: Optional path to save the figure
        
    Returns:
        matplotlib Figure
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Unfaithful (0)", "Faithful (1)"],
        yticklabels=["Unfaithful (0)", "Faithful (1)"],
        ax=ax,
    )
    
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
        logger.info(f"Saved confusion matrix to {save_path}")
    
    return fig


# =============================================================================
# Results saving
# =============================================================================

def save_predictions(
    predictions_df: pd.DataFrame,
    scores_df: pd.DataFrame,
    split: str,
    output_dir: Path,
) -> None:
    """
    Save prediction results.
    
    Args:
        predictions_df: DataFrame with case_id, faithfulness_score, faithfulness_pred
        scores_df: DataFrame with detailed NLI scores
        split: Split name ("validation" or "test")
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Predictions
    predictions_df.to_csv(
        output_dir / f"{split}_predictions.csv",
        index=False,
    )
    
    # Detailed scores
    scores_df.to_csv(
        output_dir / f"chunk_window_scores.csv",
        index=False,
    )


def save_metrics(
    metrics: dict,
    split: str,
    output_dir: Path,
) -> None:
    """
    Save metrics to JSON.
    
    Args:
        metrics: Dictionary with metrics
        split: Split name
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / f"{split}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


def save_config(
    config: dict,
    output_dir: Path,
) -> None:
    """
    Save configuration to JSON.
    
    Args:
        config: Configuration dictionary
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
