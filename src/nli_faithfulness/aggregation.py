"""
Aggregation strategies for NLI-based Faithfulness detection.

These strategies aggregate sentence-level NLI scores into sample-level
faithfulness predictions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Callable

from .constants import (
    DEFAULT_ENTAILMENT_THRESHOLD,
    DEFAULT_CONTRADICTION_THRESHOLD,
    DEFAULT_SENTENCE_SUPPORT_THRESHOLD,
)


# =============================================================================
# Strategy A: Whole Answer Max Entailment
# =============================================================================

def aggregate_whole_answer_max_entail(
    scores_df: pd.DataFrame,
    threshold: float = DEFAULT_ENTAILMENT_THRESHOLD,
) -> pd.DataFrame:
    """
    Strategy A: Whole answer max entailment.
    
    Treats the entire answer as a single unit and computes:
        score = max(p_entailment) over all (sentence, chunk, window) pairs
    
    This is the simplest baseline - if any chunk window supports any part
    of the answer with high entailment, predict faithful.
    
    Args:
        scores_df: DataFrame with columns [case_id, sentence_id, chunk_id, 
                   window_id, p_entailment, p_neutral, p_contradiction]
        threshold: Threshold for predicting faithful
        
    Returns:
        DataFrame with columns [case_id, faithfulness_score, faithfulness_pred]
    """
    # For each case, get max entailment
    result = scores_df.groupby("case_id").agg(
        faithfulness_score=("p_entailment", "max"),
    ).reset_index()
    
    result["faithfulness_pred"] = (result["faithfulness_score"] >= threshold).astype(int)
    
    return result


# =============================================================================
# Strategy B: Whole Answer Entailment Minus Contradiction
# =============================================================================

def aggregate_whole_answer_entail_minus_contrad(
    scores_df: pd.DataFrame,
    threshold: float = 0.0,
) -> pd.DataFrame:
    """
    Strategy B: Whole answer entailment minus contradiction.
    
    Computes:
        score = max(p_entailment) - max(p_contradiction)
    
    This considers both supporting and contradicting evidence.
    A higher score means more support relative to contradiction.
    
    Args:
        scores_df: DataFrame with columns [case_id, p_entailment, p_contradiction]
        threshold: Threshold for predicting faithful
        
    Returns:
        DataFrame with columns [case_id, faithfulness_score, faithfulness_pred]
    """
    result = scores_df.groupby("case_id").agg(
        max_entail=("p_entailment", "max"),
        max_contrad=("p_contradiction", "max"),
    ).reset_index()
    
    result["faithfulness_score"] = result["max_entail"] - result["max_contrad"]
    result["faithfulness_pred"] = (result["faithfulness_score"] >= threshold).astype(int)
    
    # Drop intermediate columns
    result = result.drop(columns=["max_entail", "max_contrad"])
    
    return result


# =============================================================================
# Strategy C: Sentence Min Support
# =============================================================================

def aggregate_sentence_min_support(
    scores_df: pd.DataFrame,
    threshold: float = DEFAULT_SENTENCE_SUPPORT_THRESHOLD,
) -> pd.DataFrame:
    """
    Strategy C: Sentence min support.
    
    For each sentence, computes:
        sentence_support = max(p_entailment) over all (chunk, window) pairs
    
    Then the sample score is:
        sample_score = min(sentence_support) over all sentences
    
    This approximates the "all facts must be supported" semantics by requiring
    every sentence to have at least some support from some chunk.
    
    Args:
        scores_df: DataFrame with columns [case_id, sentence_id, p_entailment]
        threshold: Threshold for considering a sentence as "supported"
        
    Returns:
        DataFrame with columns [case_id, faithfulness_score, faithfulness_pred]
    """
    # Step 1: For each (case, sentence), get max entailment across chunks/windows
    sentence_max = scores_df.groupby(["case_id", "sentence_id"]).agg(
        sentence_support=("p_entailment", "max"),
    ).reset_index()
    
    # Step 2: For each case, get the minimum sentence support
    # This represents the "weakest link"
    result = sentence_max.groupby("case_id").agg(
        faithfulness_score=("sentence_support", "min"),
    ).reset_index()
    
    result["faithfulness_pred"] = (result["faithfulness_score"] >= threshold).astype(int)
    
    return result


# =============================================================================
# Strategy D: Sentence Fraction Supported
# =============================================================================

def aggregate_sentence_fraction_supported(
    scores_df: pd.DataFrame,
    sentence_threshold: float = DEFAULT_SENTENCE_SUPPORT_THRESHOLD,
    min_fraction: float = 1.0,
) -> pd.DataFrame:
    """
    Strategy D: Sentence fraction supported.
    
    For each sentence, determines if it's supported:
        sentence_supported = (max(p_entailment) >= sentence_threshold)
    
    Then computes:
        fraction_supported = supported_sentences / total_sentences
        sample_score = fraction_supported
        sample_pred = (fraction_supported >= min_fraction)
    
    Args:
        scores_df: DataFrame with columns [case_id, sentence_id, p_entailment]
        sentence_threshold: Threshold for considering a sentence as "supported"
        min_fraction: Minimum fraction of sentences that must be supported (0.0-1.0)
                    Set to 1.0 for strict "all sentences supported"
                    
    Returns:
        DataFrame with columns [case_id, faithfulness_score, faithfulness_pred]
    """
    # Step 1: For each (case, sentence), get max entailment
    sentence_max = scores_df.groupby(["case_id", "sentence_id"]).agg(
        sentence_support=("p_entailment", "max"),
    ).reset_index()
    
    # Step 2: Determine if each sentence is supported
    sentence_max["sentence_supported"] = (
        sentence_max["sentence_support"] >= sentence_threshold
    ).astype(int)
    
    # Step 3: For each case, compute fraction supported
    result = sentence_max.groupby("case_id").agg(
        faithfulness_score=("sentence_supported", "mean"),
        total_sentences=("sentence_id", "count"),
        supported_sentences=("sentence_supported", "sum"),
    ).reset_index()
    
    result["faithfulness_pred"] = (result["faithfulness_score"] >= min_fraction).astype(int)
    
    # Drop intermediate columns
    result = result.drop(columns=["total_sentences", "supported_sentences"])
    
    return result


# =============================================================================
# Strategy E: Sentence Support with Contradiction Penalty
# =============================================================================

def aggregate_sentence_support_with_contradiction_penalty(
    scores_df: pd.DataFrame,
    entailment_threshold: float = DEFAULT_ENTAILMENT_THRESHOLD,
    contradiction_penalty: float = 0.3,
    sample_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Strategy E: Sentence support with contradiction penalty.
    
    For each sentence, computes:
        sentence_entail = max(p_entailment) over all (chunk, window)
        sentence_contrad = max(p_contradiction) over all (chunk, window)
        sentence_score = sentence_entail - contradiction_penalty * sentence_contrad
    
    For the sample:
        - If any sentence has sentence_contrad > 0.7, apply strong penalty
        - Otherwise, use min(sentence_score) as sample score
    
    Formula:
        sample_score = min(sentence_entail) 
                      - contradiction_penalty * max(sentence_contrad)
    
    This penalizes clear contradictions more than neutral/missing evidence.
    
    Args:
        scores_df: DataFrame with columns [case_id, sentence_id, 
                   p_entailment, p_contradiction]
        entailment_threshold: Threshold for considering a sentence as "supported"
        contradiction_penalty: Weight for contradiction penalty (0.0-1.0)
        sample_threshold: Threshold for predicting faithful
        
    Returns:
        DataFrame with columns [case_id, faithfulness_score, faithfulness_pred]
    """
    # Step 1: For each (case, sentence), get max entailment and max contradiction
    sentence_scores = scores_df.groupby(["case_id", "sentence_id"]).agg(
        sentence_entail=("p_entailment", "max"),
        sentence_contrad=("p_contradiction", "max"),
    ).reset_index()
    
    # Step 2: Compute sentence-level score with contradiction penalty
    sentence_scores["sentence_score"] = (
        sentence_scores["sentence_entail"] 
        - contradiction_penalty * sentence_scores["sentence_contrad"]
    )
    
    # Step 3: For each case, get the minimum sentence score
    # This represents the "weakest link" after penalty
    result = sentence_scores.groupby("case_id").agg(
        faithfulness_score=("sentence_score", "min"),
        max_sentence_contrad=("sentence_contrad", "max"),
        min_sentence_entail=("sentence_entail", "min"),
    ).reset_index()
    
    result["faithfulness_pred"] = (result["faithfulness_score"] >= sample_threshold).astype(int)
    
    # Drop intermediate columns
    result = result.drop(columns=["max_sentence_contrad", "min_sentence_entail"])
    
    return result


# =============================================================================
# Helper functions
# =============================================================================

AGGREGATION_STRATEGIES: dict[str, Callable] = {
    "whole_answer_max_entail": aggregate_whole_answer_max_entail,
    "whole_answer_entail_minus_contrad": aggregate_whole_answer_entail_minus_contrad,
    "sentence_min_support": aggregate_sentence_min_support,
    "sentence_fraction_supported": aggregate_sentence_fraction_supported,
    "sentence_support_with_contradiction_penalty": aggregate_sentence_support_with_contradiction_penalty,
}


def apply_aggregation_strategy(
    scores_df: pd.DataFrame,
    strategy: str,
    **kwargs,
) -> pd.DataFrame:
    """
    Apply an aggregation strategy by name.
    
    Args:
        scores_df: DataFrame with NLI scores
        strategy: Name of the strategy
        **kwargs: Additional arguments for the strategy function
        
    Returns:
        DataFrame with faithfulness scores and predictions
    """
    if strategy not in AGGREGATION_STRATEGIES:
        raise ValueError(
            f"Unknown strategy: {strategy}. "
            f"Available: {list(AGGREGATION_STRATEGIES.keys())}"
        )
    
    return AGGREGATION_STRATEGIES[strategy](scores_df, **kwargs)


def compare_strategies(
    scores_df: pd.DataFrame,
    strategies: list[str],
    y_true: pd.Series,
) -> pd.DataFrame:
    """
    Compare multiple aggregation strategies.
    
    Args:
        scores_df: DataFrame with NLI scores
        strategies: List of strategy names to compare
        y_true: Ground truth labels (indexed by case_id)
        
    Returns:
        DataFrame with comparison results for each strategy
    """
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
    
    results = []
    
    for strategy in strategies:
        try:
            preds_df = apply_aggregation_strategy(scores_df, strategy)
            
            # Align with ground truth
            merged = preds_df.merge(
                y_true.reset_index().rename(columns={"index": "case_id", 0: "y_true"}),
                on="case_id",
            )
            
            y_true_vals = merged["y_true"]
            y_pred_vals = merged["faithfulness_pred"]
            
            results.append({
                "strategy": strategy,
                "accuracy": accuracy_score(y_true_vals, y_pred_vals),
                "f1_macro": f1_score(y_true_vals, y_pred_vals, average="macro", zero_division=0),
                "f1_positive": f1_score(y_true_vals, y_pred_vals, average="binary", pos_label=1, zero_division=0),
                "f1_negative": f1_score(y_true_vals, y_pred_vals, average="binary", pos_label=0, zero_division=0),
                "precision_positive": precision_score(y_true_vals, y_pred_vals, average="binary", pos_label=1, zero_division=0),
                "recall_positive": recall_score(y_true_vals, y_pred_vals, average="binary", pos_label=1, zero_division=0),
            })
        except Exception as e:
            results.append({
                "strategy": strategy,
                "error": str(e),
            })
    
    return pd.DataFrame(results)


# =============================================================================
# Evidence extraction helpers
# =============================================================================

def get_best_evidence(
    scores_df: pd.DataFrame,
    case_id: str,
) -> dict:
    """
    Get the best supporting evidence for a specific case.
    
    Args:
        scores_df: DataFrame with NLI scores
        case_id: The case to get evidence for
        
    Returns:
        Dict with best entailment and contradiction evidence
    """
    case_scores = scores_df[scores_df["case_id"] == case_id]
    
    if len(case_scores) == 0:
        return {}
    
    # Best entailment
    best_entail = case_scores.loc[case_scores["p_entailment"].idxmax()]
    
    # Best contradiction
    best_contrad = case_scores.loc[case_scores["p_contradiction"].idxmax()]
    
    return {
        "best_entailment": {
            "sentence_id": int(best_entail["sentence_id"]),
            "chunk_id": int(best_entail["chunk_id"]),
            "window_id": int(best_entail["window_id"]),
            "p_entailment": float(best_entail["p_entailment"]),
            "hypothesis": best_entail["hypothesis"],
            "premise": best_entail["premise"],
        },
        "best_contradiction": {
            "sentence_id": int(best_contrad["sentence_id"]),
            "chunk_id": int(best_contrad["chunk_id"]),
            "window_id": int(best_contrad["window_id"]),
            "p_contradiction": float(best_contrad["p_contradiction"]),
            "hypothesis": best_contrad["hypothesis"],
            "premise": best_contrad["premise"],
        },
    }
