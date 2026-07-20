"""
Aggregation strategies for NLI-based Faithfulness and Relevance detection.

These strategies aggregate claim-level NLI scores into answer-level predictions.

Key definitions:
- Faithfulness: All claims are supported by context
- Relevance: All claims address the question
- Reliability: faithful AND relevant
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Callable


# =============================================================================
# Faithfulness Aggregation Strategies
# =============================================================================

def aggregate_max_entail(
    scores_df: pd.DataFrame,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Strategy: Max Entailment
    
    Computes: score = max(entailment_probability) over all (claim, window) pairs
    
    If ANY claim has strong entailment from ANY context window, predict faithful.
    This is a lenient baseline - high recall for faithful, low precision.
    
    Args:
        scores_df: DataFrame with NLI scores
        threshold: Threshold for predicting faithful
        
    Returns:
        DataFrame with [case_id, faithfulness_score, faithfulness_pred]
    """
    result = scores_df.groupby("case_id").agg(
        faithfulness_score=("entailment_probability", "max"),
    ).reset_index()
    
    result["faithfulness_pred"] = (result["faithfulness_score"] >= threshold).astype(int)
    
    return result


def aggregate_entail_minus_contradiction(
    scores_df: pd.DataFrame,
    threshold: float = 0.0,
) -> pd.DataFrame:
    """
    Strategy: Entailment Minus Contradiction
    
    Computes: score = max(entailment) - max(contradiction)
    
    Balances support against contradiction evidence.
    Higher score = more support relative to contradiction.
    
    Args:
        scores_df: DataFrame with NLI scores
        threshold: Threshold for predicting faithful
        
    Returns:
        DataFrame with [case_id, faithfulness_score, faithfulness_pred]
    """
    result = scores_df.groupby("case_id").agg(
        max_entail=("entailment_probability", "max"),
        max_contrad=("contradiction_probability", "max"),
    ).reset_index()
    
    result["faithfulness_score"] = result["max_entail"] - result["max_contrad"]
    result["faithfulness_pred"] = (result["faithfulness_score"] >= threshold).astype(int)
    
    return result[["case_id", "faithfulness_score", "faithfulness_pred"]]


def aggregate_claim_min_support(
    scores_df: pd.DataFrame,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Strategy: Claim Min Support
    
    For each claim: score = max(entailment) over all windows
    Answer score: score = min(claim_score) over all claims
    
    This enforces "all claims must be supported" semantics.
    The weakest claim determines the overall score.
    
    Args:
        scores_df: DataFrame with NLI scores
        threshold: Threshold for predicting faithful
        
    Returns:
        DataFrame with [case_id, faithfulness_score, faithfulness_pred]
    """
    # Per-claim max entailment
    claim_max = scores_df.groupby(["case_id", "claim_id"]).agg(
        claim_support=("entailment_probability", "max"),
    ).reset_index()
    
    # Answer-level: minimum claim support
    result = claim_max.groupby("case_id").agg(
        faithfulness_score=("claim_support", "min"),
    ).reset_index()
    
    result["faithfulness_pred"] = (result["faithfulness_score"] >= threshold).astype(int)
    
    return result


def aggregate_contradiction_penalized_support(
    scores_df: pd.DataFrame,
    penalty_weight: float = 0.3,
    threshold: float = 0.0,
) -> pd.DataFrame:
    """
    Strategy: Contradiction Penalized Support
    
    For each claim: 
        claim_score = max(entailment) - penalty_weight * max(contradiction)
    
    Answer score: min(claim_score) over all claims
    
    Penalizes clear contradictions while rewarding entailment.
    
    Args:
        scores_df: DataFrame with NLI scores
        penalty_weight: Weight for contradiction penalty (0.0-1.0)
        threshold: Threshold for predicting faithful
        
    Returns:
        DataFrame with [case_id, faithfulness_score, faithfulness_pred]
    """
    # Per-claim scores
    claim_scores = scores_df.groupby(["case_id", "claim_id"]).agg(
        max_entail=("entailment_probability", "max"),
        max_contrad=("contradiction_probability", "max"),
    ).reset_index()
    
    # Apply penalty
    claim_scores["claim_score"] = (
        claim_scores["max_entail"] 
        - penalty_weight * claim_scores["max_contrad"]
    )
    
    # Answer-level: minimum claim score
    result = claim_scores.groupby("case_id").agg(
        faithfulness_score=("claim_score", "min"),
    ).reset_index()
    
    result["faithfulness_pred"] = (result["faithfulness_score"] >= threshold).astype(int)
    
    return result


# =============================================================================
# Relevance Aggregation Strategies
# =============================================================================

def aggregate_max_relevance(
    scores_df: pd.DataFrame,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Strategy: Max Relevance
    
    Computes: score = max(entailment) over all claims
    
    If ANY claim is relevant, predict relevant.
    
    Args:
        scores_df: DataFrame with relevance NLI scores
        threshold: Threshold for predicting relevant
        
    Returns:
        DataFrame with [case_id, relevance_score, relevance_pred]
    """
    result = scores_df.groupby("case_id").agg(
        relevance_score=("entailment_probability", "max"),
    ).reset_index()
    
    result["relevance_pred"] = (result["relevance_score"] >= threshold).astype(int)
    
    return result


def aggregate_mean_relevance(
    scores_df: pd.DataFrame,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Strategy: Mean Relevance
    
    Computes: score = mean(entailment) over all claims
    
    Averages relevance across all claims.
    
    Args:
        scores_df: DataFrame with relevance NLI scores
        threshold: Threshold for predicting relevant
        
    Returns:
        DataFrame with [case_id, relevance_score, relevance_pred]
    """
    result = scores_df.groupby("case_id").agg(
        relevance_score=("entailment_probability", "mean"),
    ).reset_index()
    
    result["relevance_pred"] = (result["relevance_score"] >= threshold).astype(int)
    
    return result


def aggregate_claim_min_relevance(
    scores_df: pd.DataFrame,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Strategy: Claim Min Relevance
    
    Answer score = min(claim_relevance) over all claims
    
    Enforces "all claims must be relevant".
    
    Args:
        scores_df: DataFrame with relevance NLI scores
        threshold: Threshold for predicting relevant
        
    Returns:
        DataFrame with [case_id, relevance_score, relevance_pred]
    """
    result = scores_df.groupby("case_id").agg(
        relevance_score=("entailment_probability", "min"),
    ).reset_index()
    
    result["relevance_pred"] = (result["relevance_score"] >= threshold).astype(int)
    
    return result


# =============================================================================
# Reliability: Faithfulness AND Relevance
# =============================================================================

def compute_reliability(
    faithfulness_df: pd.DataFrame,
    relevance_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute reliability = faithfulness AND relevance.
    
    Args:
        faithfulness_df: DataFrame with [case_id, faithfulness_pred]
        relevance_df: DataFrame with [case_id, relevance_pred]
        
    Returns:
        DataFrame with [case_id, reliability_pred]
    """
    merged = faithfulness_df.merge(
        relevance_df,
        on="case_id",
        how="outer",
    )
    
    merged["reliability_pred"] = (
        (merged["faithfulness_pred"] == 1) & 
        (merged["relevance_pred"] == 1)
    ).astype(int)
    
    return merged[["case_id", "reliability_pred"]]


# =============================================================================
# Registry
# =============================================================================

FAITHFULNESS_STRATEGIES: dict[str, Callable] = {
    "max_entail": aggregate_max_entail,
    "entail_minus_contradiction": aggregate_entail_minus_contradiction,
    "claim_min_support": aggregate_claim_min_support,
    "contradiction_penalized_support": aggregate_contradiction_penalized_support,
}

RELEVANCE_STRATEGIES: dict[str, Callable] = {
    "max_relevance": aggregate_max_relevance,
    "mean_relevance": aggregate_mean_relevance,
    "claim_min_relevance": aggregate_claim_min_relevance,
}


def apply_faithfulness_strategy(
    scores_df: pd.DataFrame,
    strategy: str,
    **kwargs,
) -> pd.DataFrame:
    """Apply a faithfulness aggregation strategy."""
    if strategy not in FAITHFULNESS_STRATEGIES:
        raise ValueError(f"Unknown faithfulness strategy: {strategy}")
    return FAITHFULNESS_STRATEGIES[strategy](scores_df, **kwargs)


def apply_relevance_strategy(
    scores_df: pd.DataFrame,
    strategy: str,
    **kwargs,
) -> pd.DataFrame:
    """Apply a relevance aggregation strategy."""
    if strategy not in RELEVANCE_STRATEGIES:
        raise ValueError(f"Unknown relevance strategy: {strategy}")
    return RELEVANCE_STRATEGIES[strategy](scores_df, **kwargs)
