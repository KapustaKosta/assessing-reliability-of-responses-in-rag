"""
Full validation pipeline for NLI-based Faithfulness and Relevance detection.

This script:
1. Loads RAGognize dataset with adapter
2. Splits answers into claims
3. Runs Faithfulness NLI (context window → claim)
4. Runs Relevance NLI (question → claim)
5. Applies aggregation strategies
6. Computes Reliability = Faithfulness AND Relevance
7. Searches for optimal thresholds
8. Reports comprehensive metrics
"""

import sys
sys.path.insert(0, 'src')

import os
import json
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from ragognize_adapter import (
    RAGognizeAdapter, load_ragognize_dataset,
    create_train_val_split, apply_split, AVAILABLE_MODELS,
)
from nli_faithfulness import (
    DEFAULT_MODEL_NAME, CACHE_DIR, RESULTS_DIR,
    segment_dataset,
    NLIModel, batch_inference,
    apply_faithfulness_strategy, apply_relevance_strategy, compute_reliability,
    FAITHFULNESS_STRATEGIES, RELEVANCE_STRATEGIES,
)
from nli_faithfulness.data import SampleData, ChunkData

from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
    accuracy_score, balanced_accuracy_score,
    roc_auc_score, average_precision_score,
)


def compute_metrics(y_true, y_pred, y_score=None):
    """Compute comprehensive metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average='macro', zero_division=0),
        "f1_positive": f1_score(y_true, y_pred, average='binary', pos_label=1, zero_division=0),
        "f1_negative": f1_score(y_true, y_pred, average='binary', pos_label=0, zero_division=0),
        "precision_positive": precision_score(y_true, y_pred, average='binary', pos_label=1, zero_division=0),
        "recall_positive": recall_score(y_true, y_pred, average='binary', pos_label=1, zero_division=0),
        "precision_negative": precision_score(y_true, y_pred, average='binary', pos_label=0, zero_division=0),
        "recall_negative": recall_score(y_true, y_pred, average='binary', pos_label=0, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    
    # AUROC and AUPRC if scores provided
    if y_score is not None:
        try:
            metrics["auroc"] = roc_auc_score(y_true, y_score)
        except:
            metrics["auroc"] = None
        try:
            metrics["auprc"] = average_precision_score(y_true, y_score)
        except:
            metrics["auprc"] = None
    
    return metrics


def search_best_thresholds(faith_scores, rel_scores, y_faith, y_rel, y_reliab):
    """Search for best thresholds."""
    best_f1 = 0
    best_config = {}
    
    faith_thresholds = np.arange(0.1, 0.9, 0.05)
    rel_thresholds = np.arange(0.1, 0.9, 0.05)
    
    for f_th in faith_thresholds:
        for r_th in rel_thresholds:
            faith_pred = (faith_scores >= f_th).astype(int)
            rel_pred = (rel_scores >= r_th).astype(int)
            reliab_pred = (faith_pred == 1) & (rel_pred == 1)
            
            f1 = f1_score(y_reliab, reliab_pred.astype(int), average='macro', zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_config = {
                    "faith_threshold": float(f_th),
                    "rel_threshold": float(r_th),
                    "f1_macro": float(f1),
                }
    
    return best_config


def main():
    print("=" * 60)
    print("FULL VALIDATION: Faithfulness + Relevance + Reliability")
    print("=" * 60)
    
    # Setup
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load data
    print("\n1. Loading data...")
    raw = load_ragognize_dataset()
    split_info = create_train_val_split(raw, val_size=0.15, seed=42)
    adapter = RAGognizeAdapter(models=AVAILABLE_MODELS)
    raw_split = apply_split(raw, split_info)
    unified = adapter.transform_dataset(raw_split)
    
    val_data = [unified['val'][i] for i in range(len(unified['val']))]
    print(f"   Validation samples: {len(val_data)}")
    
    # Convert to SampleData
    def to_sample(d):
        chunks = [ChunkData(
            case_id=d['case_id'], chunk_id=i+1, chunk_rank=i+1,
            chunk_text=t, retrieval_config='top_1', is_available=True,
        ) for i, t in enumerate(d['chunks'])]
        return SampleData(
            case_id=d['case_id'],
            question=d['question'],
            answer=d['answer'],
            binary_faithfulness=d['faithfulness_label'],
            binary_relevancy=d['answerable'],
            chunks=chunks,
        )
    
    samples = [to_sample(unified['val'][i]) for i in range(len(unified['val']))]
    unified_lookup = {unified['val'][i]['case_id']: unified['val'][i] for i in range(len(unified['val']))}
    
    # Ground truth
    y_faith = np.array([1 if s.binary_faithfulness else 0 for s in samples])
    y_rel = np.array([1 if s.binary_relevancy else 0 for s in samples])
    y_reliab = np.array([(yf == 1) and (yr == 1) for yf, yr in zip(y_faith, y_rel)])
    case_ids = [s.case_id for s in samples]
    
    print(f"   Faithful: {sum(y_faith)}/{len(y_faith)}")
    print(f"   Relevant: {sum(y_rel)}/{len(y_rel)}")
    print(f"   Reliable: {sum(y_reliab)}/{len(y_reliab)}")
    
    # Load model
    print("\n2. Loading NLI model...")
    model = NLIModel(model_name=DEFAULT_MODEL_NAME)
    model.save_model_info()
    
    # Segment answers
    print("\n3. Segmenting answers into claims...")
    segments = segment_dataset(samples)
    total_claims = sum(len(seg) for seg in segments.values())
    print(f"   Total claims: {total_claims}")
    print(f"   Avg claims/sample: {total_claims/len(samples):.1f}")
    
    # Run Faithfulness NLI
    print("\n4. Running Faithfulness NLI...")
    faith_cache = CACHE_DIR / f"ragognize_val_faithfulness_{timestamp}.csv"
    start = time.time()
    faith_scores = batch_inference(
        model, samples, segments,
        batch_size=8,
        cache_path=faith_cache,
        verbose=True,
        task_type="faithfulness",
    )
    faith_time = time.time() - start
    print(f"   Time: {faith_time:.1f}s, Pairs: {len(faith_scores)}")
    
    # Run Relevance NLI
    print("\n5. Running Relevance NLI...")
    rel_cache = CACHE_DIR / f"ragognize_val_relevance_{timestamp}.csv"
    start = time.time()
    rel_scores = batch_inference(
        model, samples, segments,
        batch_size=8,
        cache_path=rel_cache,
        verbose=True,
        task_type="relevance",
    )
    rel_time = time.time() - start
    print(f"   Time: {rel_time:.1f}s, Pairs: {len(rel_scores)}")
    
    # Apply aggregations
    print("\n6. Applying aggregation strategies...")
    
    # Use max_entail for faithfulness (simple baseline)
    faith_agg = apply_faithfulness_strategy(faith_scores, "max_entail", threshold=0.5)
    
    # Use mean_relevance for relevance
    rel_agg = apply_relevance_strategy(rel_scores, "mean_relevance", threshold=0.5)
    
    # Compute reliability
    reliab_pred = compute_reliability(faith_agg, rel_agg)
    
    # Search best thresholds
    print("\n7. Searching best thresholds...")
    best_config = search_best_thresholds(
        faith_agg.set_index("case_id").loc[case_ids]["faithfulness_score"].values,
        rel_agg.set_index("case_id").loc[case_ids]["relevance_score"].values,
        y_faith, y_rel, y_reliab,
    )
    print(f"   Best config: {best_config}")
    
    # Apply best thresholds
    f_th = best_config["faith_threshold"]
    r_th = best_config["rel_threshold"]
    
    faith_pred = (faith_agg.set_index("case_id").loc[case_ids]["faithfulness_score"].values >= f_th).astype(int)
    rel_pred = (rel_agg.set_index("case_id").loc[case_ids]["relevance_score"].values >= r_th).astype(int)
    reliab_final = (faith_pred == 1) & (rel_pred == 1)
    
    # Compute metrics
    print("\n8. Computing metrics...")
    
    # Faithfulness metrics
    faith_metrics = compute_metrics(
        y_faith, faith_pred,
        faith_agg.set_index("case_id").loc[case_ids]["faithfulness_score"].values,
    )
    
    # Relevance metrics
    rel_metrics = compute_metrics(
        y_rel, rel_pred,
        rel_agg.set_index("case_id").loc[case_ids]["relevance_score"].values,
    )
    
    # Reliability metrics
    reliab_metrics = compute_metrics(
        y_reliab, reliab_final.astype(int),
        (faith_agg.set_index("case_id").loc[case_ids]["faithfulness_score"].values + 
         rel_agg.set_index("case_id").loc[case_ids]["relevance_score"].values) / 2,
    )
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print(f"\nFaithfulness Metrics (threshold={f_th:.2f}):")
    print(f"  Accuracy:     {faith_metrics['accuracy']:.4f}")
    print(f"  Balanced:    {faith_metrics['balanced_accuracy']:.4f}")
    print(f"  F1 Macro:   {faith_metrics['f1_macro']:.4f}")
    print(f"  F1 Positive: {faith_metrics['f1_positive']:.4f}")
    print(f"  F1 Negative: {faith_metrics['f1_negative']:.4f}")
    print(f"  AUROC:       {faith_metrics.get('auroc', 'N/A')}")
    print(f"  AUPRC:       {faith_metrics.get('auprc', 'N/A')}")
    cm = faith_metrics['confusion_matrix']
    print(f"  CM: [[TN={cm[0][0]}, FP={cm[0][1]}], [FN={cm[1][0]}, TP={cm[1][1]}]]")
    
    print(f"\nRelevance Metrics (threshold={r_th:.2f}):")
    print(f"  Accuracy:     {rel_metrics['accuracy']:.4f}")
    print(f"  Balanced:    {rel_metrics['balanced_accuracy']:.4f}")
    print(f"  F1 Macro:   {rel_metrics['f1_macro']:.4f}")
    print(f"  F1 Positive: {rel_metrics['f1_positive']:.4f}")
    print(f"  F1 Negative: {rel_metrics['f1_negative']:.4f}")
    print(f"  AUROC:       {rel_metrics.get('auroc', 'N/A')}")
    print(f"  AUPRC:       {rel_metrics.get('auprc', 'N/A')}")
    cm = rel_metrics['confusion_matrix']
    print(f"  CM: [[TN={cm[0][0]}, FP={cm[0][1]}], [FN={cm[1][0]}, TP={cm[1][1]}]]")
    
    print(f"\nReliability Metrics (Reliability = Faithful AND Relevant):")
    print(f"  Accuracy:     {reliab_metrics['accuracy']:.4f}")
    print(f"  Balanced:    {reliab_metrics['balanced_accuracy']:.4f}")
    print(f"  F1 Macro:   {reliab_metrics['f1_macro']:.4f}")
    print(f"  F1 Positive: {reliab_metrics['f1_positive']:.4f}")
    print(f"  F1 Negative: {reliab_metrics['f1_negative']:.4f}")
    print(f"  AUROC:       {reliab_metrics.get('auroc', 'N/A')}")
    print(f"  AUPRC:       {reliab_metrics.get('auprc', 'N/A')}")
    cm = reliab_metrics['confusion_matrix']
    print(f"  CM: [[TN={cm[0][0]}, FP={cm[0][1]}], [FN={cm[1][0]}, TP={cm[1][1]}]]")
    
    # Save outputs
    print("\n9. Saving outputs...")
    
    # Validation predictions
    val_pred_df = pd.DataFrame({
        "case_id": case_ids,
        "y_faithful": y_faith,
        "y_relevant": y_rel,
        "y_reliable": y_reliab,
        "faithfulness_score": faith_agg.set_index("case_id").loc[case_ids]["faithfulness_score"].values,
        "faithfulness_pred": faith_pred,
        "relevance_score": rel_agg.set_index("case_id").loc[case_ids]["relevance_score"].values,
        "relevance_pred": rel_pred,
        "reliability_pred": reliab_final.astype(int),
    })
    val_pred_df.to_csv(RESULTS_DIR / "validation_predictions.csv", index=False)
    
    # Claim-window predictions
    faith_scores.to_csv(RESULTS_DIR / "faithfulness_claim_window.csv", index=False)
    rel_scores.to_csv(RESULTS_DIR / "relevance_claim_window.csv", index=False)
    
    # Metrics
    all_metrics = {
        "timestamp": timestamp,
        "model": DEFAULT_MODEL_NAME,
        "best_config": best_config,
        "faithfulness_metrics": faith_metrics,
        "relevance_metrics": rel_metrics,
        "reliability_metrics": reliab_metrics,
        "faithfulness_threshold": f_th,
        "relevance_threshold": r_th,
        "total_samples": len(samples),
        "total_claims": total_claims,
        "faithfulness_time": faith_time,
        "relevance_time": rel_time,
    }
    
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    
    # Best config
    with open(RESULTS_DIR / "best_config.json", "w") as f:
        json.dump(best_config, f, indent=2)
    
    print(f"\n   Saved to {RESULTS_DIR}")
    
    # Error analysis
    print("\n10. Error analysis...")
    errors = val_pred_df[val_pred_df["y_reliable"] != val_pred_df["reliability_pred"]]
    errors.to_csv(RESULTS_DIR / "error_analysis.csv", index=False)
    print(f"   Errors: {len(errors)}/{len(val_pred_df)} ({100*len(errors)/len(val_pred_df):.1f}%)")
    
    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)
    
    return all_metrics


if __name__ == "__main__":
    main()
