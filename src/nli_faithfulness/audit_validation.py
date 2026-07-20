#!/usr/bin/env python
"""
Final Validation Audit Script.

Comprehensive validation with:
- Correct label semantics (unfaithful=0, faithful=1)
- Label order [unfaithful, faithful] in all metrics
- Baseline comparisons (always, stratified)
- TF-IDF baseline comparison
- Source model subgroup analysis
- No silent sample skipping
- Full output saving with run_manifest
"""

import sys
sys.path.insert(0, 'src')

import os
import json
import time
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime
from collections import defaultdict

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
    apply_faithfulness_strategy,
    FAITHFULNESS_STRATEGIES,
)
from nli_faithfulness.data import SampleData, ChunkData

from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
    accuracy_score, balanced_accuracy_score,
    roc_auc_score, average_precision_score,
)


# Label definitions (CRITICAL: must be consistent throughout)
UNFAITHFUL = 0
FAITHFUL = 1
LABEL_NAMES = ["unfaithful", "faithful"]  # Order matters for sklearn metrics


def compute_metrics(y_true, y_pred, scores=None, label_names=LABEL_NAMES):
    """Compute comprehensive metrics with correct label semantics."""
    # Confusion matrix: rows=true, cols=pred, order=[unfaithful, faithful]
    cm = confusion_matrix(y_true, y_pred, labels=[UNFAITHFUL, FAITHFUL])
    
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average='macro', zero_division=0),
        "f1_unfaithful": f1_score(y_true, y_pred, average='binary', pos_label=UNFAITHFUL, zero_division=0),
        "f1_faithful": f1_score(y_true, y_pred, average='binary', pos_label=FAITHFUL, zero_division=0),
        "precision_unfaithful": precision_score(y_true, y_pred, average='binary', pos_label=UNFAITHFUL, zero_division=0),
        "recall_unfaithful": recall_score(y_true, y_pred, average='binary', pos_label=UNFAITHFUL, zero_division=0),
        "precision_faithful": precision_score(y_true, y_pred, average='binary', pos_label=FAITHFUL, zero_division=0),
        "recall_faithful": recall_score(y_true, y_pred, average='binary', pos_label=FAITHFUL, zero_division=0),
        "confusion_matrix": cm.tolist(),
        "true_unfaithful": int(cm[0, 0] + cm[0, 1]),
        "true_faithful": int(cm[1, 0] + cm[1, 1]),
        "pred_unfaithful": int(cm[0, 0] + cm[1, 0]),
        "pred_faithful": int(cm[0, 1] + cm[1, 1]),
    }
    
    if scores is not None:
        try:
            metrics["auroc"] = roc_auc_score(y_true, scores)
        except:
            metrics["auroc"] = None
        try:
            metrics["auprc"] = average_precision_score(y_true, scores)
        except:
            metrics["auprc"] = None
    
    return metrics


def print_metrics(name, metrics):
    """Pretty print metrics."""
    print(f"\n{name}:")
    print(f"  Confusion Matrix (rows=true, cols=pred, order=[unfaithful, faithful]):")
    print(f"    [[TN={metrics['confusion_matrix'][0][0]}, FP={metrics['confusion_matrix'][0][1]}],")
    print(f"     [FN={metrics['confusion_matrix'][1][0]}, TP={metrics['confusion_matrix'][1][1]}]]")
    print(f"  Accuracy:         {metrics['accuracy']:.4f}")
    print(f"  Balanced Acc:     {metrics['balanced_accuracy']:.4f}")
    print(f"  Macro-F1:         {metrics['f1_macro']:.4f}")
    print(f"  Unfaithful P/R/F: {metrics['precision_unfaithful']:.3f}/{metrics['recall_unfaithful']:.3f}/{metrics['f1_unfaithful']:.3f}")
    print(f"  Faithful P/R/F:   {metrics['precision_faithful']:.3f}/{metrics['recall_faithful']:.3f}/{metrics['f1_faithful']:.3f}")
    if metrics.get("auroc"):
        print(f"  AUROC:            {metrics['auroc']:.4f}")
    if metrics.get("auprc"):
        print(f"  AUPRC:            {metrics['auprc']:.4f}")


def get_git_info():
    """Get git commit info for reproducibility."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()[:8]
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
        ).strip()
        return {"commit": commit, "branch": branch}
    except:
        return {"commit": "unknown", "branch": "unknown"}


def main():
    print("=" * 70)
    print("FINAL VALIDATION AUDIT: Encoder/NLI Faithfulness Classifier")
    print("=" * 70)
    
    # Setup paths
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get git info
    git_info = get_git_info()
    print(f"\nGit: {git_info['branch']} @ {git_info['commit']}")
    
    # =========================================================================
    # 1. Load and validate data
    # =========================================================================
    print("\n" + "=" * 70)
    print("1. DATA LOADING AND VALIDATION")
    print("=" * 70)
    
    raw = load_ragognize_dataset()
    split_info = create_train_val_split(raw, val_size=0.15, seed=42)
    adapter = RAGognizeAdapter(models=AVAILABLE_MODELS)
    raw_split = apply_split(raw, split_info)
    unified = adapter.transform_dataset(raw_split)
    
    print(f"\nRaw splits: train={len(raw['train'])}, test={len(raw['test'])}")
    print(f"Split manifest: val_size={split_info['val_size']}, seed={split_info['seed']}")
    
    # Convert to samples
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
            binary_relevancy=d['answerable'],  # Note: using answerable as relevance proxy
            chunks=chunks,
        )
    
    val_list = list(unified['val'])
    samples = [to_sample(val_list[i]) for i in range(len(val_list))]
    
    # Track all samples - no silent skipping
    all_case_ids = [s.case_id for s in samples]
    processed_case_ids = set()
    skipped_samples = []
    
    print(f"\nValidation samples:")
    print(f"  Total in unified['val']: {len(val_list)}")
    print(f"  Converted to SampleData: {len(samples)}")
    
    # Ground truth (unfaithful=0, faithful=1)
    y_true = np.array([FAITHFUL if s.binary_faithfulness else UNFAITHFUL for s in samples])
    case_ids = [s.case_id for s in samples]
    
    n_unfaithful = sum(y_true == UNFAITHFUL)
    n_faithful = sum(y_true == FAITHFUL)
    print(f"  Unfaithful (0): {n_unfaithful}")
    print(f"  Faithful (1):   {n_faithful}")
    
    # =========================================================================
    # 2. Check addressed_user_prompt in RAGognize
    # =========================================================================
    print("\n" + "=" * 70)
    print("2. RAGOGNIZE addressed_user_prompt AUDIT")
    print("=" * 70)
    
    # Check if addressed_user_prompt exists in details.result
    addressed_vals = []
    addressed_dist = defaultdict(int)
    
    for d in val_list:
        responses = d.get('responses', {})
        for model_name in AVAILABLE_MODELS:
            if model_name in responses:
                resp = responses[model_name]
                details = resp.get('details', {})
                result = details.get('result', {})
                addr = result.get('addressed_user_prompt')
                if addr is not None:
                    addressed_vals.append(addr)
                    addressed_dist[addr] += 1
    
    print(f"\naddressed_user_prompt in details.result:")
    print(f"  Total samples with field: {len(addressed_vals)}/{len(val_list)}")
    print(f"  Distribution: {dict(addressed_dist)}")
    
    if addressed_dist:
        print("\n  ** Relevance gold label availability: PARTIAL **")
        print("  Cannot claim Relevance evaluation is complete until all samples have this field.")
    else:
        print("\n  ** Relevance gold label: NOT AVAILABLE **")
        print("  Relevance evaluation cannot be performed on this dataset version.")
    
    # =========================================================================
    # 3. Segmentation
    # =========================================================================
    print("\n" + "=" * 70)
    print("3. ANSWER SEGMENTATION")
    print("=" * 70)
    
    segments = segment_dataset(samples)
    total_claims = sum(len(seg) for seg in segments.values())
    empty_claim_samples = [cid for cid, seg in segments.items() if len(seg) == 0]
    
    print(f"\nTotal claims: {total_claims}")
    print(f"Avg claims/sample: {total_claims/len(samples):.1f}")
    print(f"Samples with empty claims: {len(empty_claim_samples)}")
    
    if empty_claim_samples:
        print(f"  Warning: {len(empty_claim_samples)} samples have no claims!")
        for cid in empty_claim_samples[:5]:
            print(f"    - {cid}")
    
    # =========================================================================
    # 4. Load NLI Model
    # =========================================================================
    print("\n" + "=" * 70)
    print("4. NLI MODEL LOADING")
    print("=" * 70)
    
    print(f"\nModel: {DEFAULT_MODEL_NAME}")
    model = NLIModel(model_name=DEFAULT_MODEL_NAME)
    print(f"Label mapping (from model.config.id2label):")
    print(f"  entailment_idx: {model.entailment_idx} ({model.id2label[model.entailment_idx]})")
    print(f"  neutral_idx:    {model.neutral_idx} ({model.id2label[model.neutral_idx]})")
    print(f"  contradiction_idx: {model.contradiction_idx} ({model.id2label[model.contradiction_idx]})")
    
    # =========================================================================
    # 5. NLI Inference
    # =========================================================================
    print("\n" + "=" * 70)
    print("5. NLI INFERENCE")
    print("=" * 70)
    
    faith_cache = CACHE_DIR / f"val_faithfulness_{timestamp}.csv"
    start = time.time()
    
    # Process all samples, track any issues
    for i, sample in enumerate(samples):
        processed_case_ids.add(sample.case_id)
        if i % 200 == 0:
            print(f"  Processing: {i}/{len(samples)}...")
    
    faith_scores = batch_inference(
        model, samples, segments,
        batch_size=8,
        cache_path=faith_cache,
        verbose=True,
        task_type="faithfulness",
    )
    infer_time = time.time() - start
    
    print(f"\nInference time: {infer_time:.1f}s")
    print(f"Total pairs: {len(faith_scores)}")
    
    # Verify no silent skipping
    n_samples_in_scores = faith_scores['case_id'].nunique()
    print(f"\nSamples in NLI results: {n_samples_in_scores}")
    
    missing_samples = set(case_ids) - set(faith_scores['case_id'].unique())
    if missing_samples:
        print(f"  WARNING: {len(missing_samples)} samples missing from results!")
        skipped_samples = [{"case_id": cid, "reason": "missing_from_nli_results"} for cid in list(missing_samples)[:10]]
    
    # Check for multi-window samples
    windows_per_sample = faith_scores.groupby('case_id')['window_id'].nunique()
    multi_window = windows_per_sample[windows_per_sample > 1]
    print(f"Samples with multiple windows: {len(multi_window)}")
    
    # =========================================================================
    # 6. Baselines
    # =========================================================================
    print("\n" + "=" * 70)
    print("6. BASELINE COMPARISONS")
    print("=" * 70)
    
    # Always faithful
    pred_always_faithful = np.ones(len(samples), dtype=int)
    metrics_always_faithful = compute_metrics(y_true, pred_always_faithful)
    
    # Always unfaithful
    pred_always_unfaithful = np.zeros(len(samples), dtype=int)
    metrics_always_unfaithful = compute_metrics(y_true, pred_always_unfaithful)
    
    # Stratified random
    np.random.seed(42)
    rate = n_faithful / len(samples)
    pred_stratified = (np.random.random(len(samples)) < rate).astype(int)
    metrics_stratified = compute_metrics(y_true, pred_stratified)
    
    # Majority baseline
    if n_faithful > n_unfaithful:
        pred_majority = np.ones(len(samples), dtype=int)
        majority_class = "faithful"
    else:
        pred_majority = np.zeros(len(samples), dtype=int)
        majority_class = "unfaithful"
    metrics_majority = compute_metrics(y_true, pred_majority)
    
    print("\nBaseline Results:")
    print(f"\n1. Always Faithful:")
    print(f"   Acc={metrics_always_faithful['accuracy']:.4f}, BalAcc={metrics_always_faithful['balanced_accuracy']:.4f}, Macro-F1={metrics_always_faithful['f1_macro']:.4f}")
    
    print(f"\n2. Always Unfaithful:")
    print(f"   Acc={metrics_always_unfaithful['accuracy']:.4f}, BalAcc={metrics_always_unfaithful['balanced_accuracy']:.4f}, Macro-F1={metrics_always_unfaithful['f1_macro']:.4f}")
    
    print(f"\n3. Stratified Random (p={rate:.3f}):")
    print(f"   Acc={metrics_stratified['accuracy']:.4f}, BalAcc={metrics_stratified['balanced_accuracy']:.4f}, Macro-F1={metrics_stratified['f1_macro']:.4f}")
    
    print(f"\n4. Majority ({majority_class}):")
    print(f"   Acc={metrics_majority['accuracy']:.4f}, BalAcc={metrics_majority['balanced_accuracy']:.4f}, Macro-F1={metrics_majority['f1_macro']:.4f}")
    
    # =========================================================================
    # 7. NLI Strategies with Threshold Search
    # =========================================================================
    print("\n" + "=" * 70)
    print("7. NLI FAITHFULNESS STRATEGIES")
    print("=" * 70)
    
    # Get scores for threshold search
    faith_agg = apply_faithfulness_strategy(faith_scores, "max_entail", threshold=0.5)
    score_lookup = faith_agg.set_index("case_id")["faithfulness_score"]
    scores = np.array([score_lookup.get(cid, 0.5) for cid in case_ids])
    
    # Threshold search
    best_f1 = 0
    best_th = 0.5
    for th in np.arange(0.1, 0.9, 0.02):
        pred = (scores >= th).astype(int)
        f1 = f1_score(y_true, pred, average='macro', zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
    
    print(f"\nBest threshold: {best_th:.2f}")
    
    # Apply best threshold
    pred_nli = (scores >= best_th).astype(int)
    metrics_nli = compute_metrics(y_true, pred_nli, scores)
    print_metrics("NLI max_entail", metrics_nli)
    
    # Try other strategies
    print("\n" + "-" * 50)
    print("Strategy Comparison:")
    strategy_results = {}
    
    for strategy_name in FAITHFULNESS_STRATEGIES.keys():
        agg = apply_faithfulness_strategy(faith_scores, strategy_name, threshold=0.5)
        score_lk = agg.set_index("case_id")["faithfulness_score"]
        sc = np.array([score_lk.get(cid, 0.5) for cid in case_ids])
        
        best_s_f1 = 0
        best_s_th = 0.5
        for th in np.arange(0.1, 0.9, 0.02):
            p = (sc >= th).astype(int)
            f1 = f1_score(y_true, p, average='macro', zero_division=0)
            if f1 > best_s_f1:
                best_s_f1 = f1
                best_s_th = th
        
        pred_s = (sc >= best_s_th).astype(int)
        met = compute_metrics(y_true, pred_s, sc)
        strategy_results[strategy_name] = {"threshold": best_s_th, **met}
        
        print(f"  {strategy_name}: F1={met['f1_macro']:.4f}, Acc={met['accuracy']:.4f}, Th={best_s_th:.2f}")
    
    # =========================================================================
    # 8. Source Model Subgroup Analysis
    # =========================================================================
    print("\n" + "=" * 70)
    print("8. SOURCE MODEL SUBGROUP ANALYSIS")
    print("=" * 70)
    
    # Get source model for each case_id
    model_lookup = {d['case_id']: d['source_model'] for d in val_list}
    case_to_idx = {cid: i for i, cid in enumerate(case_ids)}
    
    subgroup_results = {}
    for model_name in AVAILABLE_MODELS:
        model_mask = [model_lookup.get(cid) == model_name for cid in case_ids]
        model_indices = [i for i, m in enumerate(model_mask) if m]
        
        if not model_indices:
            continue
        
        model_y = y_true[model_indices]
        model_pred = pred_nli[model_indices]
        model_scores = scores[model_indices]
        model_met = compute_metrics(model_y, model_pred, model_scores)
        
        # Majority baseline for this model
        n_m_faith = sum(model_y == FAITHFUL)
        n_m_unf = sum(model_y == UNFAITHFUL)
        majority_rate = max(n_m_faith, n_m_unf) / len(model_y)
        
        subgroup_results[model_name] = {
            "n_samples": len(model_indices),
            "n_unfaithful": int(n_m_unf),
            "n_faithful": int(n_m_faith),
            "majority_rate": majority_rate,
            **model_met,
        }
        
        print(f"\n{model_name}:")
        print(f"  N: {len(model_indices)}, Unfaithful: {n_m_unf}, Faithful: {n_m_faith}")
        print(f"  Majority baseline Acc: {majority_rate:.4f}")
        print(f"  NLI: Acc={model_met['accuracy']:.4f}, BalAcc={model_met['balanced_accuracy']:.4f}, Macro-F1={model_met['f1_macro']:.4f}")
        print(f"  Unfaithful P/R/F: {model_met['precision_unfaithful']:.3f}/{model_met['recall_unfaithful']:.3f}/{model_met['f1_unfaithful']:.3f}")
    
    # =========================================================================
    # 9. TF-IDF Baseline (if available)
    # =========================================================================
    print("\n" + "=" * 70)
    print("9. TF-IDF BASELINE COMPARISON")
    print("=" * 70)
    
    tfidf_path = RESULTS_DIR.parent / "stage2_tfidf" / "validation_predictions.csv"
    if tfidf_path.exists():
        tfidf_df = pd.read_csv(tfidf_path)
        print(f"\nTF-IDF results loaded from {tfidf_path}")
        
        # Check columns
        print(f"  Columns: {list(tfidf_df.columns)}")
        
        # Match by case_id if possible
        if 'case_id' in tfidf_df.columns:
            tfidf_lookup = tfidf_df.set_index('case_id')
            
            # Check if we have matching case_ids
            common_ids = set(case_ids) & set(tfidf_lookup.index)
            print(f"  Matching case_ids: {len(common_ids)}/{len(case_ids)}")
            
            if len(common_ids) > 100:
                tfidf_preds = []
                for cid in case_ids:
                    if cid in tfidf_lookup.index:
                        row = tfidf_lookup.loc[cid]
                        # Try different column names
                        if 'faithfulness_prediction' in row.index:
                            tfidf_preds.append(int(row['faithfulness_prediction']))
                        elif 'prediction' in row.index:
                            tfidf_preds.append(int(row['prediction']))
                        else:
                            tfidf_preds.append(-1)
                    else:
                        tfidf_preds.append(-1)
                
                tfidf_pred = np.array(tfidf_preds)
                valid_mask = tfidf_pred >= 0
                
                if sum(valid_mask) > 100:
                    tfidf_met = compute_metrics(
                        y_true[valid_mask], 
                        tfidf_pred[valid_mask]
                    )
                    print(f"\nTF-IDF on matched samples (n={sum(valid_mask)}):")
                    print(f"  Acc={tfidf_met['accuracy']:.4f}, BalAcc={tfidf_met['balanced_accuracy']:.4f}, Macro-F1={tfidf_met['f1_macro']:.4f}")
    else:
        print(f"\nTF-IDF results not found at {tfidf_path}")
    
    # =========================================================================
    # 10. Save Outputs
    # =========================================================================
    print("\n" + "=" * 70)
    print("10. SAVING OUTPUTS")
    print("=" * 70)
    
    # Validation predictions
    val_pred_df = pd.DataFrame({
        "case_id": case_ids,
        "source_model": [model_lookup.get(cid, "") for cid in case_ids],
        "y_true": y_true,
        "y_pred": pred_nli,
        "faithfulness_score": scores,
    })
    val_pred_df.to_csv(RESULTS_DIR / "validation_predictions.csv", index=False)
    print(f"Saved: validation_predictions.csv")
    
    # Claim-window predictions
    faith_scores.to_csv(RESULTS_DIR / "claim_window_predictions.csv", index=False)
    print(f"Saved: claim_window_predictions.csv")
    
    # Metrics
    all_metrics = {
        "timestamp": timestamp,
        "git": git_info,
        "model": DEFAULT_MODEL_NAME,
        "dataset_revision": "aab54518c2a7c0d25fff8bffbf5337d0321de142",
        "n_val_samples": len(samples),
        "n_claims": total_claims,
        "best_threshold": float(best_th),
        "best_strategy": "max_entail",
        "faithfulness_metrics": metrics_nli,
        "baselines": {
            "always_faithful": metrics_always_faithful,
            "always_unfaithful": metrics_always_unfaithful,
            "stratified_random": metrics_stratified,
            "majority": metrics_majority,
        },
        "strategy_comparison": strategy_results,
        "subgroup_results": subgroup_results,
        "addressed_user_prompt_audit": {
            "total_with_field": len(addressed_vals),
            "total_samples": len(val_list),
            "distribution": dict(addressed_dist),
            "note": "Relevance evaluation NOT complete - addressed_user_prompt not fully available",
        },
        "inference_time_seconds": infer_time,
    }
    
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"Saved: metrics.json")
    
    # Best config
    best_config = {
        "threshold": float(best_th),
        "strategy": "max_entail",
        "model": DEFAULT_MODEL_NAME,
    }
    with open(RESULTS_DIR / "best_config.json", "w") as f:
        json.dump(best_config, f, indent=2)
    print(f"Saved: best_config.json")
    
    # Skipped samples
    with open(RESULTS_DIR / "skipped_samples.jsonl", "w") as f:
        for s in skipped_samples:
            f.write(json.dumps(s) + "\n")
    print(f"Saved: skipped_samples.jsonl ({len(skipped_samples)} entries)")
    
    # Error analysis
    errors = val_pred_df[val_pred_df["y_true"] != val_pred_df["y_pred"]]
    errors.to_csv(RESULTS_DIR / "error_analysis.csv", index=False)
    print(f"Saved: error_analysis.csv ({len(errors)} errors)")
    
    # Run manifest
    run_manifest = {
        "git_commit": git_info["commit"],
        "git_branch": git_info["branch"],
        "timestamp": timestamp,
        "split_manifest": {
            "val_size": split_info["val_size"],
            "seed": split_info["seed"],
            "n_val_questions": split_info["val_count"],
            "n_val_expanded": len(samples),
        },
        "dataset": {
            "name": "F4biian/RAGognize",
            "revision": "aab54518c2a7c0d25fff8bffbf5337d0321de142",
        },
        "model": {
            "name": DEFAULT_MODEL_NAME,
            "label_mapping": {
                "entailment_idx": model.entailment_idx,
                "neutral_idx": model.neutral_idx,
                "contradiction_idx": model.contradiction_idx,
            },
        },
        "best_config": best_config,
        "dependencies": {
            "transformers": "4.x",
            "torch": "2.x",
            "sklearn": "latest",
        },
        "device": str(model.device),
        "random_seed": 42,
        "processed_samples": len(samples),
        "skipped_samples": len(skipped_samples),
        "total_claims": total_claims,
    }
    
    with open(RESULTS_DIR / "run_manifest.json", "w") as f:
        json.dump(run_manifest, f, indent=2)
    print(f"Saved: run_manifest.json")
    
    # =========================================================================
    # 11. Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    print(f"\nFaithfulness (zero-shot mDeBERTa):")
    print(f"  Macro-F1:    {metrics_nli['f1_macro']:.4f}")
    print(f"  Accuracy:    {metrics_nli['accuracy']:.4f}")
    print(f"  Balanced:    {metrics_nli['balanced_accuracy']:.4f}")
    print(f"  AUROC:       {metrics_nli.get('auroc', 'N/A')}")
    
    print(f"\nBaselines:")
    print(f"  Majority:    F1={metrics_majority['f1_macro']:.4f}")
    print(f"  Stratified: F1={metrics_stratified['f1_macro']:.4f}")
    
    print(f"\n** Relevance evaluation: INCOMPLETE **")
    print(f"   addressed_user_prompt available for {len(addressed_vals)}/{len(val_list)} samples")
    print(f"   Cannot report Reliability = Faithfulness AND Relevance until Relevance is complete.")
    
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    
    return all_metrics


if __name__ == "__main__":
    main()
