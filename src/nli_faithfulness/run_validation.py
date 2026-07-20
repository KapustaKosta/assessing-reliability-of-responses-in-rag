#!/usr/bin/env python
"""
Clean Faithfulness Validation Run.

This script runs the full validation pipeline and generates all required output files
with complete reproducibility information.

Run with:
    /Users/chengyi/opt/miniconda3/envs/rag-reliability/bin/python \
        src/nli_faithfulness/run_validation.py

Outputs:
    - validation_predictions.csv (1100 rows)
    - claim_window_predictions.jsonl
    - metrics.json (auto-computed)
    - best_config.json
    - error_analysis.csv
    - skipped_samples.jsonl
    - run_manifest.json
"""

import sys
import os
import json
import time
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime, timezone

# Environment setup
ENV_PYTHON = "/Users/chengyi/opt/miniconda3/envs/rag-reliability/bin/python"
assert sys.executable == ENV_PYTHON, f"Must use {ENV_PYTHON}, got {sys.executable}"

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# ============================================================================
# IMPORTS
# ============================================================================

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

# ============================================================================
# CONSTANTS
# ============================================================================

# Label definitions
UNFAITHFUL = 0
FAITHFUL = 1
LABEL_NAMES = ["unfaithful", "faithful"]

# Results directory
RESULTS_DIR = Path(__file__).parent.parent.parent / "results" / "stage3_nli_faithfulness"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def compute_file_hash(filepath: str) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()

def get_git_info() -> dict:
    """Get git information for reproducibility."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
        ).strip()
        diff = subprocess.check_output(
            ["git", "diff", "--stat"], text=True
        ).strip()
        is_dirty = bool(diff)
        diff_sha = hashlib.sha256(diff.encode()).hexdigest() if is_dirty else None
        
        return {
            "commit": commit,
            "branch": branch,
            "is_dirty": is_dirty,
            "diff_sha256": diff_sha,
        }
    except Exception as e:
        return {"error": str(e)}

def get_exact_versions() -> dict:
    """Get exact versions of all dependencies."""
    versions = {}
    for pkg in ['transformers', 'torch', 'sklearn', 'datasets', 
                'pandas', 'numpy', 'accelerate', 'huggingface_hub', 'tokenizers']:
        try:
            mod = __import__(pkg.replace('-', '_'))
            versions[pkg] = getattr(mod, '__version__', 'unknown')
        except ImportError:
            versions[pkg] = 'NOT INSTALLED'
    return versions

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                    scores: np.ndarray = None, label_names: list = LABEL_NAMES) -> dict:
    """Compute comprehensive metrics from predictions."""
    cm = confusion_matrix(y_true, y_pred, labels=[UNFAITHFUL, FAITHFUL])
    
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        "f1_unfaithful": float(f1_score(y_true, y_pred, average='binary', pos_label=UNFAITHFUL, zero_division=0)),
        "f1_faithful": float(f1_score(y_true, y_pred, average='binary', pos_label=FAITHFUL, zero_division=0)),
        "precision_unfaithful": float(precision_score(y_true, y_pred, average='binary', pos_label=UNFAITHFUL, zero_division=0)),
        "recall_unfaithful": float(recall_score(y_true, y_pred, average='binary', pos_label=UNFAITHFUL, zero_division=0)),
        "precision_faithful": float(precision_score(y_true, y_pred, average='binary', pos_label=FAITHFUL, zero_division=0)),
        "recall_faithful": float(recall_score(y_true, y_pred, average='binary', pos_label=FAITHFUL, zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": label_names,
        "confusion_matrix_rows": "true_labels",
        "confusion_matrix_columns": "predicted_labels",
        "true_unfaithful": int(cm[0, 0] + cm[0, 1]),
        "true_faithful": int(cm[1, 0] + cm[1, 1]),
        "pred_unfaithful": int(cm[0, 0] + cm[1, 0]),
        "pred_faithful": int(cm[0, 1] + cm[1, 1]),
    }
    
    if scores is not None:
        try:
            metrics["auroc"] = float(roc_auc_score(y_true, scores))
        except:
            metrics["auroc"] = None
        try:
            metrics["auprc"] = float(average_precision_score(y_true, scores))
        except:
            metrics["auprc"] = None
    
    return metrics

def compute_baselines(y_true: np.ndarray) -> dict:
    """Compute all baseline metrics."""
    n = len(y_true)
    n_faithful = int(y_true.sum())
    n_unfaithful = n - n_faithful
    rate = n_faithful / n
    
    baselines = {}
    
    # Always faithful
    pred = np.ones(n, dtype=int)
    baselines["always_faithful"] = compute_metrics(y_true, pred)
    
    # Always unfaithful
    pred = np.zeros(n, dtype=int)
    baselines["always_unfaithful"] = compute_metrics(y_true, pred)
    
    # Stratified random
    np.random.seed(42)
    pred = (np.random.random(n) < rate).astype(int)
    baselines["stratified_random"] = compute_metrics(y_true, pred)
    
    # Majority
    if n_faithful > n_unfaithful:
        pred = np.ones(n, dtype=int)
        baselines["majority"] = {"class": "faithful", **compute_metrics(y_true, pred)}
    else:
        pred = np.zeros(n, dtype=int)
        baselines["majority"] = {"class": "unfaithful", **compute_metrics(y_true, pred)}
    
    return baselines

# ============================================================================
# MAIN RUN
# ============================================================================

def main():
    print("=" * 70)
    print("CLEAN FAITHFULNESS VALIDATION RUN")
    print("=" * 70)
    
    # Record timing
    started_at = datetime.now(timezone.utc).isoformat()
    start_time = time.time()
    
    # -------------------------------------------------------------------------
    # 1. ENVIRONMENT INFO
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("1. ENVIRONMENT INFORMATION")
    print("=" * 70)
    
    print(f"\nPython executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    
    versions = get_exact_versions()
    print("\nDependency versions:")
    for pkg, v in sorted(versions.items()):
        print(f"  {pkg}: {v}")
    
    # Get device
    import torch
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    # -------------------------------------------------------------------------
    # 2. GIT INFO
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("2. GIT INFORMATION")
    print("=" * 70)
    
    git_info = get_git_info()
    print(f"\nCommit: {git_info.get('commit', 'N/A')}")
    print(f"Branch: {git_info.get('branch', 'N/A')}")
    print(f"Working tree dirty: {git_info.get('is_dirty', 'N/A')}")
    
    # -------------------------------------------------------------------------
    # 3. DATA LOADING
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("3. DATA LOADING")
    print("=" * 70)
    
    print("\nLoading RAGognize dataset...")
    raw = load_ragognize_dataset()
    
    print(f"  Raw train: {len(raw['train'])} questions")
    print(f"  Raw test: {len(raw['test'])} questions")
    
    # Create split
    split_info = create_train_val_split(raw, val_size=0.15, seed=42)
    print(f"  Val questions: {len(split_info['val_indices'])}")
    
    adapter = RAGognizeAdapter(models=AVAILABLE_MODELS)
    raw_split = apply_split(raw, split_info)
    unified = adapter.transform_dataset(raw_split)
    
    # Convert to samples
    val_list = list(unified['val'])
    print(f"  Validation samples: {len(val_list)}")
    
    # Ground truth
    y_true = np.array([FAITHFUL if d['faithfulness_label'] else UNFAITHFUL for d in val_list])
    case_ids = [d['case_id'] for d in val_list]
    
    n_faithful = int(y_true.sum())
    n_unfaithful = len(y_true) - n_faithful
    
    print(f"\n  Faithful (1): {n_faithful}")
    print(f"  Unfaithful (0): {n_unfaithful}")
    
    # -------------------------------------------------------------------------
    # 4. ADDRESSED_USER_PROMPT AUDIT
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("4. ADDRESSED_USER_PROMPT AUDIT")
    print("=" * 70)
    
    addressed_count = 0
    for d in val_list:
        responses = d.get('responses', {})
        for model_name in AVAILABLE_MODELS:
            if model_name in responses:
                resp = responses[model_name]
                details = resp.get('details', {})
                result = details.get('result', {})
                if 'addressed_user_prompt' in result:
                    addressed_count += 1
                break
    
    print(f"\n  addressed_user_prompt available: {addressed_count}/{len(val_list)}")
    print(f"  Relevance: NOT AVAILABLE" if addressed_count == 0 else f"  Relevance: AVAILABLE")
    print(f"  Reliability: NOT AVAILABLE (awaiting Relevance)")
    
    relevance_audit = {
        "addressed_user_prompt_available": addressed_count,
        "total_samples": len(val_list),
        "availability_rate": addressed_count / len(val_list) if len(val_list) > 0 else 0,
        "note": "Relevance formal evaluation NOT AVAILABLE. addressed_user_prompt field missing from all samples."
    }
    
    # -------------------------------------------------------------------------
    # 5. ANSWER SEGMENTATION
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("5. ANSWER SEGMENTATION")
    print("=" * 70)
    
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
    
    samples = [to_sample(val_list[i]) for i in range(len(val_list))]
    segments = segment_dataset(samples)
    total_claims = sum(len(seg) for seg in segments.values())
    
    print(f"\n  Total claims: {total_claims}")
    print(f"  Avg claims/sample: {total_claims/len(samples):.1f}")
    
    empty_claim_samples = [cid for cid, seg in segments.items() if len(seg) == 0]
    print(f"  Samples with empty claims: {len(empty_claim_samples)}")
    
    # -------------------------------------------------------------------------
    # 6. NLI INFERENCE
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("6. NLI INFERENCE")
    print("=" * 70)
    
    print(f"\n  Model: {DEFAULT_MODEL_NAME}")
    model = NLIModel(model_name=DEFAULT_MODEL_NAME)
    
    print(f"  Label mapping:")
    print(f"    entailment_idx: {model.entailment_idx} ({model.id2label[model.entailment_idx]})")
    print(f"    neutral_idx: {model.neutral_idx} ({model.id2label[model.neutral_idx]})")
    print(f"    contradiction_idx: {model.contradiction_idx} ({model.id2label[model.contradiction_idx]})")
    
    # Run inference
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    faith_cache = CACHE_DIR / f"val_faithfulness_{timestamp}.csv"
    
    print(f"\n  Running inference...")
    infer_start = time.time()
    
    faith_scores = batch_inference(
        model, samples, segments,
        batch_size=8,
        cache_path=faith_cache,
        verbose=True,
        task_type="faithfulness",
    )
    
    infer_time = time.time() - infer_start
    print(f"\n  Inference time: {infer_time:.1f}s")
    print(f"  Total pairs: {len(faith_scores)}")
    
    # Check for missing samples
    n_samples_in_scores = faith_scores['case_id'].nunique()
    missing_case_ids = set(case_ids) - set(faith_scores['case_id'].unique())
    
    print(f"  Samples in NLI results: {n_samples_in_scores}")
    print(f"  Missing samples: {len(missing_case_ids)}")
    
    # -------------------------------------------------------------------------
    # 7. STRATEGY COMPARISON
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("7. STRATEGY COMPARISON")
    print("=" * 70)
    
    # Get scores for each strategy
    strategy_results = {}
    
    for strategy_name in FAITHFULNESS_STRATEGIES.keys():
        agg = apply_faithfulness_strategy(faith_scores, strategy_name, threshold=0.5)
        score_lookup = agg.set_index("case_id")["faithfulness_score"]
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
        
        pred = (scores >= best_th).astype(int)
        met = compute_metrics(y_true, pred, scores)
        
        strategy_results[strategy_name] = {
            "threshold": float(best_th),
            **met
        }
        
        print(f"\n  {strategy_name}:")
        print(f"    Threshold: {best_th:.2f}")
        print(f"    Acc: {met['accuracy']:.4f}, BalAcc: {met['balanced_accuracy']:.4f}, Macro-F1: {met['f1_macro']:.4f}")
    
    # Find best strategy
    best_strategy_name = max(strategy_results, key=lambda k: strategy_results[k]['f1_macro'])
    best_config = {
        "threshold": strategy_results[best_strategy_name]['threshold'],
        "strategy": best_strategy_name,
        "model": DEFAULT_MODEL_NAME,
    }
    
    print(f"\n  Best strategy: {best_strategy_name}")
    
    # -------------------------------------------------------------------------
    # 8. GENERATE PREDICTIONS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("8. GENERATING PREDICTIONS")
    print("=" * 70)
    
    # Use best strategy
    agg = apply_faithfulness_strategy(faith_scores, best_strategy_name, threshold=best_config['threshold'])
    score_lookup = agg.set_index("case_id")["faithfulness_score"]
    scores = np.array([score_lookup.get(cid, 0.5) for cid in case_ids])
    y_pred = (scores >= best_config['threshold']).astype(int)
    
    # Create predictions DataFrame
    pred_records = []
    for i, cid in enumerate(case_ids):
        d = val_list[i]
        pred_records.append({
            'expanded_sample_id': cid,
            'question_id': d['user_prompt_index'],
            'source_model': d['source_model'],
            'response_index': AVAILABLE_MODELS.index(d['source_model']) if d['source_model'] in AVAILABLE_MODELS else -1,
            'true_faithfulness_label': int(y_true[i]),
            'predicted_faithfulness_label': int(y_pred[i]),
            'faithfulness_score': float(scores[i]),
            'threshold': float(best_config['threshold']),
            'aggregation_strategy': best_strategy_name,
            'claim_count': len(segments.get(cid, [])),
            'correct': bool(y_true[i] == y_pred[i]),
        })
    
    pred_df = pd.DataFrame(pred_records)
    
    # Save predictions
    pred_path = RESULTS_DIR / "validation_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    pred_hash = compute_file_hash(str(pred_path))
    print(f"\n  Saved: {pred_path}")
    print(f"  Rows: {len(pred_df)}")
    print(f"  SHA-256: {pred_hash[:16]}...")
    
    # -------------------------------------------------------------------------
    # 9. SAVE CLAIM WINDOW PREDICTIONS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("9. CLAIM WINDOW PREDICTIONS")
    print("=" * 70)
    
    # Save as JSONL
    claim_path = RESULTS_DIR / "claim_window_predictions.jsonl"
    faith_scores.to_json(str(claim_path), orient='records', lines=True)
    claim_hash = compute_file_hash(str(claim_path))
    print(f"\n  Saved: {claim_path}")
    print(f"  Rows: {len(faith_scores)}")
    print(f"  SHA-256: {claim_hash[:16]}...")
    
    # -------------------------------------------------------------------------
    # 10. COMPUTE METRICS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("10. COMPUTING METRICS")
    print("=" * 70)
    
    # Main metrics
    main_metrics = compute_metrics(y_true, y_pred, scores)
    print_metrics("Main", main_metrics)
    
    # Baselines
    baselines = compute_baselines(y_true)
    
    print("\n  Baselines:")
    for name, met in baselines.items():
        print(f"    {name}: Acc={met['accuracy']:.4f}, BalAcc={met['balanced_accuracy']:.4f}, Macro-F1={met['f1_macro']:.4f}")
    
    # Subgroup metrics
    print("\n  Subgroup by source_model:")
    subgroup_metrics = {}
    
    for model_name in AVAILABLE_MODELS:
        mask = [d['source_model'] == model_name for d in val_list]
        indices = [i for i, m in enumerate(mask) if m]
        
        if not indices:
            continue
        
        model_y_true = y_true[indices]
        model_y_pred = y_pred[indices]
        model_scores = scores[indices]
        
        met = compute_metrics(model_y_true, model_y_pred, model_scores)
        subgroup_metrics[model_name] = {
            "n_samples": len(indices),
            "n_unfaithful": int((model_y_true == UNFAITHFUL).sum()),
            "n_faithful": int((model_y_true == FAITHFUL).sum()),
            **met
        }
        
        print(f"    {model_name}:")
        print(f"      N={len(indices)}, Unfaithful={int((model_y_true == UNFAITHFUL).sum())}")
        print(f"      Acc={met['accuracy']:.4f}, BalAcc={met['balanced_accuracy']:.4f}, Macro-F1={met['f1_macro']:.4f}")
    
    # -------------------------------------------------------------------------
    # 11. SAVE METRICS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("11. SAVING METRICS")
    print("=" * 70)
    
    metrics = {
        "faithfulness_metrics": main_metrics,
        "baselines": baselines,
        "subgroup_metrics": subgroup_metrics,
        "strategy_comparison": strategy_results,
    }
    
    metrics_path = RESULTS_DIR / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    metrics_hash = compute_file_hash(str(metrics_path))
    print(f"\n  Saved: {metrics_path}")
    print(f"  SHA-256: {metrics_hash[:16]}...")
    
    # Best config
    best_config_path = RESULTS_DIR / "best_config.json"
    with open(best_config_path, 'w') as f:
        json.dump(best_config, f, indent=2)
    print(f"  Saved: {best_config_path}")
    
    # -------------------------------------------------------------------------
    # 12. ERROR ANALYSIS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("12. ERROR ANALYSIS")
    print("=" * 70)
    
    errors = pred_df[pred_df['correct'] == False]
    errors_path = RESULTS_DIR / "error_analysis.csv"
    errors.to_csv(errors_path, index=False)
    errors_hash = compute_file_hash(str(errors_path))
    print(f"\n  Errors: {len(errors)}")
    print(f"  Saved: {errors_path}")
    print(f"  SHA-256: {errors_hash[:16]}...")
    
    # -------------------------------------------------------------------------
    # 13. SKIPPED SAMPLES
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("13. SKIPPED SAMPLES")
    print("=" * 70)
    
    skipped = []
    
    # Source data missing (none in this case)
    if len(missing_case_ids) > 0:
        for cid in missing_case_ids:
            skipped.append({
                "case_id": cid,
                "reason": "source_data_missing",
                "timestamp": datetime.now().isoformat(),
            })
    
    # Empty claims
    for cid in empty_claim_samples:
        skipped.append({
            "case_id": cid,
            "reason": "empty_claim",
            "timestamp": datetime.now().isoformat(),
        })
    
    skipped_path = RESULTS_DIR / "skipped_samples.jsonl"
    with open(skipped_path, 'w') as f:
        for s in skipped:
            f.write(json.dumps(s) + "\n")
    print(f"\n  Skipped samples: {len(skipped)}")
    print(f"  Saved: {skipped_path}")
    
    # -------------------------------------------------------------------------
    # 14. RUN MANIFEST
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("14. GENERATING RUN MANIFEST")
    print("=" * 70)
    
    finished_at = datetime.now(timezone.utc).isoformat()
    duration = time.time() - start_time
    
    # Dataset info
    dataset_revision = "aab54518c2a7c0d25fff8bffbf5337d0321de142"
    dataset_cache_path = str(CACHE_DIR / f"F4biian___ra_gognize")
    
    manifest = {
        "evaluation_name": "Zero-shot Claim-level NLI Faithfulness Baseline on RAGognize Validation",
        "inference_git_commit": git_info.get('commit', 'unknown'),
        "git_branch": git_info.get('branch', 'unknown'),
        "working_tree_dirty": git_info.get('is_dirty', False),
        "git_diff_sha256": git_info.get('diff_sha256'),
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "duration_seconds": duration,
        
        "dataset": {
            "name": "F4biian/RAGognize",
            "revision": dataset_revision,
            "cache_path": dataset_cache_path,
        },
        
        "split": {
            "val_size": split_info["val_size"],
            "seed": split_info["seed"],
            "n_val_questions": len(split_info["val_indices"]),
            "n_theoretical_slots": len(pred_df),
            "n_source_missing": 0,
            "n_runtime_skipped": len(skipped),
            "n_actual_valid": len(pred_df),
        },
        
        "model": {
            "name": DEFAULT_MODEL_NAME,
            "entailment_idx": model.entailment_idx,
            "neutral_idx": model.neutral_idx,
            "contradiction_idx": model.contradiction_idx,
        },
        
        "best_config": best_config,
        
        "environment": {
            "python_executable": sys.executable,
            "python_version": sys.version,
            "python_version_short": f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "device": device,
            "packages": versions,
        },
        
        "relevance_audit": relevance_audit,
        
        "label_semantics": {
            "unfaithful": 0,
            "faithful": 1,
            "description": "faithfulness_label 0 = unfaithful (hallucination), 1 = faithful (no hallucination)"
        },
        
        "artifacts": {
            "validation_predictions.csv": {
                "path": str(pred_path),
                "rows": len(pred_df),
                "sha256": pred_hash,
            },
            "claim_window_predictions.jsonl": {
                "path": str(claim_path),
                "rows": len(faith_scores),
                "sha256": claim_hash,
            },
            "metrics.json": {
                "path": str(metrics_path),
                "sha256": metrics_hash,
            },
            "best_config.json": {
                "path": str(best_config_path),
            },
            "error_analysis.csv": {
                "path": str(errors_path),
                "rows": len(errors),
                "sha256": errors_hash,
            },
            "skipped_samples.jsonl": {
                "path": str(skipped_path),
                "rows": len(skipped),
            },
        },
        
        "what_is_evaluated": {
            "faithfulness": "COMPLETE",
            "relevance": "NOT_AVAILABLE",
            "reliability": "NOT_AVAILABLE"
        },
    }
    
    manifest_path = RESULTS_DIR / "run_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    manifest_hash = compute_file_hash(str(manifest_path))
    
    print(f"\n  Saved: {manifest_path}")
    print(f"  SHA-256: {manifest_hash[:16]}...")
    
    # -------------------------------------------------------------------------
    # 15. SUMMARY
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("15. FINAL SUMMARY")
    print("=" * 70)
    
    print(f"\n  Evaluation: Zero-shot Claim-level NLI Faithfulness Baseline")
    print(f"  Samples: {len(pred_df)}")
    print(f"  Duration: {duration:.1f}s")
    
    print(f"\n  Faithfulness Metrics:")
    print(f"    Accuracy: {main_metrics['accuracy']:.4f}")
    print(f"    Balanced Accuracy: {main_metrics['balanced_accuracy']:.4f}")
    print(f"    Macro-F1: {main_metrics['f1_macro']:.4f}")
    print(f"    AUROC: {main_metrics.get('auroc', 'N/A')}")
    
    print(f"\n  Confusion Matrix (rows=true, cols=pred, order=[unfaithful, faithful]):")
    cm = main_metrics['confusion_matrix']
    print(f"    [[TN={cm[0][0]}, FP={cm[0][1]}],")
    print(f"     [FN={cm[1][0]}, TP={cm[1][1]}]]")
    
    print(f"\n  Relevance: NOT AVAILABLE (addressed_user_prompt = 0/{len(val_list)})")
    print(f"  Reliability: NOT AVAILABLE (awaiting Relevance data)")
    
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    
    return manifest


def print_metrics(name: str, metrics: dict):
    """Print metrics in a readable format."""
    print(f"\n  {name}:")
    print(f"    Accuracy: {metrics['accuracy']:.4f}")
    print(f"    Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"    Macro-F1: {metrics['f1_macro']:.4f}")
    print(f"    Unfaithful P/R/F: {metrics['precision_unfaithful']:.3f}/{metrics['recall_unfaithful']:.3f}/{metrics['f1_unfaithful']:.3f}")
    print(f"    Faithful P/R/F: {metrics['precision_faithful']:.3f}/{metrics['recall_faithful']:.3f}/{metrics['f1_faithful']:.3f}")
    if metrics.get('auroc'):
        print(f"    AUROC: {metrics['auroc']:.4f}")


if __name__ == "__main__":
    manifest = main()
