#!/usr/bin/env python
"""
Clean Faithfulness Validation Run - Offline Mode.

Uses cached models to avoid network timeouts.
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

# Add project root and src to path
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

# Set offline mode to use cached models
os.environ['HF_HUB_OFFLINE'] = '1'

# ============================================================================
# IMPORTS
# ============================================================================

import numpy as np
import pandas as pd

from ragognize_adapter import (
    RAGognizeAdapter, load_ragognize_dataset,
    create_train_val_split, apply_split, AVAILABLE_MODELS,
)
from ragognize_adapter.parsing_helpers import parse_annotation_result
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

UNFAITHFUL = 0
FAITHFUL = 1
LABEL_NAMES = ["unfaithful", "faithful"]
RESULTS_DIR = Path(__file__).parent.parent.parent / "results" / "stage3_nli_faithfulness"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def compute_file_hash(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()

def get_git_info() -> dict:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
        diff = subprocess.check_output(["git", "diff", "--stat"], text=True).strip()
        is_dirty = bool(diff)
        diff_sha = hashlib.sha256(diff.encode()).hexdigest() if is_dirty else None
        return {"commit": commit, "branch": branch, "is_dirty": is_dirty, "diff_sha256": diff_sha}
    except Exception as e:
        return {"error": str(e)}

def get_exact_versions() -> dict:
    versions = {}
    for pkg in ['transformers', 'torch', 'sklearn', 'datasets', 'pandas', 'numpy', 'accelerate', 'huggingface_hub', 'tokenizers']:
        try:
            mod = __import__(pkg.replace('-', '_'))
            versions[pkg] = getattr(mod, '__version__', 'unknown')
        except ImportError:
            versions[pkg] = 'NOT INSTALLED'
    return versions

def compute_metrics(y_true, y_pred, scores=None, label_names=LABEL_NAMES):
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
        "true_unfaithful": int(cm[0, 0] + cm[0, 1]),
        "true_faithful": int(cm[1, 0] + cm[1, 1]),
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

def compute_baselines(y_true):
    n = len(y_true)
    n_faithful = int(y_true.sum())
    rate = n_faithful / n
    baselines = {}
    
    for name, pred in [("always_faithful", np.ones(n, dtype=int)), 
                        ("always_unfaithful", np.zeros(n, dtype=int))]:
        baselines[name] = compute_metrics(y_true, pred)
    
    np.random.seed(42)
    pred = (np.random.random(n) < rate).astype(int)
    baselines["stratified_random"] = compute_metrics(y_true, pred)
    
    if n_faithful > n - n_faithful:
        baselines["majority"] = {"class": "faithful", **compute_metrics(y_true, np.ones(n, dtype=int))}
    else:
        baselines["majority"] = {"class": "unfaithful", **compute_metrics(y_true, np.zeros(n, dtype=int))}
    
    return baselines

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("CLEAN FAITHFULNESS VALIDATION RUN (OFFLINE MODE)")
    print("=" * 70)
    
    started_at = datetime.now(timezone.utc).isoformat()
    start_time = time.time()
    
    # Environment
    print("\n1. ENVIRONMENT")
    print(f"   Python: {sys.executable}")
    print(f"   Version: {sys.version}")
    versions = get_exact_versions()
    for pkg in sorted(versions):
        print(f"   {pkg}: {versions[pkg]}")
    
    import torch
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"   Device: {device}")
    
    # Git
    print("\n2. GIT")
    git_info = get_git_info()
    print(f"   Commit: {git_info.get('commit', 'N/A')}")
    print(f"   Branch: {git_info.get('branch', 'N/A')}")
    print(f"   Dirty: {git_info.get('is_dirty', 'N/A')}")
    
    # Data
    print("\n3. DATA LOADING")
    raw = load_ragognize_dataset()
    split_info = create_train_val_split(raw, val_size=0.15, seed=42)
    adapter = RAGognizeAdapter(models=AVAILABLE_MODELS)
    raw_split = apply_split(raw, split_info)
    unified = adapter.transform_dataset(raw_split)
    val_list = list(unified['val'])
    print(f"   Val questions: {len(split_info['val_indices'])}")
    print(f"   Validation samples: {len(val_list)}")
    
    y_true = np.array([FAITHFUL if d['faithfulness_label'] else UNFAITHFUL for d in val_list])
    case_ids = [d['case_id'] for d in val_list]
    print(f"   Faithful: {int(y_true.sum())}, Unfaithful: {len(y_true) - int(y_true.sum())}")
    
    # Relevance audit (correct nested path)
    print("\n4. ADDRESSED_USER_PROMPT AUDIT")
    total_valid_responses = 0
    addressed_true = 0
    addressed_false = 0
    addressed_missing = 0
    addressed_invalid = 0
    source_missing_per_model = {m: 0 for m in AVAILABLE_MODELS}
    per_model_relevance = {m: {"total": 0, "true": 0, "false": 0, "missing": 0, "invalid": 0}
                           for m in AVAILABLE_MODELS}

    for item in raw_split["val"]:
        responses = item.get("responses", {})
        for model_name in AVAILABLE_MODELS:
            if model_name not in responses:
                source_missing_per_model[model_name] += 1
                continue

            total_valid_responses += 1
            ann = parse_annotation_result(responses[model_name])
            cat = ann.addressed_user_prompt
            per_model_relevance[model_name]["total"] += 1
            if cat == "true":
                addressed_true += 1
                per_model_relevance[model_name]["true"] += 1
            elif cat == "false":
                addressed_false += 1
                per_model_relevance[model_name]["false"] += 1
            elif cat == "missing":
                addressed_missing += 1
                per_model_relevance[model_name]["missing"] += 1
            else:
                addressed_invalid += 1
                per_model_relevance[model_name]["invalid"] += 1

    available = addressed_true + addressed_false
    missing_or_invalid = addressed_missing + addressed_invalid
    n_val_questions = len(split_info["val_indices"])
    n_theoretical_slots = n_val_questions * len(AVAILABLE_MODELS)
    n_source_missing = sum(source_missing_per_model.values())

    print(f"   Extraction path: details.annotations.result.addressed_user_prompt")
    print(f"   Val questions: {n_val_questions}")
    print(f"   Theoretical slots: {n_theoretical_slots}")
    print(f"   Source-missing: {n_source_missing} {dict(source_missing_per_model)}")
    print(f"   Actual valid responses: {total_valid_responses}")
    print(f"   Invariant: {n_theoretical_slots} = {n_source_missing} + {total_valid_responses} → {n_theoretical_slots == n_source_missing + total_valid_responses}")
    print(f"   addressed_true={addressed_true}, false={addressed_false}, missing={addressed_missing}, invalid={addressed_invalid}")
    print(f"   available (true+false)={available}")
    print(f"   Invariant: available({available}) + missing({missing_or_invalid}) == total({total_valid_responses}) → {available + missing_or_invalid == total_valid_responses}")

    for model_name in AVAILABLE_MODELS:
        mc = per_model_relevance[model_name]
        sm = source_missing_per_model[model_name]
        print(f"   {model_name}: theoretical={mc['total']+sm}, missing={sm}, valid={mc['total']}, "
              f"true={mc['true']}, false={mc['false']}, missing={mc['missing']}, invalid={mc['invalid']}")

    relevance_audit = {
        "extraction_path": "details.annotations.result.addressed_user_prompt",
        "total_valid_responses": total_valid_responses,
        "addressed_true": addressed_true,
        "addressed_false": addressed_false,
        "addressed_missing": addressed_missing,
        "addressed_invalid": addressed_invalid,
        "available": available,
        "missing_or_invalid": missing_or_invalid,
        "per_source_model": {
            m: {**per_model_relevance[m], "source_missing": source_missing_per_model[m]}
            for m in AVAILABLE_MODELS
        },
        "note": "addressed_user_prompt is a GOLD relevance label. Recovering it does NOT mean a relevance classifier has been implemented. Relevance prediction/model: NOT_AVAILABLE.",
    }
    
    # Segmentation
    print("\n5. SEGMENTATION")
    def to_sample(d):
        chunks = [ChunkData(
            case_id=d['case_id'], chunk_id=i+1, chunk_rank=i+1,
            chunk_text=t, retrieval_config='top_1', is_available=True,
        ) for i, t in enumerate(d['chunks'])]
        return SampleData(
            case_id=d['case_id'], question=d['question'], answer=d['answer'],
            binary_faithfulness=d['faithfulness_label'], binary_relevancy=d['answerable'],
            chunks=chunks,
        )
    
    samples = [to_sample(val_list[i]) for i in range(len(val_list))]
    segments = segment_dataset(samples)
    total_claims = sum(len(seg) for seg in segments.values())
    print(f"   Total claims: {total_claims}")
    print(f"   Avg claims/sample: {total_claims/len(samples):.1f}")
    empty_claim_samples = [cid for cid, seg in segments.items() if len(seg) == 0]
    print(f"   Empty claim samples: {len(empty_claim_samples)}")
    
    # NLI Inference
    print("\n6. NLI INFERENCE")
    print(f"   Model: {DEFAULT_MODEL_NAME}")
    model = NLIModel(model_name=DEFAULT_MODEL_NAME)
    print(f"   Entailment idx: {model.entailment_idx}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    faith_cache = CACHE_DIR / f"val_faithfulness_{timestamp}.csv"
    
    print(f"   Running inference...")
    infer_start = time.time()
    faith_scores = batch_inference(model, samples, segments, batch_size=8, 
                                   cache_path=faith_cache, verbose=True, task_type="faithfulness")
    infer_time = time.time() - infer_start
    print(f"   Inference time: {infer_time:.1f}s")
    print(f"   Total pairs: {len(faith_scores)}")
    
    missing_case_ids = set(case_ids) - set(faith_scores['case_id'].unique())
    print(f"   Missing samples: {len(missing_case_ids)}")
    
    # Strategy comparison
    print("\n7. STRATEGY COMPARISON")
    strategy_results = {}
    for strategy_name in FAITHFULNESS_STRATEGIES.keys():
        agg = apply_faithfulness_strategy(faith_scores, strategy_name, threshold=0.5)
        score_lookup = agg.set_index("case_id")["faithfulness_score"]
        scores = np.array([score_lookup.get(cid, 0.5) for cid in case_ids])
        
        best_f1, best_th = 0, 0.5
        for th in np.arange(0.1, 0.9, 0.02):
            pred = (scores >= th).astype(int)
            f1 = f1_score(y_true, pred, average='macro', zero_division=0)
            if f1 > best_f1:
                best_f1, best_th = f1, th
        
        pred = (scores >= best_th).astype(int)
        strategy_results[strategy_name] = {"threshold": float(best_th), **compute_metrics(y_true, pred, scores)}
        print(f"   {strategy_name}: F1={best_f1:.4f}, th={best_th:.2f}")
    
    best_strategy = max(strategy_results, key=lambda k: strategy_results[k]['f1_macro'])
    best_config = {"threshold": strategy_results[best_strategy]['threshold'], "strategy": best_strategy, "model": DEFAULT_MODEL_NAME}
    print(f"   Best: {best_strategy}")
    
    # Generate predictions
    print("\n8. PREDICTIONS")
    agg = apply_faithfulness_strategy(faith_scores, best_strategy, threshold=best_config['threshold'])
    score_lookup = agg.set_index("case_id")["faithfulness_score"]
    scores = np.array([score_lookup.get(cid, 0.5) for cid in case_ids])
    y_pred = (scores >= best_config['threshold']).astype(int)
    
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
            'aggregation_strategy': best_strategy,
            'claim_count': len(segments.get(cid, [])),
            'correct': bool(y_true[i] == y_pred[i]),
        })
    
    pred_df = pd.DataFrame(pred_records)
    pred_path = RESULTS_DIR / "validation_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    pred_hash = compute_file_hash(str(pred_path))
    print(f"   Saved: {pred_path}")
    print(f"   Rows: {len(pred_df)}, SHA-256: {pred_hash[:16]}...")
    
    # Claim window predictions
    print("\n9. CLAIM WINDOW PREDICTIONS")
    claim_path = RESULTS_DIR / "claim_window_predictions.jsonl"
    faith_scores.to_json(str(claim_path), orient='records', lines=True)
    claim_hash = compute_file_hash(str(claim_path))
    print(f"   Saved: {claim_path}")
    print(f"   Rows: {len(faith_scores)}, SHA-256: {claim_hash[:16]}...")
    
    # Metrics
    print("\n10. METRICS")
    main_metrics = compute_metrics(y_true, y_pred, scores)
    print(f"   Accuracy: {main_metrics['accuracy']:.4f}")
    print(f"   Balanced Accuracy: {main_metrics['balanced_accuracy']:.4f}")
    print(f"   Macro-F1: {main_metrics['f1_macro']:.4f}")
    print(f"   AUROC: {main_metrics.get('auroc', 'N/A')}")
    cm = main_metrics['confusion_matrix']
    print(f"   CM: [[TN={cm[0][0]}, FP={cm[0][1]}], [FN={cm[1][0]}, TP={cm[1][1]}]]")
    
    baselines = compute_baselines(y_true)
    print("\n   Baselines:")
    for name, met in baselines.items():
        print(f"     {name}: Acc={met['accuracy']:.4f}, Macro-F1={met['f1_macro']:.4f}")
    
    # Subgroup metrics
    subgroup_metrics = {}
    print("\n   Subgroups:")
    for model_name in AVAILABLE_MODELS:
        mask = [d['source_model'] == model_name for d in val_list]
        indices = [i for i, m in enumerate(mask) if m]
        if not indices:
            continue
        model_y = y_true[indices]
        model_pred = y_pred[indices]
        model_scores = scores[indices]
        met = compute_metrics(model_y, model_pred, model_scores)
        subgroup_metrics[model_name] = {"n_samples": len(indices), "n_unfaithful": int((model_y == UNFAITHFUL).sum()), **met}
        print(f"     {model_name}: N={len(indices)}, Unfaithful={int((model_y == UNFAITHFUL).sum())}, Acc={met['accuracy']:.4f}, F1={met['f1_macro']:.4f}")
    
    # Save metrics
    metrics = {"faithfulness_metrics": main_metrics, "baselines": baselines, "subgroup_metrics": subgroup_metrics, "strategy_comparison": strategy_results}
    metrics_path = RESULTS_DIR / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    metrics_hash = compute_file_hash(str(metrics_path))
    print(f"\n   Saved: {metrics_path}, SHA-256: {metrics_hash[:16]}...")
    
    best_config_path = RESULTS_DIR / "best_config.json"
    with open(best_config_path, 'w') as f:
        json.dump(best_config, f, indent=2)
    print(f"   Saved: {best_config_path}")
    
    # Error analysis
    print("\n11. ERROR ANALYSIS")
    errors = pred_df[pred_df['correct'] == False]
    errors_path = RESULTS_DIR / "error_analysis.csv"
    errors.to_csv(errors_path, index=False)
    errors_hash = compute_file_hash(str(errors_path))
    print(f"   Errors: {len(errors)}, Saved: {errors_path}")
    
    # Skipped samples
    print("\n12. SKIPPED SAMPLES")
    skipped = []
    for item in raw_split["val"]:
        responses = item.get("responses", {})
        for model_name in AVAILABLE_MODELS:
            if model_name not in responses:
                skipped.append({
                    "question_id": item.get("user_prompt_index", -1),
                    "source_model": model_name,
                    "missing_reason": "source_data_missing",
                    "timestamp": datetime.now().isoformat(),
                })
    for cid in empty_claim_samples:
        skipped.append({"case_id": cid, "reason": "empty_claim"})
    skipped_path = RESULTS_DIR / "skipped_samples.jsonl"
    with open(skipped_path, 'w') as f:
        for s in skipped:
            f.write(json.dumps(s) + "\n")
    print(f"   Skipped: {len(skipped)}")

    # Run manifest
    print("\n13. RUN MANIFEST")
    finished_at = datetime.now(timezone.utc).isoformat()
    duration = time.time() - start_time

    dataset_revision = "aab54518c2a7c0d25fff8bffbf5337d0321de142"
    manifest = {
        "evaluation_name": "Zero-shot Claim-level NLI Faithfulness Baseline on RAGognize Validation",
        "inference_git_commit": git_info.get('commit', 'unknown'),
        "git_branch": git_info.get('branch', 'unknown'),
        "working_tree_dirty": git_info.get('is_dirty', False),
        "git_diff_sha256": git_info.get('diff_sha256'),
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "duration_seconds": duration,
        "offline_mode": True,
        "dataset": {"name": "F4biian/RAGognize", "revision": dataset_revision, "cache_path": str(CACHE_DIR)},
        "split": {
            "val_size": 0.15,
            "seed": 42,
            "n_val_questions": n_val_questions,
            "n_models": len(AVAILABLE_MODELS),
            "n_theoretical_slots": n_theoretical_slots,
            "n_source_missing": n_source_missing,
            "n_runtime_skipped": len(skipped),
            "n_actual_valid": total_valid_responses,
            "source_missing_per_model": source_missing_per_model,
            "invariant": f"{n_theoretical_slots} = {n_source_missing} + {total_valid_responses}",
        },
        "model": {"name": DEFAULT_MODEL_NAME, "entailment_idx": model.entailment_idx, "neutral_idx": model.neutral_idx, "contradiction_idx": model.contradiction_idx},
        "best_config": best_config,
        "environment": {"python_executable": sys.executable, "python_version": sys.version, "device": device, "packages": versions},
        "relevance_audit": relevance_audit,
        "label_semantics": {"unfaithful": 0, "faithful": 1},
        "artifacts": {
            "validation_predictions.csv": {"path": str(pred_path), "rows": len(pred_df), "sha256": pred_hash},
            "claim_window_predictions.jsonl": {"path": str(claim_path), "rows": len(faith_scores), "sha256": claim_hash},
            "metrics.json": {"path": str(metrics_path), "sha256": metrics_hash},
            "best_config.json": {"path": str(best_config_path)},
            "error_analysis.csv": {"path": str(errors_path), "rows": len(errors), "sha256": errors_hash},
            "skipped_samples.jsonl": {"path": str(skipped_path), "rows": len(skipped)},
        },
        "what_is_evaluated": {"faithfulness": "COMPLETE", "relevance": "NOT_AVAILABLE", "reliability": "NOT_AVAILABLE"},
    }
    
    manifest_path = RESULTS_DIR / "run_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    manifest_hash = compute_file_hash(str(manifest_path))
    print(f"   Saved: {manifest_path}, SHA-256: {manifest_hash[:16]}...")
    
    # Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"   Evaluation: Zero-shot Claim-level NLI Faithfulness Baseline")
    print(f"   Samples: {len(pred_df)}")
    print(f"   Duration: {duration:.1f}s")
    print(f"   Accuracy: {main_metrics['accuracy']:.4f}")
    print(f"   Balanced Accuracy: {main_metrics['balanced_accuracy']:.4f}")
    print(f"   Macro-F1: {main_metrics['f1_macro']:.4f}")
    print(f"   Relevance: NOT AVAILABLE")
    print(f"   Reliability: NOT AVAILABLE")
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    
    return manifest

if __name__ == "__main__":
    manifest = main()
