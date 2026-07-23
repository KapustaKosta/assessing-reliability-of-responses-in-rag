"""
Data splitting utilities for RAGognize adapter.
"""

import csv
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional

from datasets import DatasetDict

from .constants import SOURCE_MODELS, DEFAULT_SEED, DEFAULT_VAL_RATIO

logger = logging.getLogger(__name__)


def create_prompt_split(
    dataset: DatasetDict,
    val_ratio: float = DEFAULT_VAL_RATIO,
    seed: int = DEFAULT_SEED,
    stratify_by: Optional[list[str]] = None,
) -> dict:
    """
    Create prompt-level train/validation split.
    
    This ensures all model responses from the same prompt stay in the same split.
    
    Args:
        dataset: DatasetDict with 'train' split.
        val_ratio: Fraction of prompts for validation.
        seed: Random seed.
        stratify_by: Fields to stratify by (e.g., ['answerable', 'category']).
    
    Returns:
        Dictionary with split information:
        - train_prompts: Set of prompt indices for training.
        - val_prompts: Set of prompt indices for validation.
        - test_prompts: Set of prompt indices for test (from official test).
    """
    random.seed(seed)
    
    # Collect train prompts with metadata
    train_data = dataset.get("train", dataset.get(list(dataset.keys())[0]))
    
    prompt_meta = {}  # prompt_index -> metadata
    
    for sample in train_data:
        prompt_idx = sample.get("user_prompt_index")
        if prompt_idx is None:
            continue
        
        if prompt_idx not in prompt_meta:
            prompt_meta[prompt_idx] = {
                "answerable": sample.get("answerable", True),
                "category": sample.get("category", ""),
                "information_type": sample.get("information_type", ""),
                "has_hallucination_responses": set(),
            }
    
    # Get hallucination status from responses
    for sample in train_data:
        prompt_idx = sample.get("user_prompt_index")
        if prompt_idx is None:
            continue
        
        responses = sample.get("responses", {})
        for model_name in SOURCE_MODELS:
            if model_name not in responses:
                continue
            
            response = responses[model_name]
            if not isinstance(response, dict):
                continue
            
            hallucinations = response.get("hallucinations", [])
            if hallucinations and isinstance(hallucinations, list):
                # Check for valid hallucinations
                has_valid = False
                for h in hallucinations:
                    if isinstance(h, dict) and h.get("valid", True):
                        has_valid = True
                        break
                
                if has_valid:
                    prompt_meta[prompt_idx]["has_hallucination_responses"].add(model_name)
    
    # Convert to list for stratification
    prompt_list = list(prompt_meta.keys())
    random.shuffle(prompt_list)
    
    # Simple random split (stratification optional)
    n_val = max(1, int(len(prompt_list) * val_ratio))
    
    val_prompts = set(prompt_list[:n_val])
    train_prompts = set(prompt_list[n_val:])
    
    # Get test prompts from official test
    test_data = dataset.get("test", dataset.get(list(dataset.keys())[0]))
    test_prompts = set()
    for sample in test_data:
        prompt_idx = sample.get("user_prompt_index")
        if prompt_idx is not None:
            test_prompts.add(prompt_idx)
    
    return {
        "train_prompts": list(train_prompts),
        "val_prompts": list(val_prompts),
        "test_prompts": list(test_prompts),
        "train_count": len(train_prompts),
        "val_count": len(val_prompts),
        "test_count": len(test_prompts),
        "val_ratio": val_ratio,
        "seed": seed,
        "prompt_metadata": {
            str(k): v for k, v in prompt_meta.items()
        },
    }


def apply_split(
    dataset: DatasetDict,
    split_info: dict,
) -> dict:
    """
    Apply split to dataset and return samples by split.
    
    Args:
        dataset: Full dataset.
        split_info: Output from create_prompt_split.
    
    Returns:
        Dictionary with lists of UnifiedSamples per split.
    """
    from .adapter import RAGognizeAdapter
    
    train_prompts = set(split_info["train_prompts"])
    val_prompts = set(split_info["val_prompts"])
    
    adapter = RAGognizeAdapter()
    
    result = {
        "train": [],
        "val": [],
        "test": [],
    }
    
    # Process each split
    for split_name, split_data in dataset.items():
        for row_idx, sample in enumerate(split_data):
            prompt_idx = sample.get("user_prompt_index")
            
            if split_name == "test":
                target_split = "test"
            elif prompt_idx in train_prompts:
                target_split = "train"
            elif prompt_idx in val_prompts:
                target_split = "val"
            else:
                logger.warning(f"Prompt {prompt_idx} not in any split")
                continue
            
            transformed = adapter.transform_sample(sample, split_name, row_index=row_idx)
            
            # Update split assignment
            for sample_obj in transformed:
                sample_obj.split = target_split
            
            result[target_split].extend(transformed)
    
    return result


def save_split_manifest(
    split_info: dict,
    output_path: Path,
) -> None:
    """
    Save split manifest to CSV.
    
    Args:
        split_info: Output from create_prompt_split.
        output_path: Path to save CSV.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    prompt_meta = split_info.get("prompt_metadata", {})
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "user_prompt_index", "split", "answerable",
            "category", "information_type", "n_hallucinated_models"
        ])
        
        for prompt_idx in sorted(prompt_meta.keys()):
            meta = prompt_meta[prompt_idx]
            
            if prompt_idx in split_info["train_prompts"]:
                split = "train"
            elif prompt_idx in split_info["val_prompts"]:
                split = "val"
            elif prompt_idx in split_info["test_prompts"]:
                split = "test"
            else:
                split = "unknown"
            
            writer.writerow([
                prompt_idx,
                split,
                meta.get("answerable", True),
                meta.get("category", ""),
                meta.get("information_type", ""),
                len(meta.get("has_hallucination_responses", set())),
            ])
    
    logger.info(f"Saved split manifest to: {output_path}")


def save_split_summary(
    split_info: dict,
    expanded_counts: dict,
    output_path: Path,
) -> None:
    """
    Save split summary to JSON.
    
    Args:
        split_info: Output from create_prompt_split.
        expanded_counts: Expanded sample counts per split.
        output_path: Path to save JSON.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Compute label distribution
    train_prompts = set(split_info["train_prompts"])
    val_prompts = set(split_info["val_prompts"])
    test_prompts = set(split_info["test_prompts"])
    
    label_stats = {
        "train": {"hallucinated": 0, "faithful": 0, "total": 0},
        "val": {"hallucinated": 0, "faithful": 0, "total": 0},
        "test": {"hallucinated": 0, "faithful": 0, "total": 0},
    }
    
    model_stats = {
        "train": {},
        "val": {},
        "test": {},
    }
    
    prompt_meta = split_info.get("prompt_metadata", {})
    
    # Count at prompt level
    for prompt_idx, meta in prompt_meta.items():
        if prompt_idx in train_prompts:
            split = "train"
        elif prompt_idx in val_prompts:
            split = "val"
        else:
            continue
        
        has_hall = len(meta.get("has_hallucination_responses", set())) > 0
        label_stats[split]["total"] += 1
        if has_hall:
            label_stats[split]["hallucinated"] += 1
        else:
            label_stats[split]["faithful"] += 1
    
    # Get model distribution from expanded counts
    for split_name, samples in expanded_counts.items():
        if isinstance(samples[0], dict):
            sample_list = samples
        else:
            sample_list = [s.to_dict() if hasattr(s, 'to_dict') else s for s in samples]
        
        for sample in sample_list:
            model = sample.get("source_model", "unknown")
            if model not in model_stats[split_name]:
                model_stats[split_name][model] = 0
            model_stats[split_name][model] += 1
    
    summary = {
        "split_info": {
            "train_prompts": len(train_prompts),
            "val_prompts": len(val_prompts),
            "test_prompts": len(test_prompts),
            "val_ratio": split_info.get("val_ratio"),
            "seed": split_info.get("seed"),
        },
        "prompt_level_stats": label_stats,
        "expanded_sample_counts": {
            split: len(samples)
            for split, samples in expanded_counts.items()
        },
        "model_distribution": model_stats,
        "answerable_distribution": {
            "train": sum(
                1 for p in prompt_meta.values()
                if p.get("answerable", True) and str(p) in train_prompts
            ) if train_prompts else 0,
            "val": sum(
                1 for p in prompt_meta.values()
                if p.get("answerable", True) and str(p) in val_prompts
            ) if val_prompts else 0,
        },
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved split summary to: {output_path}")


def verify_no_overlap(
    train_samples: list,
    val_samples: list,
    test_samples: list,
) -> dict:
    """
    Verify that splits have no overlapping prompt indices.
    
    Args:
        train_samples: Training samples.
        val_samples: Validation samples.
        test_samples: Test samples.
    
    Returns:
        Dictionary with verification results.
    """
    train_prompts = {s.user_prompt_index for s in train_samples}
    val_prompts = {s.user_prompt_index for s in val_samples}
    test_prompts = {s.user_prompt_index for s in test_samples}
    
    train_val = train_prompts & val_prompts
    train_test = train_prompts & test_prompts
    val_test = val_prompts & test_prompts
    
    return {
        "train_prompts": len(train_prompts),
        "val_prompts": len(val_prompts),
        "test_prompts": len(test_prompts),
        "train_val_overlap": len(train_val),
        "train_test_overlap": len(train_test),
        "val_test_overlap": len(val_test),
        "has_leakage": bool(train_val or train_test or val_test),
        "overlap_details": {
            "train_val": sorted(list(train_val))[:10] if train_val else [],
            "train_test": sorted(list(train_test))[:10] if train_test else [],
            "val_test": sorted(list(val_test))[:10] if val_test else [],
        },
    }
