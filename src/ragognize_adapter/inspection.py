"""
Comprehensive inspection of RAGognize dataset structure.
"""

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import DatasetDict, Dataset

from .constants import SOURCE_MODELS, RESULTS_DIR

logger = logging.getLogger(__name__)


def extract_response_fields(sample: dict, model_name: str) -> dict:
    """Extract fields from a single model response."""
    responses = sample.get("responses", {})
    response = responses.get(model_name, {})
    
    return {
        "has_response": model_name in responses,
        "text": response.get("text") if isinstance(response, dict) else None,
        "has_text": "text" in response if isinstance(response, dict) else False,
        "hallucinations": response.get("hallucinations") if isinstance(response, dict) else None,
        "details": response.get("details") if isinstance(response, dict) else None,
    }


def inspect_hallucinations_structure(hallucinations: Any) -> dict:
    """
    Inspect the structure of hallucinations field.
    
    Returns:
        Dict with type info, field types, and sample.
    """
    result = {
        "type": str(type(hallucinations).__name__),
        "is_none": hallucinations is None,
        "is_list": isinstance(hallucinations, list),
        "is_dict": isinstance(hallucinations, dict),
        "is_empty": False,
        "length": 0,
        "item_types": [],
        "sample": None,
    }
    
    if hallucinations is None:
        return result
    
    if isinstance(hallucinations, list):
        result["is_empty"] = len(hallucinations) == 0
        result["length"] = len(hallucinations)
        
        if len(hallucinations) > 0:
            # Collect all unique field types
            all_keys = set()
            for item in hallucinations[:50]:  # Sample first 50
                if isinstance(item, dict):
                    all_keys.update(item.keys())
            
            result["all_keys"] = sorted(list(all_keys))
            
            # Get item type sample
            if isinstance(hallucinations[0], dict):
                result["item_types"] = ["dict"]
                result["sample"] = {k: _truncate(v, 100) for k, v in hallucinations[0].items()}
            elif isinstance(hallucinations[0], (list, tuple)):
                result["item_types"] = ["list_or_tuple"]
            else:
                result["item_types"] = [type(hallucinations[0]).__name__]
                result["sample"] = _truncate(hallucinations[0], 100)
    
    elif isinstance(hallucinations, dict):
        result["all_keys"] = sorted(list(hallucinations.keys()))
        result["sample"] = {k: _truncate(v, 100) for k, v in hallucinations.items()}
    
    return result


def _truncate(value: Any, max_len: int = 100) -> str:
    """Truncate a value for display."""
    if value is None:
        return None
    s = str(value)
    if len(s) > max_len:
        return s[:max_len] + "..."
    return s


def comprehensive_response_inspection(
    dataset: DatasetDict,
    n_samples_per_category: int = 20,
) -> dict:
    """
    Thoroughly inspect the responses structure across the dataset.
    
    Args:
        dataset: The loaded dataset.
        n_samples_per_category: Number of samples per answerable category.
    
    Returns:
        Comprehensive inspection results.
    """
    results = {
        "dataset_info": {},
        "response_structure": {},
        "hallucinations_analysis": {},
        "details_analysis": {},
        "consistency_check": {},
    }
    
    # Basic dataset info
    for split_name, split_data in dataset.items():
        results["dataset_info"][split_name] = {
            "rows": len(split_data),
            "features": list(split_data.features.keys()),
        }
    
    # Collect samples by category
    train_data = dataset.get("train", dataset.get(list(dataset.keys())[0]))
    
    # Get samples by answerable status
    answerable_true = []
    answerable_false = []
    
    for i, sample in enumerate(train_data):
        if sample.get("answerable", False):
            answerable_true.append((i, sample))
        else:
            answerable_false.append((i, sample))
    
    # Sample from each category
    samples_to_check = []
    
    # Take samples ensuring variety
    import random
    random.seed(42)
    
    random.shuffle(answerable_true)
    random.shuffle(answerable_false)
    
    samples_to_check.extend(answerable_true[:n_samples_per_category])
    samples_to_check.extend(answerable_false[:n_samples_per_category])
    
    # Analyze each sample
    hallucination_types = Counter()
    hallucination_keys = Counter()
    hallucination_field_values = {}
    
    details_keys = Counter()
    details_samples = []
    
    valid_field_stats = {"true": 0, "false": 0, "missing": 0}
    
    for idx, sample in samples_to_check:
        responses = sample.get("responses", {})
        
        for model_name in SOURCE_MODELS:
            if model_name not in responses:
                continue
            
            response = responses[model_name]
            if not isinstance(response, dict):
                continue
            
            # Check hallucinations structure
            hallucinations = response.get("hallucinations")
            hall_info = inspect_hallucinations_structure(hallucinations)
            
            hallucination_types[hall_info["type"]] += 1
            
            if hall_info.get("all_keys"):
                for key in hall_info["all_keys"]:
                    hallucination_keys[key] += 1
            
            # Analyze valid field
            if hall_info["is_list"] and not hall_info["is_empty"]:
                for item in hallucinations:
                    if isinstance(item, dict):
                        if "valid" in item:
                            if item["valid"]:
                                valid_field_stats["true"] += 1
                            else:
                                valid_field_stats["false"] += 1
                        else:
                            valid_field_stats["missing"] += 1
                    # Track other possible fields
                    for field in ["start", "end", "text", "label", "type", "explanation"]:
                        if isinstance(item, dict) and field in item:
                            if field not in hallucination_field_values:
                                hallucination_field_values[field] = {
                                    "count": 0,
                                    "sample": _truncate(item.get(field), 50),
                                    "types": set(),
                                }
                            hallucination_field_values[field]["count"] += 1
                            hallucination_field_values[field]["types"].add(
                                type(item.get(field)).__name__
                            )
            
            # Check details structure
            details = response.get("details")
            if details is not None and isinstance(details, dict):
                for key in details.keys():
                    details_keys[key] += 1
                
                if len(details_samples) < 10:
                    details_samples.append({
                        "model": model_name,
                        "sample_id": sample.get("user_prompt_index"),
                        "keys": list(details.keys()),
                        "sample": {k: _truncate(v, 100) for k, v in details.items()},
                    })
    
    # Compile hallucination analysis
    results["hallucinations_analysis"] = {
        "types_found": dict(hallucination_types),
        "keys_found": dict(hallucination_keys),
        "field_analysis": {
            k: {
                "count": v["count"],
                "sample": v["sample"],
                "types": list(v["types"]),
            }
            for k, v in hallucination_field_values.items()
        },
        "valid_field_stats": valid_field_stats,
    }
    
    # Compile details analysis
    results["details_analysis"] = {
        "keys_found": dict(details_keys),
        "sample_details": details_samples,
    }
    
    # Check for specific annotation fields
    possible_annotation_fields = [
        "addressed_user_prompt",
        "relevance",
        "relevancy",
        "answerable",
        "correctness",
        "hallucination_result",
        "faithfulness",
        "grounded",
        "groundedness",
    ]
    
    found_annotation_fields = {}
    for field in possible_annotation_fields:
        if field in hallucination_keys or field in details_keys:
            found_annotation_fields[field] = {
                "in_hallucinations": field in hallucination_keys,
                "in_details": field in details_keys,
                "count": hallucination_keys.get(field, 0) + details_keys.get(field, 0),
            }
    
    results["details_analysis"]["possible_annotation_fields"] = found_annotation_fields
    
    return results


def analyze_span_validity(
    dataset: DatasetDict,
    max_samples: int = 500,
) -> dict:
    """
    Analyze hallucination span validity and matching.
    
    Args:
        dataset: The loaded dataset.
        max_samples: Maximum number of samples to analyze.
    
    Returns:
        Analysis of span validity and matching.
    """
    results = {
        "total_responses_analyzed": 0,
        "responses_with_hallucinations": 0,
        "total_spans": 0,
        "span_analysis": {
            "exact_match": 0,
            "stripped_match": 0,
            "mismatch": 0,
            "out_of_bounds": 0,
            "empty_span": 0,
            "missing_offset": 0,
        },
        "valid_field_stats": {
            "valid_true": 0,
            "valid_false": 0,
            "valid_missing": 0,
        },
        "mismatch_examples": [],
        "interval_type": None,  # Will be determined
    }
    
    mismatch_examples = []
    
    for split_name, split_data in dataset.items():
        for i, sample in enumerate(split_data):
            if i >= max_samples:
                break
            
            answer = sample.get("user_prompt", "")  # This is the question, not the answer
            
            # Actually, we need to check responses
            responses = sample.get("responses", {})
            
            for model_name in SOURCE_MODELS:
                if model_name not in responses:
                    continue
                
                response = responses[model_name]
                if not isinstance(response, dict):
                    continue
                
                results["total_responses_analyzed"] += 1
                
                hallucinations = response.get("hallucinations")
                if not hallucinations:
                    continue
                
                results["responses_with_hallucinations"] += 1
                
                model_answer = response.get("text", "")
                
                if not isinstance(hallucinations, list):
                    continue
                
                for span in hallucinations:
                    if not isinstance(span, dict):
                        continue
                    
                    results["total_spans"] += 1
                    
                    # Check valid field
                    if "valid" in span:
                        if span["valid"]:
                            results["valid_field_stats"]["valid_true"] += 1
                        else:
                            results["valid_field_stats"]["valid_false"] += 1
                    else:
                        results["valid_field_stats"]["valid_missing"] += 1
                    
                    # Check span coordinates
                    start = span.get("start")
                    end = span.get("end")
                    span_text = span.get("text")
                    
                    if start is None or end is None:
                        results["span_analysis"]["missing_offset"] += 1
                        continue
                    
                    if not isinstance(start, int) or not isinstance(end, int):
                        results["span_analysis"]["missing_offset"] += 1
                        continue
                    
                    if start >= end:
                        results["span_analysis"]["empty_span"] += 1
                        continue
                    
                    if end > len(model_answer):
                        results["span_analysis"]["out_of_bounds"] += 1
                        if len(mismatch_examples) < 10:
                            mismatch_examples.append({
                                "model": model_name,
                                "prompt_index": sample.get("user_prompt_index"),
                                "issue": "out_of_bounds",
                                "start": start,
                                "end": end,
                                "answer_length": len(model_answer),
                                "span_text": _truncate(span_text, 50),
                            })
                        continue
                    
                    if start < 0:
                        results["span_analysis"]["out_of_bounds"] += 1
                        continue
                    
                    # Check text match
                    if span_text is None:
                        results["span_analysis"]["missing_offset"] += 1
                        continue
                    
                    extracted = model_answer[start:end]
                    
                    if extracted == span_text:
                        results["span_analysis"]["exact_match"] += 1
                    elif extracted.strip() == span_text.strip():
                        results["span_analysis"]["stripped_match"] += 1
                    else:
                        results["span_analysis"]["mismatch"] += 1
                        if len(mismatch_examples) < 10:
                            mismatch_examples.append({
                                "model": model_name,
                                "prompt_index": sample.get("user_prompt_index"),
                                "issue": "text_mismatch",
                                "start": start,
                                "end": end,
                                "expected_text": _truncate(span_text, 50),
                                "extracted_text": _truncate(extracted, 50),
                            })
    
    results["mismatch_examples"] = mismatch_examples
    
    # Determine interval type (closed vs open)
    if results["span_analysis"]["exact_match"] > results["span_analysis"]["stripped_match"]:
        results["interval_type"] = "closed"  # [start, end)
    else:
        results["interval_type"] = "open_or_whitespace"
    
    return results


def save_inspection_results(results: dict, output_dir: Path) -> None:
    """Save inspection results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save schema summary
    schema_path = output_dir / "schema_summary.json"
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(results.get("hallucinations_analysis", {}), f, indent=2, ensure_ascii=False)
    logger.info(f"Saved schema summary to: {schema_path}")
    
    # Save details analysis
    details_path = output_dir / "details_summary.json"
    with open(details_path, "w", encoding="utf-8") as f:
        json.dump(results.get("details_analysis", {}), f, indent=2, ensure_ascii=False)
    logger.info(f"Saved details summary to: {details_path}")
    
    # Save span validation
    if "span_analysis" in results:
        span_path = output_dir / "span_validation_summary.json"
        with open(span_path, "w", encoding="utf-8") as f:
            json.dump(results["span_analysis"], f, indent=2, ensure_ascii=False)
        logger.info(f"Saved span validation to: {span_path}")
    
    # Save mismatch examples
    if "mismatch_examples" in results and results["mismatch_examples"]:
        import csv
        mismatch_path = output_dir / "span_mismatch_examples.csv"
        with open(mismatch_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=results["mismatch_examples"][0].keys())
            writer.writeheader()
            writer.writerows(results["mismatch_examples"])
        logger.info(f"Saved mismatch examples to: {mismatch_path}")
