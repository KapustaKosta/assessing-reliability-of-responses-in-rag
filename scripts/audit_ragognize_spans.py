#!/usr/bin/env python3
"""
Audit RAGognize dataset hallucination spans.

This script:
1. Loads RAGognize from HuggingFace
2. Expands prompts to model-level samples
3. Validates hallucination span annotations
4. Checks for span overlaps, duplicates, mismatches
5. Outputs detailed audit reports
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from datasets import load_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_ragognize():
    """Load RAGognize dataset from HuggingFace."""
    logger.info("Loading RAGognize dataset from HuggingFace...")
    dataset = load_dataset("F4biian/RAGognize", trust_remote_code=True)
    logger.info(f"Dataset loaded: {dataset}")
    return dataset


def expand_prompts_to_samples(dataset) -> list[dict]:
    """
    Expand each prompt to individual model-level samples.
    
    Returns list of samples with structure:
    {
        "sample_id": "<user_prompt_index>::<model_name>",
        "group_id": "<user_prompt_index>",
        "model_name": "...",
        "context": "<documents_str>",
        "question": "<user_prompt>",
        "answer": "<response.text>",
        "hallucination_spans": [...],
        "answerable": bool,
        "information_type": "...",
        "category": "...",
        "split": "train|test"
    }
    """
    samples = []
    
    for split_name, split_data in dataset.items():
        logger.info(f"Processing split: {split_name} with {len(split_data)} prompts")
        
        for item in split_data:
            user_prompt_index = item.get("user_prompt_index", 0)
            user_prompt = item.get("user_prompt", "")
            
            # Get context from documents
            documents = item.get("documents", [])
            if isinstance(documents, list):
                context = " ".join(
                    doc.get("text", "") if isinstance(doc, dict) else str(doc)
                    for doc in documents
                )
            else:
                context = str(documents)
            
            # Get responses for each model
            responses = item.get("responses", {})
            
            for model_name, response_data in responses.items():
                if not isinstance(response_data, dict):
                    continue
                
                # Get answer
                answer = response_data.get("text", "")
                if not answer:
                    continue
                
                # Get hallucination spans
                raw_spans = response_data.get("hallucinations", [])
                hallucination_spans = []
                
                for span in raw_spans:
                    if isinstance(span, dict):
                        hallucination_spans.append({
                            "text": span.get("text", ""),
                            "start": span.get("start", 0),
                            "end": span.get("end", 0),
                            "valid": span.get("valid", True),
                        })
                
                # Get metadata
                answerable = item.get("answerable", True)
                information_type = item.get("information_type", "unknown")
                category = item.get("category", "unknown")
                
                sample = {
                    "sample_id": f"{user_prompt_index}::{model_name}",
                    "group_id": str(user_prompt_index),
                    "model_name": model_name,
                    "context": context,
                    "question": user_prompt,
                    "answer": answer,
                    "hallucination_spans": hallucination_spans,
                    "answerable": answerable,
                    "information_type": information_type,
                    "category": category,
                    "split": split_name,
                }
                
                samples.append(sample)
    
    return samples


def validate_span(span: dict, answer: str) -> dict:
    """
    Validate a single hallucination span.
    
    Returns dict with:
    - valid: bool
    - error: str or None
    - details: dict with validation details
    """
    result = {"valid": True, "error": None, "details": {}}
    
    start = span.get("start")
    end = span.get("end")
    span_text = span.get("text", "")
    
    # Check types
    if not isinstance(start, int) or not isinstance(end, int):
        result["valid"] = False
        result["error"] = f"start/end must be int, got {type(start).__name__}/{type(end).__name__}"
        return result
    
    # Check range: 0 <= start < end <= len(answer)
    if start < 0:
        result["valid"] = False
        result["error"] = f"start must be >= 0, got {start}"
        return result
    
    if end <= start:
        result["valid"] = False
        result["error"] = f"end must be > start, got start={start}, end={end}"
        return result
    
    if end > len(answer):
        result["valid"] = False
        result["error"] = f"end={end} exceeds answer length={len(answer)}"
        return result
    
    # Check text match
    extracted_text = answer[start:end]
    if extracted_text != span_text:
        result["details"]["text_mismatch"] = True
        result["details"]["extracted_text"] = extracted_text
        result["details"]["given_text"] = span_text
    
    # Check valid flag
    valid_flag = span.get("valid", True)
    result["details"]["valid_flag"] = valid_flag
    
    return result


def audit_samples(samples: list[dict]) -> dict:
    """
    Audit all samples for span validation.
    
    Returns audit report with statistics and anomalies.
    """
    stats = {
        "total_prompts": set(),
        "total_responses": 0,
        "hallucinated_responses": 0,
        "clean_responses": 0,
        "total_spans": 0,
        "invalid_spans": 0,
        "mismatch_spans": 0,
        "duplicate_spans": 0,
        "overlapping_spans": 0,
        "containing_spans": 0,
        "span_issues_by_model": defaultdict(int),
        "label_distribution_by_model": defaultdict(lambda: {"hallucinated": 0, "clean": 0}),
        "answer_length_distribution": [],
        "context_length_distribution": [],
        "span_length_distribution": [],
    }
    
    invalid_spans_log = []
    anomalies = []
    
    # Group samples by sample_id for duplicate detection
    seen_spans = defaultdict(list)
    
    for sample in samples:
        sample_id = sample["sample_id"]
        group_id = sample["group_id"]
        model_name = sample["model_name"]
        answer = sample["answer"]
        spans = sample.get("hallucination_spans", [])
        
        stats["total_prompts"].add(group_id)
        stats["total_responses"] += 1
        
        # Track answer/context lengths
        stats["answer_length_distribution"].append(len(answer))
        stats["context_length_distribution"].append(len(sample["context"]))
        
        # Check for hallucination
        valid_spans = [s for s in spans if s.get("valid", True)]
        
        if len(valid_spans) > 0:
            stats["hallucinated_responses"] += 1
            stats["label_distribution_by_model"][model_name]["hallucinated"] += 1
        else:
            stats["clean_responses"] += 1
            stats["label_distribution_by_model"][model_name]["clean"] += 1
        
        stats["total_spans"] += len(spans)
        
        # Check each span
        for span_idx, span in enumerate(spans):
            validation = validate_span(span, answer)
            
            if not validation["valid"]:
                stats["invalid_spans"] += 1
                stats["span_issues_by_model"][model_name] += 1
                
                anomaly = {
                    "sample_id": sample_id,
                    "span_idx": span_idx,
                    "span": span,
                    "error": validation["error"],
                    "answer_length": len(answer),
                }
                invalid_spans_log.append(anomaly)
            elif validation["details"].get("text_mismatch"):
                stats["mismatch_spans"] += 1
                
                anomaly = {
                    "sample_id": sample_id,
                    "span_idx": span_idx,
                    "span": span,
                    "error": "text_mismatch",
                    "extracted_text": validation["details"]["extracted_text"],
                    "given_text": validation["details"]["given_text"],
                }
                anomalies.append(anomaly)
            
            # Track span for duplicate/overlap detection
            span_key = (span.get("start"), span.get("end"), span.get("text"))
            seen_spans[span_key].append((sample_id, span_idx))
            
            # Track span length
            span_len = span.get("end", 0) - span.get("start", 0)
            stats["span_length_distribution"].append(span_len)
        
        # Check for duplicates (same start/end/text)
        for span_key, occurrences in seen_spans.items():
            if len(occurrences) > 1:
                stats["duplicate_spans"] += 1
        
        # Check for overlapping spans
        valid_span_list = [(s.get("start", 0), s.get("end", 0)) for s in spans]
        for i in range(len(valid_span_list)):
            for j in range(i + 1, len(valid_span_list)):
                start_i, end_i = valid_span_list[i]
                start_j, end_j = valid_span_list[j]
                
                # Check overlap
                if start_i < end_j and start_j < end_i:
                    stats["overlapping_spans"] += 1
                
                # Check containment
                if start_i <= start_j and end_j <= end_i:
                    stats["containing_spans"] += 1
                if start_j <= start_i and end_i <= end_j:
                    stats["containing_spans"] += 1
    
    # Convert sets to counts
    stats["total_prompts"] = len(stats["total_prompts"])
    
    return {
        "stats": stats,
        "invalid_spans": invalid_spans_log,
        "anomalies": anomalies,
    }


def main():
    """Main function."""
    # Load dataset
    dataset = load_ragognize()
    
    # Expand to samples
    samples = expand_prompts_to_samples(dataset)
    logger.info(f"Expanded to {len(samples)} samples")
    
    # Audit
    audit_result = audit_samples(samples)
    stats = audit_result["stats"]
    
    # Print summary
    logger.info("=" * 60)
    logger.info("RAGOGNIZE DATA AUDIT SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total prompts: {stats['total_prompts']}")
    logger.info(f"Total responses: {stats['total_responses']}")
    logger.info(f"Hallucinated responses: {stats['hallucinated_responses']}")
    logger.info(f"Clean responses: {stats['clean_responses']}")
    logger.info(f"Total spans: {stats['total_spans']}")
    logger.info(f"Invalid spans: {stats['invalid_spans']}")
    logger.info(f"Mismatch spans: {stats['mismatch_spans']}")
    logger.info(f"Duplicate spans: {stats['duplicate_spans']}")
    logger.info(f"Overlapping spans: {stats['overlapping_spans']}")
    logger.info(f"Containing spans: {stats['containing_spans']}")
    logger.info("")
    logger.info("Label distribution by model:")
    for model, dist in sorted(stats["label_distribution_by_model"].items()):
        total = dist["hallucinated"] + dist["clean"]
        rate = dist["hallucinated"] / total if total > 0 else 0
        logger.info(f"  {model}: {dist['hallucinated']} hallucinated, {dist['clean']} clean ({rate:.1%})")
    
    # Save reports
    output_dir = Path("reports/token_level")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON stats
    stats_json = {k: v if not isinstance(v, defaultdict) else dict(v) for k, v in stats.items()}
    stats_json["answer_length_stats"] = {
        "mean": sum(stats["answer_length_distribution"]) / len(stats["answer_length_distribution"]) if stats["answer_length_distribution"] else 0,
        "min": min(stats["answer_length_distribution"]) if stats["answer_length_distribution"] else 0,
        "max": max(stats["answer_length_distribution"]) if stats["answer_length_distribution"] else 0,
    }
    stats_json["context_length_stats"] = {
        "mean": sum(stats["context_length_distribution"]) / len(stats["context_length_distribution"]) if stats["context_length_distribution"] else 0,
        "min": min(stats["context_length_distribution"]) if stats["context_length_distribution"] else 0,
        "max": max(stats["context_length_distribution"]) if stats["context_length_distribution"] else 0,
    }
    stats_json["span_length_stats"] = {
        "mean": sum(stats["span_length_distribution"]) / len(stats["span_length_distribution"]) if stats["span_length_distribution"] else 0,
        "min": min(stats["span_length_distribution"]) if stats["span_length_distribution"] else 0,
        "max": max(stats["span_length_distribution"]) if stats["span_length_distribution"] else 0,
    }
    
    with open(output_dir / "data_audit.json", "w") as f:
        json.dump(stats_json, f, indent=2)
    logger.info(f"Saved data_audit.json")
    
    # Save invalid spans
    with open(output_dir / "invalid_spans.jsonl", "w") as f:
        for item in audit_result["invalid_spans"]:
            f.write(json.dumps(item) + "\n")
    logger.info(f"Saved invalid_spans.jsonl with {len(audit_result['invalid_spans'])} entries")
    
    # Save full audit
    full_audit = {
        "summary": {
            "total_prompts": stats["total_prompts"],
            "total_responses": stats["total_responses"],
            "hallucinated_responses": stats["hallucinated_responses"],
            "clean_responses": stats["clean_responses"],
            "total_spans": stats["total_spans"],
            "invalid_spans": stats["invalid_spans"],
            "mismatch_spans": stats["mismatch_spans"],
            "text_match_rate": 1.0 - (stats["mismatch_spans"] / stats["total_spans"]) if stats["total_spans"] > 0 else 1.0,
        },
        "label_distribution_by_model": dict(stats["label_distribution_by_model"]),
    }
    
    with open(output_dir / "data_audit.md", "w") as f:
        f.write("# RAGognize Data Audit Report\n\n")
        f.write(f"**Date**: 2026-07-22\n\n")
        f.write("## Summary\n\n")
        f.write(f"- Total prompts: {stats['total_prompts']}\n")
        f.write(f"- Total responses: {stats['total_responses']}\n")
        f.write(f"- Hallucinated responses: {stats['hallucinated_responses']}\n")
        f.write(f"- Clean responses: {stats['clean_responses']}\n")
        f.write(f"- Total spans: {stats['total_spans']}\n")
        f.write(f"- Invalid spans: {stats['invalid_spans']}\n")
        f.write(f"- Mismatch spans: {stats['mismatch_spans']}\n")
        f.write(f"- **Text match rate**: {full_audit['summary']['text_match_rate']:.2%}\n\n")
        
        f.write("## Label Distribution by Model\n\n")
        for model, dist in sorted(stats["label_distribution_by_model"].items()):
            total = dist["hallucinated"] + dist["clean"]
            rate = dist["hallucinated"] / total if total > 0 else 0
            f.write(f"- **{model}**: {dist['hallucinated']} hallucinated, {dist['clean']} clean ({rate:.1%})\n")
        
        f.write("\n## Length Statistics\n\n")
        f.write(f"- Answer length: mean={stats_json['answer_length_stats']['mean']:.0f}, ")
        f.write(f"min={stats_json['answer_length_stats']['min']}, ")
        f.write(f"max={stats_json['answer_length_stats']['max']}\n")
        f.write(f"- Context length: mean={stats_json['context_length_stats']['mean']:.0f}, ")
        f.write(f"min={stats_json['context_length_stats']['min']}, ")
        f.write(f"max={stats_json['context_length_stats']['max']}\n")
        f.write(f"- Span length: mean={stats_json['span_length_stats']['mean']:.0f}, ")
        f.write(f"min={stats_json['span_length_stats']['min']}, ")
        f.write(f"max={stats_json['span_length_stats']['max']}\n")
    
    logger.info(f"Saved data_audit.md")
    
    # GATE 1: Check text match rate
    match_rate = full_audit["summary"]["text_match_rate"]
    invalid_count = stats["invalid_spans"] + stats["mismatch_spans"]
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("GATE 1 CHECK")
    logger.info("=" * 60)
    logger.info(f"Text match rate: {match_rate:.2%}")
    logger.info(f"Invalid spans: {invalid_count}")
    
    if match_rate < 1.0:
        logger.error("GATE 1 FAILED: Text match rate is not 100%")
        logger.error(f"Mismatch spans: {stats['mismatch_spans']}")
        logger.error("Stopping before training.")
        return 1
    
    if stats["invalid_spans"] > 0:
        logger.warning(f"WARNING: {stats['invalid_spans']} invalid spans found")
        logger.warning("These spans will be filtered out during training.")
    
    logger.info("GATE 1 PASSED: All valid spans have correct text match")
    logger.info("Proceeding to data splitting...")
    
    # Save samples for next stage
    samples_output = Path("data/processed/ragognize_raw_samples.jsonl")
    with open(samples_output, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    logger.info(f"Saved raw samples to {samples_output}")
    
    return 0


if __name__ == "__main__":
    exit(main())
