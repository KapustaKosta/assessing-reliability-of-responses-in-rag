"""
NLI inference module for Faithfulness and Relevance detection.

Handles model loading, tokenizer-aware chunk windowing, batch inference,
and caching of results for both faithfulness and relevance NLI tasks.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .constants import (
    DEFAULT_MODEL_NAME,
    DEVICE_PREFERENCE,
    CACHE_DIR,
    DEFAULT_WINDOW_OVERLAP_TOKENS,
    MIN_WINDOW_TOKENS,
    MAX_CHUNK_WINDOWS_PER_CHUNK,
)


logger = logging.getLogger(__name__)


@dataclass
class ChunkWindow:
    """
    Represents a window within a chunk for NLI inference.
    
    Includes token range information for debugging and verification.
    """
    case_id: str
    chunk_id: int
    window_id: int
    window_text: str
    token_start: int  # Start token index in original chunk
    token_end: int    # End token index in original chunk
    token_count: int
    was_truncated: bool = False
    doc_source: str = ""  # Source document/chunk identifier


@dataclass
class NLIScore:
    """
    Single NLI inference result for Faithfulness or Relevance detection.
    
    Faithfulness: premise = context window, hypothesis = claim
    Relevance: premise = question, hypothesis = claim
    """
    case_id: str
    claim_id: int  # 0-indexed claim within answer
    claim_text: str  # The claim text
    chunk_id: int
    window_id: int
    
    # NLI probabilities (from model.config.id2label)
    entailment_probability: float
    neutral_probability: float
    contradiction_probability: float
    
    # Aggregated claim-level score
    claim_faithfulness_score: float = 0.0
    claim_relevance_score: float = 0.0
    
    # Predictions
    faithfulness_prediction: int = 0
    relevance_prediction: int = 0
    
    # For tracking
    task_type: str = "faithfulness"  # "faithfulness" or "relevance"
    premise: str = ""  # context window or question
    hypothesis: str = ""  # claim text
    
    def __post_init__(self):
        total = (self.entailment_probability + 
                  self.neutral_probability + 
                  self.contradiction_probability)
        if abs(total - 1.0) > 0.01:
            logger.warning(
                f"Probabilities sum to {total:.4f}, expected ~1.0 for case {self.case_id}"
            )


class NLIModel:
    """
    Wrapper for NLI model with dynamic label mapping.
    
    Supports both:
    - Faithfulness NLI: premise = context window, hypothesis = claim
    - Relevance NLI: premise = question, hypothesis = claim
    """
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: Optional[str] = None,
        cache_dir: Optional[Path] = None,
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine device
        if device is None:
            device = self._get_best_device()
        self.device = torch.device(device)
        
        logger.info(f"Loading model: {model_name}")
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Get dynamic label mapping from model config
        self.id2label = self.model.config.id2label
        self.label2id = self.model.config.label2id
        
        # Map our semantic labels to model labels
        self._setup_label_mapping()
        
        # Log model info
        self._log_model_info()
        
        # Cache for model info
        self.model_info_cache = self.cache_dir / "model_info.json"
    
    def _get_best_device(self) -> str:
        """Select the best available device."""
        for device_type in DEVICE_PREFERENCE:
            if device_type == "cuda" and torch.cuda.is_available():
                return "cuda"
            elif device_type == "mps" and torch.backends.mps.is_available():
                return "mps"
            elif device_type == "cpu":
                return "cpu"
        
        logger.warning("No preferred device available, falling back to CPU")
        return "cpu"
    
    def _setup_label_mapping(self) -> None:
        """
        Map our semantic NLI labels to model's actual labels.
        
        Label indices are read dynamically from model.config.id2label.
        No hardcoded label order assumptions.
        """
        self.entailment_idx = None
        self.neutral_idx = None
        self.contradiction_idx = None
        
        # Find labels based on semantic meaning
        for idx, label in self.id2label.items():
            label_lower = label.lower()
            if "entail" in label_lower or "support" in label_lower:
                self.entailment_idx = idx
            elif "contradict" in label_lower or "refute" in label_lower:
                self.contradiction_idx = idx
            elif "neutral" in label_lower:
                self.neutral_idx = idx
        
        # Fallback with warning
        if self.entailment_idx is None:
            logger.warning(
                f"Could not auto-detect NLI label mapping. "
                f"Found labels: {self.id2label}. "
                f"Model label order may differ from standard [entailment, neutral, contradiction]."
            )
            if len(self.id2label) >= 3:
                self.entailment_idx = 0
                self.neutral_idx = 1
                self.contradiction_idx = 2
            else:
                raise ValueError(f"Model has unexpected number of labels: {len(self.id2label)}")
        
        # Validate all indices are set
        assert self.entailment_idx is not None, "entailment_idx is None after mapping"
        assert self.neutral_idx is not None, "neutral_idx is None after mapping"
        assert self.contradiction_idx is not None, "contradiction_idx is None after mapping"
        
        # Validate indices are distinct
        assert len({self.entailment_idx, self.neutral_idx, self.contradiction_idx}) == 3, \
            f"Label indices are not distinct: {self.entailment_idx}, {self.neutral_idx}, {self.contradiction_idx}"
        
        logger.info(
            f"NLI label mapping (from model.config.id2label): "
            f"entailment={self.entailment_idx} ({self.id2label[self.entailment_idx]}), "
            f"neutral={self.neutral_idx} ({self.id2label[self.neutral_idx]}), "
            f"contradiction={self.contradiction_idx} ({self.id2label[self.contradiction_idx]})"
        )
    
    def _log_model_info(self) -> None:
        """Log model configuration info."""
        logger.info(f"Model architecture: {self.model.config.model_type}")
        logger.info(f"Number of labels: {self.model.config.num_labels}")
        logger.info(f"Label mapping: {self.id2label}")
        
        if hasattr(self.tokenizer, "model_max_length"):
            logger.info(f"Tokenizer max length: {self.tokenizer.model_max_length}")
        else:
            logger.info("Tokenizer max length: not specified (using default)")
    
    def save_model_info(self, path: Optional[Path] = None) -> None:
        """Save model info to JSON for reproducibility."""
        info = {
            "model_name": self.model_name,
            "device": str(self.device),
            "id2label": {str(k): v for k, v in self.id2label.items()},
            "label2id": {str(k): v for k, v in self.label2id.items()},
            "entailment_idx": self.entailment_idx,
            "neutral_idx": self.neutral_idx,
            "contradiction_idx": self.contradiction_idx,
        }
        
        path = path or self.model_info_cache
        with open(path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Model info saved to {path}")
    
    @property
    def max_length(self) -> int:
        """Get the model's maximum sequence length."""
        if hasattr(self.tokenizer, "model_max_length"):
            return self.tokenizer.model_max_length
        return 512  # Default fallback
    
    def create_chunk_windows(
        self,
        chunk_text: str,
        case_id: str,
        chunk_id: int,
        hypothesis_tokens: int,
        overlap_tokens: int = DEFAULT_WINDOW_OVERLAP_TOKENS,
    ) -> list[ChunkWindow]:
        """
        Create token-aware windows from a chunk for NLI inference.
        
        Claims are never truncated. Uses pair truncation only_first for premise.
        
        Args:
            chunk_text: The full chunk text
            case_id: Sample identifier
            chunk_id: Which chunk (1-8)
            hypothesis_tokens: Number of tokens reserved for the hypothesis
            overlap_tokens: Number of overlapping tokens between windows
            
        Returns:
            List of ChunkWindow objects with token range information
        """
        budget = self.max_length - hypothesis_tokens - 3
        
        if budget <= 0:
            logger.warning(
                f"Chunk budget ({budget}) is too small for hypothesis ({hypothesis_tokens}). "
                f"Skipping chunk {chunk_id} for case {case_id}."
            )
            return []
        
        # Tokenize the chunk
        tokens = self.tokenizer.encode(
            chunk_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).squeeze()
        
        total_tokens = len(tokens)
        
        # If chunk fits within budget, no windowing needed
        if total_tokens <= budget:
            return [ChunkWindow(
                case_id=case_id,
                chunk_id=chunk_id,
                window_id=0,
                window_text=chunk_text,
                token_start=0,
                token_end=total_tokens,
                token_count=total_tokens,
                was_truncated=False,
                doc_source=f"{case_id}_chunk_{chunk_id}",
            )]
        
        # Create overlapping windows
        windows = []
        window_id = 0
        stride = budget - overlap_tokens
        
        if stride <= 0:
            logger.warning(f"Stride ({stride}) is too small for chunk {chunk_id}.")
            stride = MIN_WINDOW_TOKENS
        
        start = 0
        while start < total_tokens and len(windows) < MAX_CHUNK_WINDOWS_PER_CHUNK:
            end = min(start + budget, total_tokens)
            
            window_tokens = tokens[start:end]
            window_text = self.tokenizer.decode(window_tokens, skip_special_tokens=True)
            
            was_truncated = end < total_tokens
            
            windows.append(ChunkWindow(
                case_id=case_id,
                chunk_id=chunk_id,
                window_id=window_id,
                window_text=window_text,
                token_start=start,
                token_end=end,
                token_count=len(window_tokens),
                was_truncated=was_truncated,
                doc_source=f"{case_id}_chunk_{chunk_id}",
            ))
            
            start += stride
            window_id += 1
        
        if start < total_tokens:
            logger.warning(
                f"Chunk {chunk_id} for case {case_id} has {total_tokens} tokens "
                f"but was limited to {MAX_CHUNK_WINDOWS_PER_CHUNK} windows."
            )
        
        return windows
    
    @torch.inference_mode()
    def predict(
        self,
        premises: list[str],
        hypotheses: list[str],
        batch_size: int = 8,
    ) -> list[NLIScore]:
        """
        Run NLI inference on premise-hypothesis pairs.
        
        Args:
            premises: List of premise texts (chunk windows or questions)
            hypotheses: List of hypothesis texts (claims)
            batch_size: Batch size for inference
            
        Returns:
            List of NLIScore objects
        """
        if len(premises) != len(hypotheses):
            raise ValueError("Premises and hypotheses must have the same length")
        
        all_scores = []
        
        for i in range(0, len(premises), batch_size):
            batch_premises = premises[i:i + batch_size]
            batch_hypotheses = hypotheses[i:i + batch_size]
            
            # Tokenize with truncation only on premise (first position)
            # Claims are never truncated
            inputs = self.tokenizer(
                batch_premises,
                batch_hypotheses,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Forward pass
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Get probabilities using dynamic label indices
            probs = torch.softmax(logits, dim=-1)
            
            for j, (premise, hypothesis) in enumerate(zip(batch_premises, batch_hypotheses)):
                p_entail = probs[j, self.entailment_idx].item()
                p_neutral = probs[j, self.neutral_idx].item()
                p_contrad = probs[j, self.contradiction_idx].item()
                
                all_scores.append(NLIScore(
                    case_id="",  # Will be filled by caller
                    claim_id=0,  # Will be filled by caller
                    claim_text="",  # Will be filled by caller
                    chunk_id=0,
                    window_id=0,
                    entailment_probability=p_entail,
                    neutral_probability=p_neutral,
                    contradiction_probability=p_contrad,
                    premise=premise,
                    hypothesis=hypothesis,
                ))
        
        return all_scores
    
    def close(self):
        """Clean up model resources."""
        del self.model
        del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_nli_cache(cache_path: Path) -> Optional[pd.DataFrame]:
    """Load cached NLI scores if available."""
    if cache_path.exists():
        logger.info(f"Loading cached NLI scores from {cache_path}")
        df = pd.read_csv(cache_path)
        return df
    return None


def save_nli_cache(scores: list[NLIScore], cache_path: Path) -> None:
    """Save NLI scores to cache."""
    records = []
    for score in scores:
        records.append({
            "case_id": score.case_id,
            "claim_id": score.claim_id,
            "claim_text": score.claim_text,
            "chunk_id": score.chunk_id,
            "window_id": score.window_id,
            "task_type": score.task_type,
            "premise": score.premise,
            "hypothesis": score.hypothesis,
            "entailment_probability": score.entailment_probability,
            "neutral_probability": score.neutral_probability,
            "contradiction_probability": score.contradiction_probability,
        })
    
    df = pd.DataFrame(records)
    df.to_csv(cache_path, index=False)
    logger.info(f"Cached {len(scores)} NLI scores to {cache_path}")


def batch_inference(
    model: NLIModel,
    samples: list,
    segments: dict,
    batch_size: int = 8,
    cache_path: Optional[Path] = None,
    verbose: bool = True,
    task_type: str = "faithfulness",
) -> pd.DataFrame:
    """
    Run NLI inference on a batch of samples.
    
    For faithfulness: premise = context window, hypothesis = claim
    For relevance: premise = question, hypothesis = claim
    
    Args:
        model: NLIModel instance
        samples: List of SampleData objects
        segments: Dict mapping case_id to AnswerSegments
        batch_size: Batch size for inference
        cache_path: Path to cache file
        verbose: Whether to log progress
        task_type: "faithfulness" or "relevance"
        
    Returns:
        DataFrame with all NLI scores
    """
    # Check cache
    if cache_path and cache_path.exists():
        return load_nli_cache(cache_path)
    
    all_scores = []
    
    if verbose:
        logger.info(f"Running {task_type} NLI inference on {len(samples)} samples")
    
    start_time = time.time()
    
    for sample in samples:
        seg = segments.get(sample.case_id)
        if seg is None:
            logger.warning(f"No segments found for case {sample.case_id}")
            continue
        
        # Determine premise based on task type
        if task_type == "faithfulness":
            # Faithfulness: premise = chunk window
            for claim in seg.claims:
                hyp_tokens = len(model.tokenizer.encode(
                    claim.text, add_special_tokens=False
                )) + 10
                
                for chunk in sample.chunks:
                    windows = model.create_chunk_windows(
                        chunk_text=chunk.chunk_text,
                        case_id=sample.case_id,
                        chunk_id=chunk.chunk_id,
                        hypothesis_tokens=hyp_tokens,
                    )
                    
                    if not windows:
                        continue
                    
                    premises = [w.window_text for w in windows]
                    hypotheses = [claim.text] * len(windows)
                    
                    scores = model.predict(premises, hypotheses, batch_size=batch_size)
                    
                    for score, window in zip(scores, windows):
                        score.case_id = sample.case_id
                        score.claim_id = claim.claim_id
                        score.claim_text = claim.text
                        score.chunk_id = chunk.chunk_id
                        score.window_id = window.window_id
                        score.task_type = task_type
                        score.premise = window.window_text
                        score.hypothesis = claim.text
                    
                    all_scores.extend(scores)
        
        else:
            # Relevance: premise = question, hypothesis = claim
            premises = [sample.question] * len(seg.claims)
            hypotheses = [claim.text for claim in seg.claims]
            
            scores = model.predict(premises, hypotheses, batch_size=batch_size)
            
            for score, claim in zip(scores, seg.claims):
                score.case_id = sample.case_id
                score.claim_id = claim.claim_id
                score.claim_text = claim.text
                score.chunk_id = 0  # No chunks for relevance
                score.window_id = 0
                score.task_type = task_type
                score.premise = sample.question
                score.hypothesis = claim.text
            
            all_scores.extend(scores)
    
    if verbose:
        elapsed = time.time() - start_time
        logger.info(f"Completed {task_type} NLI: {len(all_scores)} scores in {elapsed:.1f}s")
    
    # Convert to DataFrame
    records = []
    for score in all_scores:
        records.append({
            "case_id": score.case_id,
            "claim_id": score.claim_id,
            "claim_text": score.claim_text,
            "chunk_id": score.chunk_id,
            "window_id": score.window_id,
            "task_type": score.task_type,
            "premise": score.premise,
            "hypothesis": score.hypothesis,
            "entailment_probability": score.entailment_probability,
            "neutral_probability": score.neutral_probability,
            "contradiction_probability": score.contradiction_probability,
        })
    
    df = pd.DataFrame(records)
    
    # Save cache
    if cache_path and all_scores:
        save_nli_cache(all_scores, cache_path)
    
    return df
