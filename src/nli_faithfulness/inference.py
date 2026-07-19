"""
NLI inference module for Faithfulness detection.

Handles model loading, tokenizer-aware chunk windowing, batch inference,
and caching of results.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, asdict
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
    NLI_ENTAILMENT,
    NLI_NEUTRAL,
    NLI_CONTRADICTION,
)


logger = logging.getLogger(__name__)


@dataclass
class ChunkWindow:
    """Represents a window within a chunk for NLI inference."""
    case_id: str
    chunk_id: int
    window_id: int
    window_text: str
    token_count: int
    was_truncated: bool = False


@dataclass
class NLIScore:
    """Single NLI inference result."""
    case_id: str
    sentence_id: int
    chunk_id: int
    window_id: int
    premise: str  # chunk window text
    hypothesis: str  # sentence text
    p_entailment: float
    p_neutral: float
    p_contradiction: float
    predicted_label: str  # entailment, neutral, or contradiction
    
    def __post_init__(self):
        # Validate probabilities sum to ~1
        total = self.p_entailment + self.p_neutral + self.p_contradiction
        if abs(total - 1.0) > 0.01:
            logger.warning(
                f"Probabilities sum to {total:.4f}, expected ~1.0 for case {self.case_id}"
            )


class NLIModel:
    """Wrapper for NLI model with dynamic label mapping."""
    
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
        """Map our semantic NLI labels to model's actual labels."""
        # Our target labels
        self.entailment_idx = None
        self.neutral_idx = None
        self.contradiction_idx = None
        
        # Try to find labels based on semantic meaning
        # Model labels may be: entailment, contradiction, neutral (any order)
        for idx, label in self.id2label.items():
            label_lower = label.lower()
            if "entail" in label_lower or "support" in label_lower:
                self.entailment_idx = idx
            elif "contradict" in label_lower or "refute" in label_lower:
                self.contradiction_idx = idx
            elif "neutral" in label_lower:
                self.neutral_idx = idx
        
        # Fallback: assume label order is [entailment, neutral, contradiction]
        # which is common for many NLI models
        if self.entailment_idx is None:
            logger.warning(
                f"Could not auto-detect NLI label mapping. "
                f"Found labels: {self.id2label}. Assuming order [entailment, neutral, contradiction]."
            )
            if len(self.id2label) >= 3:
                self.entailment_idx = 0
                self.neutral_idx = 1
                self.contradiction_idx = 2
            else:
                raise ValueError(f"Model has unexpected number of labels: {len(self.id2label)}")
        
        logger.info(
            f"NLI label mapping: "
            f"entailment={self.entailment_idx} ({self.id2label[self.entailment_idx]}), "
            f"neutral={self.neutral_idx} ({self.id2label[self.neutral_idx]}), "
            f"contradiction={self.contradiction_idx} ({self.id2label[self.contradiction_idx]})"
        )
    
    def _log_model_info(self) -> None:
        """Log model configuration info."""
        logger.info(f"Model architecture: {self.model.config.model_type}")
        logger.info(f"Number of labels: {self.model.config.num_labels}")
        logger.info(f"Label mapping: {self.id2label}")
        
        # Get max length
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
        
        Args:
            chunk_text: The full chunk text
            case_id: Sample identifier
            chunk_id: Which chunk (1-8)
            hypothesis_tokens: Number of tokens reserved for the hypothesis
            overlap_tokens: Number of overlapping tokens between windows
            
        Returns:
            List of ChunkWindow objects
        """
        # Calculate available budget for premise (chunk)
        # Subtract 3 for [CLS], [SEP], [SEP] tokens in most tokenizers
        budget = self.max_length - hypothesis_tokens - 3
        
        if budget <= 0:
            logger.warning(
                f"Chunk budget ({budget}) is too small for hypothesis ({hypothesis_tokens}). "
                f"Skipping chunk {chunk_id} for case {case_id}."
            )
            return []
        
        # Tokenize the chunk to get token-level boundaries
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
                token_count=total_tokens,
                was_truncated=False,
            )]
        
        # Need to create windows
        windows = []
        window_id = 0
        stride = budget - overlap_tokens
        
        if stride <= 0:
            logger.warning(
                f"Stride ({stride}) is too small for chunk {chunk_id}. "
                f"Using minimum window size."
            )
            stride = MIN_WINDOW_TOKENS
        
        start = 0
        while start < total_tokens and len(windows) < MAX_CHUNK_WINDOWS_PER_CHUNK:
            end = min(start + budget, total_tokens)
            
            # Decode this window
            window_tokens = tokens[start:end]
            window_text = self.tokenizer.decode(window_tokens, skip_special_tokens=True)
            
            # Check if we truncated the end
            was_truncated = end < total_tokens
            
            # Verify token count
            verify_tokens = self.tokenizer.encode(
                window_text,
                add_special_tokens=False,
                return_tensors="pt",
            ).squeeze()
            
            windows.append(ChunkWindow(
                case_id=case_id,
                chunk_id=chunk_id,
                window_id=window_id,
                window_text=window_text,
                token_count=len(verify_tokens),
                was_truncated=was_truncated,
            ))
            
            start += stride
            window_id += 1
        
        # Log if we hit the limit
        if start < total_tokens:
            logger.warning(
                f"Chunk {chunk_id} for case {case_id} has {total_tokens} tokens "
                f"but was limited to {MAX_CHUNK_WINDOWS_PER_CHUNK} windows. "
                f"Last {total_tokens - start} tokens not covered."
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
            premises: List of premise texts (chunk windows)
            hypotheses: List of hypothesis texts (sentence units)
            batch_size: Batch size for inference
            
        Returns:
            List of NLIScore objects
        """
        if len(premises) != len(hypotheses):
            raise ValueError(" premises and hypotheses must have the same length")
        
        all_scores = []
        
        for i in range(0, len(premises), batch_size):
            batch_premises = premises[i:i + batch_size]
            batch_hypotheses = hypotheses[i:i + batch_size]
            
            # Tokenize
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
            
            # Get probabilities
            probs = torch.softmax(logits, dim=-1)
            
            # Extract scores for each class
            for j, (premise, hypothesis) in enumerate(zip(batch_premises, batch_hypotheses)):
                p_entail = probs[j, self.entailment_idx].item()
                p_neutral = probs[j, self.neutral_idx].item()
                p_contrad = probs[j, self.contradiction_idx].item()
                
                # Determine predicted label
                scores = {
                    NLI_ENTAILMENT: p_entail,
                    NLI_NEUTRAL: p_neutral,
                    NLI_CONTRADICTION: p_contrad,
                }
                pred_label = max(scores, key=scores.get)
                
                all_scores.append(NLIScore(
                    case_id="",  # Will be filled by caller
                    sentence_id=0,  # Will be filled by caller
                    chunk_id=0,  # Will be filled by caller
                    window_id=0,  # Will be filled by caller
                    premise=premise,
                    hypothesis=hypothesis,
                    p_entailment=p_entail,
                    p_neutral=p_neutral,
                    p_contradiction=p_contrad,
                    predicted_label=pred_label,
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
            "sentence_id": score.sentence_id,
            "chunk_id": score.chunk_id,
            "window_id": score.window_id,
            "premise": score.premise,
            "hypothesis": score.hypothesis,
            "p_entailment": score.p_entailment,
            "p_neutral": score.p_neutral,
            "p_contradiction": score.p_contradiction,
            "predicted_label": score.predicted_label,
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
) -> pd.DataFrame:
    """
    Run NLI inference on a batch of samples.
    
    Args:
        model: NLIModel instance
        samples: List of SampleData objects
        segments: Dict mapping case_id to AnswerSegments
        batch_size: Batch size for inference
        cache_path: Path to cache file
        verbose: Whether to log progress
        
    Returns:
        DataFrame with all NLI scores
    """
    # Check cache
    if cache_path and cache_path.exists():
        return load_nli_cache(cache_path)
    
    all_scores = []
    total_pairs = 0
    
    # Count total pairs for progress
    for sample in samples:
        seg = segments.get(sample.case_id)
        if seg is None:
            continue
        total_pairs += len(seg) * sample.chunk_count
    
    if verbose:
        logger.info(f"Running NLI inference on {len(samples)} samples ({total_pairs} pairs)")
    
    processed = 0
    start_time = time.time()
    
    for sample in samples:
        seg = segments.get(sample.case_id)
        if seg is None:
            logger.warning(f"No segments found for case {sample.case_id}")
            continue
        
        # For each sentence in the answer
        for sentence_unit in seg.units:
            # Estimate hypothesis tokens (add buffer)
            hyp_tokens = len(model.tokenizer.encode(
                sentence_unit.text,
                add_special_tokens=False,
            )) + 10
            
            # For each chunk, create windows and run inference
            for chunk in sample.chunks:
                windows = model.create_chunk_windows(
                    chunk_text=chunk.chunk_text,
                    case_id=sample.case_id,
                    chunk_id=chunk.chunk_id,
                    hypothesis_tokens=hyp_tokens,
                )
                
                if not windows:
                    continue
                
                # Prepare batch inputs
                premises = [w.window_text for w in windows]
                hypotheses = [sentence_unit.text] * len(windows)
                
                # Check for truncation
                truncated_count = sum(1 for w in windows if w.was_truncated)
                if truncated_count > 0:
                    logger.debug(
                        f"Case {sample.case_id}, chunk {chunk.chunk_id}, "
                        f"sentence {sentence_unit.sentence_id}: "
                        f"{truncated_count}/{len(windows)} windows were truncated"
                    )
                
                # Run inference
                scores = model.predict(premises, hypotheses, batch_size=batch_size)
                
                # Fill metadata
                for score, window in zip(scores, windows):
                    score.case_id = sample.case_id
                    score.sentence_id = sentence_unit.sentence_id
                    score.chunk_id = chunk.chunk_id
                    score.window_id = window.window_id
                
                all_scores.extend(scores)
        
        processed += 1
        if verbose and processed % 50 == 0:
            elapsed = time.time() - start_time
            rate = processed / elapsed
            remaining = (len(samples) - processed) / rate if rate > 0 else 0
            logger.info(
                f"Processed {processed}/{len(samples)} samples "
                f"({100 * processed / len(samples):.1f}%), "
                f"rate: {rate:.1f} samples/s, "
                f"ETA: {remaining:.0f}s"
            )
    
    if verbose:
        logger.info(f"Completed NLI inference: {len(all_scores)} scores in {time.time() - start_time:.1f}s")
    
    # Convert to DataFrame
    records = []
    for score in all_scores:
        records.append({
            "case_id": score.case_id,
            "sentence_id": score.sentence_id,
            "chunk_id": score.chunk_id,
            "window_id": score.window_id,
            "premise": score.premise,
            "hypothesis": score.hypothesis,
            "p_entailment": score.p_entailment,
            "p_neutral": score.p_neutral,
            "p_contradiction": score.p_contradiction,
            "predicted_label": score.predicted_label,
        })
    
    df = pd.DataFrame(records)
    
    # Validate probability sums
    df["prob_sum"] = df["p_entailment"] + df["p_neutral"] + df["p_contradiction"]
    invalid = df[abs(df["prob_sum"] - 1.0) > 0.01]
    if len(invalid) > 0:
        logger.warning(f"Found {len(invalid)} rows with probabilities not summing to ~1.0")
    
    df = df.drop(columns=["prob_sum"])
    
    # Save cache
    if cache_path:
        save_nli_cache(all_scores, cache_path)
    
    return df
