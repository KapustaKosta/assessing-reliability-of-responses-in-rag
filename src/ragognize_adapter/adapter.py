"""
Data adapter for RAGognize dataset.

Provides unified interface for downstream tasks:
- Token-level hallucination detection
- NLI-based faithfulness detection
- Reliability fusion
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Literal

from datasets import DatasetDict, Dataset

from .constants import SOURCE_MODELS, GOLDEN_ANSWER_MODEL, ALL_MODELS, DEFAULT_SEED
from .loader import load_ragognize_dataset
from .inspection import inspect_hallucinations_structure

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class HallucinationSpan:
    """
    Represents a hallucinated span in the model response.
    
    Attributes:
        text: The hallucinated text span.
        start: Character start index (inclusive).
        end: Character end index (exclusive).
        valid: Whether this is a confirmed hallucination (vs marked but unverified).
    """
    text: str
    start: int
    end: int
    valid: bool = True


@dataclass
class UnifiedSample:
    """
    Unified sample format for faithfulness detection.
    
    Each sample represents ONE question-model pair.
    
    Fields:
        case_id: Globally unique identifier (stable across runs).
        user_prompt_index: Original prompt index from RAGognize.
        question: The user's question.
        answer: The model's response.
        chunks: List of retrieved document chunks (text only).
        chunk_titles: Titles of the documents (may be empty strings).
        golden_answer: Reference golden answer (not used as a model).
        hallucination_spans: List of hallucinated character spans.
        has_hallucination: Binary label (1=has hallucination, 0=clean).
        faithfulness_label: Binary label (1=faithful, 0=unfaithful).
        answerable: Whether the question can be answered from context.
        source_model: Which model's response this is.
        information_type: Type of information in the question.
        category: Category of the question.
        tags: List of tags.
        information_date: Date information in the question.
    """
    case_id: str
    user_prompt_index: int
    question: str
    answer: str
    chunks: list[str]
    chunk_titles: list[str]
    golden_answer: str
    hallucination_spans: list[HallucinationSpan]
    has_hallucination: int
    faithfulness_label: int
    answerable: bool
    source_model: str
    information_type: str
    category: str
    tags: list[str]
    information_date: str
    
    def __post_init__(self):
        """Ensure lists are lists."""
        if self.chunks is None:
            self.chunks = []
        if self.chunk_titles is None:
            self.chunk_titles = []
        if self.tags is None:
            self.tags = []
        if self.hallucination_spans is None:
            self.hallucination_spans = []
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "case_id": self.case_id,
            "user_prompt_index": self.user_prompt_index,
            "question": self.question,
            "answer": self.answer,
            "chunks": self.chunks,
            "chunk_titles": self.chunk_titles,
            "golden_answer": self.golden_answer,
            "hallucination_spans": [
                {
                    "text": s.text,
                    "start": s.start,
                    "end": s.end,
                    "valid": s.valid,
                }
                for s in self.hallucination_spans
            ],
            "has_hallucination": self.has_hallucination,
            "faithfulness_label": self.faithfulness_label,
            "answerable": self.answerable,
            "source_model": self.source_model,
            "information_type": self.information_type,
            "category": self.category,
            "tags": self.tags,
            "information_date": self.information_date,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> UnifiedSample:
        """Create from dictionary."""
        spans = [
            HallucinationSpan(
                text=s["text"],
                start=s["start"],
                end=s["end"],
                valid=s.get("valid", True),
            )
            for s in d.get("hallucination_spans", [])
        ]
        
        return cls(
            case_id=d["case_id"],
            user_prompt_index=d["user_prompt_index"],
            question=d["question"],
            answer=d["answer"],
            chunks=d.get("chunks", []),
            chunk_titles=d.get("chunk_titles", []),
            golden_answer=d.get("golden_answer", ""),
            hallucination_spans=spans,
            has_hallucination=d.get("has_hallucination", 0),
            faithfulness_label=d.get("faithfulness_label", 1),
            answerable=d.get("answerable", True),
            source_model=d["source_model"],
            information_type=d.get("information_type", ""),
            category=d.get("category", ""),
            tags=d.get("tags", []),
            information_date=d.get("information_date", ""),
        )


# =============================================================================
# Adapter
# =============================================================================

class RAGognizeAdapter:
    """
    Adapter to transform RAGognize samples to UnifiedSample format.
    """
    
    def __init__(
        self,
        models: Optional[list[str]] = None,
        include_golden: bool = False,
    ):
        """
        Initialize the adapter.
        
        Args:
            models: List of model names to include. Defaults to SOURCE_MODELS.
            include_golden: Whether to include golden_answer (not recommended).
        """
        self.models = models if models is not None else list(SOURCE_MODELS)
        self.include_golden = include_golden
        
        # Validate models
        for model in self.models:
            if model not in ALL_MODELS:
                logger.warning(f"Unknown model: {model}")
    
    def _generate_case_id(
        self,
        user_prompt_index: int,
        model_name: str,
        row_index: int = 0,
    ) -> str:
        """
        Generate a stable, unique case_id.
        
        Uses MD5 hash of the combination to ensure determinism.
        When user_prompt_index may have duplicates, row_index is used as disambiguator.
        """
        raw = f"{user_prompt_index}_{model_name}_{row_index}"
        digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]
        return f"case_{digest}"
    
    def _parse_documents(self, documents: list) -> tuple[list[str], list[str]]:
        """
        Extract text and titles from documents.
        
        Returns:
            Tuple of (texts, titles).
        """
        texts = []
        titles = []
        
        for doc in documents:
            if isinstance(doc, dict):
                text = doc.get("text", "")
                title = doc.get("title", "")
            elif isinstance(doc, str):
                text = doc
                title = ""
            else:
                text = ""
                title = ""
            
            if text:
                texts.append(text)
                titles.append(title if title else "")
        
        return texts, titles
    
    def _parse_hallucinations(
        self,
        hallucination_data,
        answer: str,
    ) -> list[HallucinationSpan]:
        """
        Parse hallucination data into HallucinationSpan objects.
        
        Args:
            hallucination_data: Raw hallucination data from the dataset.
            answer: The model's answer (for validation).
        
        Returns:
            List of HallucinationSpan objects.
        """
        spans = []
        
        if hallucination_data is None:
            return spans
        
        if not isinstance(hallucination_data, list):
            logger.warning(f"Unexpected hallucination type: {type(hallucination_data)}")
            return spans
        
        for item in hallucination_data:
            if not isinstance(item, dict):
                continue
            
            text = item.get("text")
            start = item.get("start")
            end = item.get("end")
            valid = item.get("valid", True)  # Default to True if not specified
            
            # Skip items without coordinates
            if start is None or end is None:
                continue
            
            # Validate coordinates
            if not isinstance(start, int) or not isinstance(end, int):
                continue
            
            if start < 0 or end <= start:
                continue
            
            if end > len(answer):
                # Truncate to answer length
                end = len(answer)
                if start >= end:
                    continue
            
            # Use answer text if text not provided
            if text is None:
                text = answer[start:end]
            else:
                # Validate text match
                expected = answer[start:end]
                if expected != text and expected.strip() != text.strip():
                    logger.debug(
                        f"Text mismatch at [{start}:{end}]: "
                        f"expected '{expected[:30]}...', got '{text[:30]}...'"
                    )
            
            spans.append(HallucinationSpan(
                text=text,
                start=start,
                end=end,
                valid=valid,
            ))
        
        return spans
    
    def _determine_faithfulness(
        self,
        hallucination_spans: list[HallucinationSpan],
        completely_hallucinated: bool = False,
    ) -> tuple[int, int]:
        """
        Determine faithfulness labels from hallucination spans.
        
        RAGognize semantics:
        - completely_hallucinated=True: Entire answer is hallucinated (no spans)
        - hallucination_spans: Specific hallucinated character regions
        
        Only considers spans where valid=True as true hallucinations.
        completely_hallucinated=True is treated as having hallucinations.
        
        Returns:
            Tuple of (has_hallucination, faithfulness_label).
            - has_hallucination: 1 if any hallucination exists, else 0
            - faithfulness_label: 1 if faithful (no hallucinations), else 0
        """
        valid_spans = [s for s in hallucination_spans if s.valid]
        
        # Either has valid spans or entire answer is hallucinated
        has_hallucination = 1 if (len(valid_spans) > 0 or completely_hallucinated) else 0
        faithfulness_label = 1 if has_hallucination == 0 else 0
        
        return has_hallucination, faithfulness_label
    
    def transform_sample(
        self,
        sample: dict,
        source_split: str,
        row_index: int = 0,
    ) -> list[UnifiedSample]:
        """
        Transform a single RAGognize sample to UnifiedSample(s).
        
        One RAGognize sample contains responses from multiple models.
        This expands it into one UnifiedSample per model.
        
        Args:
            sample: Raw sample from RAGognize.
            source_split: Source split name ('train' or 'test').
        
        Returns:
            List of UnifiedSample objects.
        """
        samples = []
        
        # Extract common fields
        user_prompt_index = sample.get("user_prompt_index", 0)
        question = sample.get("user_prompt", "")
        answerable = sample.get("answerable", True)
        information_type = sample.get("information_type", "")
        category = sample.get("category", "")
        tags = sample.get("tags", [])
        information_date = sample.get("information_date", "")
        
        # Parse documents
        documents = sample.get("documents", [])
        chunks, chunk_titles = self._parse_documents(documents)
        
        # Get golden answer
        responses = sample.get("responses", {})
        golden_answer = ""
        if GOLDEN_ANSWER_MODEL in responses:
            golden_raw = responses[GOLDEN_ANSWER_MODEL]
            if isinstance(golden_raw, str):
                golden_answer = golden_raw
            elif isinstance(golden_raw, dict):
                golden_answer = golden_raw.get("text", "")
        
        # Process each model response
        for model_name in self.models:
            if model_name not in responses:
                logger.debug(f"Model {model_name} not in responses for prompt {user_prompt_index}")
                continue
            
            response = responses[model_name]
            if not isinstance(response, dict):
                continue
            
            answer = response.get("text", "")
            if not answer:
                continue
            
            # Parse hallucinations
            hallucination_data = response.get("hallucinations", [])
            hallucination_spans = self._parse_hallucinations(
                hallucination_data, answer
            )
            
            # Get completely_hallucinated from annotations
            annotations = response.get("details", {}).get("annotations", {})
            original_output = annotations.get("original_output", {})
            completely_hallucinated = original_output.get("completely_hallucinated", False)
            
            # Determine labels
            has_hallucination, faithfulness_label = self._determine_faithfulness(
                hallucination_spans,
                completely_hallucinated=completely_hallucinated,
            )
            
            # Generate case_id (include row_index to handle duplicate prompt indices)
            case_id = self._generate_case_id(user_prompt_index, model_name, row_index)
            
            samples.append(UnifiedSample(
                case_id=case_id,
                user_prompt_index=user_prompt_index,
                question=question,
                answer=answer,
                chunks=chunks,
                chunk_titles=chunk_titles,
                golden_answer=golden_answer,
                hallucination_spans=hallucination_spans,
                has_hallucination=has_hallucination,
                faithfulness_label=faithfulness_label,
                answerable=answerable,
                source_model=model_name,
                information_type=information_type,
                category=category,
                tags=tags if isinstance(tags, list) else [],
                information_date=information_date,
            ))
        
        return samples
    
    def transform_dataset(self, dataset: DatasetDict) -> dict[str, list[UnifiedSample]]:
        """
        Transform a DatasetDict to UnifiedSamples.
        
        Args:
            dataset: DatasetDict with 'train' and 'test' splits.
        
        Returns:
            Dictionary mapping split names to lists of UnifiedSamples.
        """
        result = {}
        
        for split_name, split_data in dataset.items():
            logger.info(f"Transforming split: {split_name} ({len(split_data)} samples)")
            samples = []
            
            for sample in split_data:
                transformed = self.transform_sample(sample, split_name)
                samples.extend(transformed)
            
            logger.info(f"  -> {len(samples)} UnifiedSamples")
            result[split_name] = samples
        
        return result


def create_unified_dataset(
    data_dir: Optional[Path] = None,
) -> dict[str, list[UnifiedSample]]:
    """
    Convenience function to load and transform RAGognize dataset.
    
    Args:
        data_dir: Path to directory containing Parquet files.
    
    Returns:
        Dictionary with UnifiedSamples for each split.
    """
    dataset = load_ragognize_dataset(data_dir=data_dir)
    adapter = RAGognizeAdapter()
    return adapter.transform_dataset(dataset)
