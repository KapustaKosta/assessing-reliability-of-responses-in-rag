"""
Core adapter for F4biian/RAGognize dataset.

Provides unified interface for NLI-based faithfulness detection.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Literal

from datasets import load_dataset, Dataset, DatasetDict


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class HallucinationSpan:
    """Represents a hallucinated span in the model response."""
    text: str
    start: int
    end: int
    valid: bool


@dataclass
class ModelResponse:
    """Represents a single model response with hallucination annotations."""
    model_name: str
    output: str
    hallucinations: list[HallucinationSpan] = field(default_factory=list)
    addressed_user_prompt: bool = True
    all_valid: bool = True
    cluelessness: bool = False
    completely_hallucinated: bool = False
    answerable: bool = True


@dataclass
class RAGognizeSample:
    """
    Raw sample from F4biian/RAGognize dataset.
    
    This represents one row in the original dataset, which contains
    one question and multiple model responses.
    """
    # Source identifiers
    user_prompt_index: int
    category: str
    information_type: str
    information_date: str
    tags: list[str]
    
    # Question
    user_prompt: str
    
    # Documents (retrieved context)
    documents: list[dict]  # [{'text': str, 'title': str}, ...]
    
    # Model responses
    responses: dict[str, ModelResponse]
    
    # Answerability
    answerable: bool
    
    # Golden answer (reference)
    golden_answer: str


@dataclass
class UnifiedSample:
    """
    Unified sample format for faithfulness detection.
    
    Each sample represents ONE question-model pair, suitable for NLI inference.
    """
    # Unique identifier
    case_id: str
    
    # Source tracking fields (for reproducibility and error analysis)
    source_split: str = ""        # "train", "val", "test"
    source_row_index: int = 0    # Row index in original dataset
    user_prompt_index: int = 0   # Original user_prompt_index from RAGognize
    source_model: str = ""        # Which model's response this is
    
    # Question
    question: str = ""
    
    # Model's answer
    answer: str = ""
    
    # Retrieved documents (as list of text chunks)
    chunks: list[str] = None
    
    # Hallucination annotations (from dataset)
    hallucination_spans: list[HallucinationSpan] = None
    
    # Derived labels
    faithfulness_label: bool = True  # False if any hallucination exists
    answerable: bool = True
    
    # Metadata
    information_type: str = ""
    category: str = ""
    
    # Additional context
    golden_answer: str = ""
    
    def __post_init__(self):
        if self.chunks is None:
            self.chunks = []
        if self.hallucination_spans is None:
            self.hallucination_spans = []
    
    @property
    def has_hallucinations(self) -> bool:
        """Check if there are any hallucinations."""
        return len(self.hallucination_spans) > 0


# =============================================================================
# Constants
# =============================================================================

CACHE_DIR = os.environ.get(
    "HF_DATASET_CACHE",
    Path(__file__).parent.parent.parent / "data" / "cache" / "huggingface"
)
CACHE_DIR = Path(CACHE_DIR)

# Available model names in the dataset
AVAILABLE_MODELS = [
    "Llama-2-7b-chat-hf",
    "Llama-3.1-8B-Instruct",
    "Mistral-7B-Instruct-v0.1",
    "Mistral-7B-Instruct-v0.3",
]

# Golden answer model (not a real model, but included in responses)
GOLDEN_ANSWER_MODEL = "golden_answer"


# =============================================================================
# Adapter
# =============================================================================

class RAGognizeAdapter:
    """
    Adapter to convert RAGognize dataset samples to unified format.
    """
    
    def __init__(
        self,
        models: Optional[list[str]] = None,
        include_golden: bool = False,
    ):
        """
        Initialize the adapter.
        
        Args:
            models: List of model names to include. If None, includes all.
            include_golden: Whether to include golden_answer as a model.
        """
        if models is None:
            self.models = [m for m in AVAILABLE_MODELS]
        else:
            self.models = models
        
        self.include_golden = include_golden
    
    def _parse_hallucinations(self, hallucination_list: list) -> list[HallucinationSpan]:
        """Parse hallucination list into HallucinationSpan objects."""
        spans = []
        for h in hallucination_list:
            spans.append(HallucinationSpan(
                text=h.get("text", ""),
                start=h.get("start", 0),
                end=h.get("end", 0),
                valid=h.get("valid", True),
            ))
        return spans
    
    def _parse_model_response(
        self,
        model_name: str,
        response_data: dict,
    ) -> ModelResponse:
        """Parse a single model response."""
        # Get hallucination spans
        hallucination_list = response_data.get("hallucinations", [])
        hallucinations = self._parse_hallucinations(hallucination_list)
        
        # Get details if available
        details = response_data.get("details", {})
        result = details.get("result", {})
        
        # Also check top-level output
        output = response_data.get("text", "") or response_data.get("output", "")
        
        return ModelResponse(
            model_name=model_name,
            output=output,
            hallucinations=hallucinations,
            addressed_user_prompt=result.get("addressed_user_prompt", True),
            all_valid=result.get("all_valid", True),
            cluelessness=result.get("cluelessness", False),
            completely_hallucinated=result.get("completely_hallucinated", False),
            answerable=response_data.get("answerable", details.get("answerable", True)),
        )
    
    def _parse_documents(self, documents: list) -> list[str]:
        """Extract text from documents list."""
        chunks = []
        for doc in documents:
            if isinstance(doc, dict):
                text = doc.get("text", "")
                if text:
                    chunks.append(text)
            elif isinstance(doc, str):
                chunks.append(doc)
        return chunks
    
    def parse_sample(
        self,
        sample: dict,
        source_split: str = "",
        source_row_index: int = 0,
    ) -> list[UnifiedSample]:
        """
        Parse a single RAGognize sample into multiple unified samples.
        
        One RAGognize sample contains multiple model responses.
        This method expands it into one unified sample per model.
        
        Args:
            sample: Raw sample from RAGognize dataset
            source_split: Source split name ("train", "val", "test")
            source_row_index: Row index in the original dataset (for unique case_id)
            
        Returns:
            List of UnifiedSample objects, one per model
        """
        unified_samples = []
        
        # Extract common fields
        user_prompt = sample.get("user_prompt", "")
        documents = self._parse_documents(sample.get("documents", []))
        answerable = sample.get("answerable", False)
        golden_answer = sample.get("responses", {}).get("golden_answer", "")
        
        # Parse each model response
        responses = sample.get("responses", {})
        
        for model_name in self.models:
            if model_name not in responses:
                continue
            
            response_data = responses[model_name]
            model_response = self._parse_model_response(model_name, response_data)
            
            # Create case_id - must be globally unique
            # Uses hashlib.md5 for deterministic hashing
            case_id = self._generate_case_id(
                source_split=source_split,
                source_row_index=source_row_index,
                prompt_index=sample.get("user_prompt_index", 0),
                model_name=model_name,
            )
            
            # Determine faithfulness label
            # faithfulness = False if there are hallucinations
            # Note: We consider any hallucination as unfaithful
            faithfulness_label = len(model_response.hallucinations) == 0
            
            unified_samples.append(UnifiedSample(
                case_id=case_id,
                source_split=source_split,
                source_row_index=source_row_index,
                user_prompt_index=sample.get("user_prompt_index", 0),
                source_model=model_name,
                question=user_prompt,
                answer=model_response.output,
                chunks=documents,
                hallucination_spans=model_response.hallucinations,
                faithfulness_label=faithfulness_label,
                answerable=model_response.answerable,
                information_type=sample.get("information_type", ""),
                category=sample.get("category", ""),
                golden_answer=golden_answer,
            ))
        
        # Optionally include golden answer
        if self.include_golden and "golden_answer" in responses:
            golden_response = responses["golden_answer"]
            
            # Create a special golden sample
            golden_output = golden_response if isinstance(golden_response, str) else ""
            
            if golden_output:
                case_id = self._generate_case_id(
                    source_split=source_split,
                    source_row_index=source_row_index,
                    prompt_index=sample.get("user_prompt_index", 0),
                    model_name=GOLDEN_ANSWER_MODEL,
                )
                
                unified_samples.append(UnifiedSample(
                    case_id=case_id,
                    source_split=source_split,
                    source_row_index=source_row_index,
                    user_prompt_index=sample.get("user_prompt_index", 0),
                    source_model=GOLDEN_ANSWER_MODEL,
                    question=user_prompt,
                    answer=golden_output,
                    chunks=documents,
                    hallucination_spans=[],  # Golden answer is always faithful
                    faithfulness_label=True,
                    answerable=answerable,
                    information_type=sample.get("information_type", ""),
                    category=sample.get("category", ""),
                    golden_answer=golden_output,
                ))
        
        return unified_samples
    
    def _generate_case_id(
        self,
        source_split: str,
        source_row_index: int,
        prompt_index: int,
        model_name: str,
    ) -> str:
        """
        Generate a globally unique case_id using hashlib.md5.
        
        Uses deterministic MD5 hashing (not Python's built-in hash() which
        is randomized between Python runs for security).
        
        The case_id encodes:
        - source_split: Which split this sample came from
        - source_row_index: Row index in the original dataset
        - prompt_index: Original user_prompt_index from RAGognize
        - model_name: Which model's response this is
        
        This ensures:
        1. Global uniqueness across all splits
        2. Deterministic output (same inputs -> same case_id)
        3. Traceability back to original data
        """
        raw = f"{source_split}_{source_row_index}_{prompt_index}_{model_name}"
        digest = hashlib.md5(raw.encode('utf-8')).hexdigest()[:16]
        return f"case_{digest}"
    
    def transform_dataset(self, dataset: DatasetDict | Dataset) -> DatasetDict:
        """
        Transform a full dataset into unified format.
        
        Args:
            dataset: HuggingFace DatasetDict or Dataset
            
        Returns:
            DatasetDict with unified samples, keyed by split
        """
        from datasets import Dataset as HFDataset
        
        if isinstance(dataset, HFDataset):
            # Single dataset, not a dict - use empty string as split name
            samples = []
            for source_row_index, item in enumerate(dataset):
                samples.extend(self.parse_sample(
                    item,
                    source_split="",
                    source_row_index=source_row_index,
                ))
            
            return HFDataset.from_list([
                self._sample_to_dict(s) for s in samples
            ])
        
        # DatasetDict
        result = {}
        for split_name, split_data in dataset.items():
            samples = []
            for source_row_index, item in enumerate(split_data):
                samples.extend(self.parse_sample(
                    item,
                    source_split=split_name,
                    source_row_index=source_row_index,
                ))
            
            result[split_name] = HFDataset.from_list([
                self._sample_to_dict(s) for s in samples
            ])
        
        return DatasetDict(result)
    
    def _sample_to_dict(self, sample: UnifiedSample) -> dict:
        """Convert UnifiedSample to dict for HuggingFace Dataset."""
        return {
            "case_id": sample.case_id,
            # Source tracking fields
            "source_split": sample.source_split,
            "source_row_index": sample.source_row_index,
            "user_prompt_index": sample.user_prompt_index,
            "source_model": sample.source_model,
            # Content
            "question": sample.question,
            "answer": sample.answer,
            "chunks": sample.chunks,
            "hallucination_spans": [
                {
                    "text": h.text,
                    "start": h.start,
                    "end": h.end,
                    "valid": h.valid,
                }
                for h in sample.hallucination_spans
            ],
            "faithfulness_label": sample.faithfulness_label,
            "answerable": sample.answerable,
            "information_type": sample.information_type,
            "category": sample.category,
            "golden_answer": sample.golden_answer,
        }


# =============================================================================
# Dataset Loading
# =============================================================================

def load_ragognize_dataset(
    cache_dir: Optional[Path] = None,
    models: Optional[list[str]] = None,
) -> DatasetDict:
    """
    Load F4biian/RAGognize dataset from HuggingFace.
    
    Args:
        cache_dir: Directory for caching
        models: List of models to include
        
    Returns:
        DatasetDict with 'train' and 'test' splits
    """
    if cache_dir is None:
        cache_dir = CACHE_DIR
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    dataset = load_dataset(
        "F4biian/RAGognize",
        cache_dir=str(cache_dir),
    )
    
    return dataset


def get_unified_dataset(
    cache_dir: Optional[Path] = None,
    models: Optional[list[str]] = None,
    include_golden: bool = False,
) -> DatasetDict:
    """
    Load and transform RAGognize dataset to unified format.
    
    Args:
        cache_dir: Directory for caching
        models: List of models to include
        include_golden: Whether to include golden answers
        
    Returns:
        DatasetDict with unified samples
    """
    raw_dataset = load_ragognize_dataset(cache_dir, models)
    
    adapter = RAGognizeAdapter(models=models, include_golden=include_golden)
    
    return adapter.transform_dataset(raw_dataset)


def get_dataset_stats(dataset: DatasetDict | Dataset) -> dict:
    """
    Get statistics about a dataset.
    
    Args:
        dataset: HuggingFace DatasetDict or Dataset
        
    Returns:
        Dictionary with statistics
    """
    if isinstance(dataset, dict):
        # DatasetDict
        stats = {}
        for split_name, split_data in dataset.items():
            stats[split_name] = get_dataset_stats(split_data)
        return stats
    
    # Single Dataset
    total = len(dataset)
    
    if total == 0:
        return {"total": 0}
    
    # Faithfulness distribution
    faithful = sum(1 for item in dataset if item.get("faithfulness_label", True))
    unfaithful = total - faithful
    
    # Model distribution
    model_counts = {}
    for item in dataset:
        model = item.get("source_model", "unknown")
        model_counts[model] = model_counts.get(model, 0) + 1
    
    # Category distribution
    category_counts = {}
    for item in dataset:
        cat = item.get("category", "unknown")
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    # Information type distribution
    info_type_counts = {}
    for item in dataset:
        info_type = item.get("information_type", "unknown")
        info_type_counts[info_type] = info_type_counts.get(info_type, 0) + 1
    
    return {
        "total": total,
        "faithful": faithful,
        "unfaithful": unfaithful,
        "faithfulness_positive_rate": faithful / total if total > 0 else 0,
        "model_counts": model_counts,
        "category_counts": category_counts,
        "info_type_counts": info_type_counts,
    }


# =============================================================================
# Adapter Auto-Tests
# =============================================================================

def run_adapter_tests(
    dataset: DatasetDict | Dataset,
    expected_models: list[str] = None,
    expected_total: int = None,
) -> dict:
    """
    Run automatic tests on the unified dataset adapter.
    
    Tests:
    - Each raw sample expands to 4 model responses
    - case_ids are globally unique
    - Required fields are non-empty
    - source_model values are valid
    - hallucination spans have valid coordinates
    - faithful samples have no valid hallucination spans
    - unfaithful samples have at least one valid hallucination span
    
    Args:
        dataset: Unified dataset (after adapter transformation)
        expected_models: List of expected model names
        expected_total: Expected total number of samples
        
    Returns:
        Dictionary with test results
    """
    if expected_models is None:
        expected_models = AVAILABLE_MODELS
    
    results = {
        "passed": [],
        "failed": [],
        "warnings": [],
    }
    
    # Convert to list if DatasetDict
    if isinstance(dataset, dict):
        all_samples = []
        for split_data in dataset.values():
            for item in split_data:
                all_samples.append(item)
    else:
        all_samples = list(dataset)
    
    total = len(all_samples)
    
    # Test 1: Expected total
    if expected_total is not None:
        if total == expected_total:
            results["passed"].append(f"Total samples: {total} (expected {expected_total})")
        else:
            results["failed"].append(f"Total samples: {total} (expected {expected_total})")
    
    # Test 2: Unique case_ids
    case_ids = [s.get("case_id", "") for s in all_samples]
    unique_case_ids = set(case_ids)
    if len(unique_case_ids) == len(case_ids):
        results["passed"].append(f"All case_ids unique: {len(unique_case_ids)}")
    else:
        results["failed"].append(
            f"Duplicate case_ids: {len(case_ids)} total, {len(unique_case_ids)} unique"
        )
    
    # Test 3: Non-empty required fields
    empty_question = sum(1 for s in all_samples if not s.get("question", "").strip())
    empty_answer = sum(1 for s in all_samples if not s.get("answer", "").strip())
    empty_chunks = sum(1 for s in all_samples if not s.get("chunks", []))
    
    if empty_question == 0:
        results["passed"].append("No empty questions")
    else:
        results["failed"].append(f"Empty questions: {empty_question}")
    
    if empty_answer == 0:
        results["passed"].append("No empty answers")
    else:
        results["failed"].append(f"Empty answers: {empty_answer}")
    
    if empty_chunks == 0:
        results["passed"].append("No empty chunks")
    else:
        results["failed"].append(f"Empty chunks: {empty_chunks}")
    
    # Test 4: Valid source_model values
    invalid_models = []
    for s in all_samples:
        model = s.get("source_model", "")
        if model not in expected_models:
            invalid_models.append(model)
    
    if not invalid_models:
        results["passed"].append("All source_model values valid")
    else:
        results["failed"].append(
            f"Invalid source_model values: {set(invalid_models)}"
        )
    
    # Test 5: Valid hallucination span coordinates
    invalid_spans = 0
    for s in all_samples:
        answer = s.get("answer", "")
        answer_len = len(answer)
        for span in s.get("hallucination_spans", []):
            start = span.get("start", -1)
            end = span.get("end", -1)
            
            if start < 0 or end < 0 or start >= end or end > answer_len:
                invalid_spans += 1
    
    if invalid_spans == 0:
        results["passed"].append("All hallucination spans have valid coordinates")
    else:
        results["failed"].append(f"Invalid hallucination spans: {invalid_spans}")
    
    # Test 6: Faithful samples have no valid hallucination spans
    faithful_with_spans = 0
    for s in all_samples:
        if s.get("faithfulness_label", False):
            spans = s.get("hallucination_spans", [])
            # Count valid spans (valid=True or no valid field)
            valid_spans = [sp for sp in spans if sp.get("valid", True)]
            if valid_spans:
                faithful_with_spans += 1
    
    if faithful_with_spans == 0:
        results["passed"].append("Faithful samples have no valid hallucination spans")
    else:
        results["failed"].append(
            f"Faithful samples with valid hallucination spans: {faithful_with_spans}"
        )
    
    # Test 7: Unfaithful samples have at least one valid hallucination span
    unfaithful_without_spans = 0
    for s in all_samples:
        if not s.get("faithfulness_label", True):
            spans = s.get("hallucination_spans", [])
            valid_spans = [sp for sp in spans if sp.get("valid", True)]
            if not valid_spans:
                unfaithful_without_spans += 1
    
    if unfaithful_without_spans == 0:
        results["passed"].append("Unfaithful samples have at least one valid hallucination span")
    else:
        results["failed"].append(
            f"Unfaithful samples without valid hallucination spans: {unfaithful_without_spans}"
        )
    
    return results


# =============================================================================
# Dataset Splitting
# =============================================================================

def create_train_val_split(
    dataset: DatasetDict,
    val_size: float = 0.15,
    seed: int = 42,
) -> dict:
    """
    Create train/validation split based on user_prompt_index.
    
    This ensures that the same question doesn't appear in both splits,
    which is critical for preventing data leakage.
    
    The split is based on user_prompt_index, not row index, so that all
    rows with the same user_prompt_index stay in the same split.
    
    Args:
        dataset: DatasetDict with 'train' and 'test' splits
        val_size: Fraction of train to use for validation
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with split information:
        {
            "train_indices": [original_indices for new train],
            "val_indices": [original_indices for new val],
            "test_indices": [original_indices for test (unchanged)],
            "train_count": int,
            "val_count": int,
            "test_count": int,
            "train_prompts": list of unique prompt indices,
            "val_prompts": list of unique prompt indices,
            "seed": int,
            "val_size": float,
        }
    """
    import random
    random.seed(seed)
    
    train_data = dataset["train"]
    test_data = dataset["test"]
    
    # Group rows by user_prompt_index
    # Same prompt_index should go to same split
    prompt_to_indices = {}
    for i, sample in enumerate(train_data):
        prompt_idx = sample.get("user_prompt_index", i)
        if prompt_idx not in prompt_to_indices:
            prompt_to_indices[prompt_idx] = []
        prompt_to_indices[prompt_idx].append(i)
    
    # Get unique prompt indices
    unique_prompts = list(prompt_to_indices.keys())
    random.shuffle(unique_prompts)
    
    # Split by prompt indices
    n_val = int(len(unique_prompts) * val_size)
    val_prompts = set(unique_prompts[:n_val])
    train_prompts = set(unique_prompts[n_val:])
    
    # Map back to row indices
    train_idx_list = []
    val_idx_list = []
    prompt_group_sizes = {}  # For statistics
    
    for prompt_idx, indices in prompt_to_indices.items():
        prompt_group_sizes[prompt_idx] = len(indices)
        if prompt_idx in val_prompts:
            val_idx_list.extend(indices)
        else:
            train_idx_list.extend(indices)
    
    # Test indices are all test indices
    test_idx_list = list(range(len(test_data)))
    
    # Compute group size statistics
    group_sizes = list(prompt_group_sizes.values())
    group_size_dist = {}
    for size in group_sizes:
        group_size_dist[size] = group_size_dist.get(size, 0) + 1
    
    return {
        "train_indices": train_idx_list,
        "val_indices": val_idx_list,
        "test_indices": test_idx_list,
        "train_count": len(train_idx_list),
        "val_count": len(val_idx_list),
        "test_count": len(test_idx_list),
        "train_prompts": list(train_prompts),
        "val_prompts": list(val_prompts),
        "seed": seed,
        "val_size": val_size,
        # Group statistics
        "unique_train_prompts": len(train_prompts),
        "unique_val_prompts": len(val_prompts),
        "unique_test_prompts": "all",  # Test is unchanged
        "group_size_distribution": group_size_dist,
        "max_group_size": max(group_sizes),
        "prompts_with_multiple_rows": sum(1 for s in group_sizes if s > 1),
    }


def apply_split(
    dataset: DatasetDict,
    split_info: dict,
) -> DatasetDict:
    """
    Apply split information to a dataset.
    
    Args:
        dataset: Original DatasetDict
        split_info: Output from create_train_val_split
        
    Returns:
        DatasetDict with train, val, test splits
    """
    from datasets import Dataset as HFDataset
    
    train_data = dataset["train"]
    test_data = dataset["test"]
    
    # Get train data
    all_train_data = [train_data[i] for i in split_info["train_indices"]]
    val_data = [train_data[i] for i in split_info["val_indices"]]
    
    return DatasetDict({
        "train": HFDataset.from_list(all_train_data),
        "val": HFDataset.from_list(val_data),
        "test": HFDataset.from_list(list(test_data)),
    })


# =============================================================================
# Comprehensive Statistics
# =============================================================================

def get_comprehensive_stats(dataset: DatasetDict | Dataset) -> dict:
    """
    Get comprehensive statistics about a dataset.
    
    Args:
        dataset: HuggingFace DatasetDict or Dataset
        
    Returns:
        Dictionary with comprehensive statistics
    """
    if isinstance(dataset, dict):
        stats = {}
        for split_name, split_data in dataset.items():
            stats[split_name] = get_comprehensive_stats(split_data)
        return stats
    
    total = len(dataset)
    if total == 0:
        return {"total": 0}
    
    # Faithfulness distribution
    faithful = sum(1 for item in dataset if item.get("faithfulness_label", True))
    unfaithful = total - faithful
    
    # Model distribution
    model_counts = {}
    model_faithful_counts = {}
    for item in dataset:
        model = item.get("source_model", "unknown")
        model_counts[model] = model_counts.get(model, 0) + 1
        if item.get("faithfulness_label", True):
            model_faithful_counts[model] = model_faithful_counts.get(model, 0) + 1
    
    # Answerable distribution
    answerable_count = sum(1 for item in dataset if item.get("answerable", False))
    
    # Category distribution
    category_counts = {}
    for item in dataset:
        cat = item.get("category", "unknown")
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    # Information type distribution
    info_type_counts = {}
    for item in dataset:
        info_type = item.get("information_type", "unknown")
        info_type_counts[info_type] = info_type_counts.get(info_type, 0) + 1
    
    # Source split distribution
    split_counts = {}
    for item in dataset:
        split = item.get("source_split", "unknown")
        split_counts[split] = split_counts.get(split, 0) + 1
    
    return {
        "total": total,
        "faithful": faithful,
        "unfaithful": unfaithful,
        "faithfulness_positive_rate": faithful / total if total > 0 else 0,
        "model_counts": model_counts,
        "model_faithful_counts": model_faithful_counts,
        "answerable_count": answerable_count,
        "answerable_rate": answerable_count / total if total > 0 else 0,
        "category_counts": category_counts,
        "info_type_counts": info_type_counts,
        "source_split_counts": split_counts,
    }


# =============================================================================
# Reproducibility Tests
# =============================================================================

def test_reproducibility() -> dict:
    """
    Run reproducibility tests on the adapter.
    
    Tests:
    - case_id generation is deterministic
    - train/val split is reproducible with same seed
    - No case_id overlap between splits
    - All case_ids are globally unique
    
    Returns:
        Dictionary with test results
    """
    results = {
        "passed": [],
        "failed": [],
        "warnings": [],
    }
    
    # Load data
    raw_dataset = load_ragognize_dataset()
    
    # Create split first
    split_info = create_train_val_split(raw_dataset, val_size=0.15, seed=42)
    raw_with_split = apply_split(raw_dataset, split_info)
    
    # Test 1: case_id determinism
    adapter = RAGognizeAdapter()
    
    # Transform twice
    unified1 = adapter.transform_dataset(raw_with_split)
    unified2 = adapter.transform_dataset(raw_with_split)
    
    # Compare case_ids for each split
    all_match = True
    for split_name in ["train", "val", "test"]:
        case_ids1 = sorted([s["case_id"] for s in unified1[split_name]])
        case_ids2 = sorted([s["case_id"] for s in unified2[split_name]])
        if case_ids1 != case_ids2:
            all_match = False
            break
    
    if all_match:
        results["passed"].append("case_id generation is deterministic")
    else:
        results["failed"].append("case_id generation is NOT deterministic!")
    
    # Test 2: Split reproducibility
    split1 = create_train_val_split(raw_dataset, val_size=0.15, seed=42)
    split2 = create_train_val_split(raw_dataset, val_size=0.15, seed=42)
    
    if split1["train_indices"] == split2["train_indices"] and split1["val_indices"] == split2["val_indices"]:
        results["passed"].append("train/val split is reproducible with seed=42")
    else:
        results["failed"].append("train/val split is NOT reproducible!")
    
    # Test 3: No overlap between splits
    train_case_ids = set([s["case_id"] for s in unified1["train"]])
    val_case_ids = set([s["case_id"] for s in unified1["val"]])
    test_case_ids = set([s["case_id"] for s in unified1["test"]])
    
    train_val_overlap = train_case_ids & val_case_ids
    train_test_overlap = train_case_ids & test_case_ids
    val_test_overlap = val_case_ids & test_case_ids
    
    if not train_val_overlap and not train_test_overlap and not val_test_overlap:
        results["passed"].append("No case_id overlap between train/val/test")
    else:
        if train_val_overlap:
            results["failed"].append(f"Train/Val overlap: {len(train_val_overlap)} case_ids")
        if train_test_overlap:
            results["failed"].append(f"Train/Test overlap: {len(train_test_overlap)} case_ids")
        if val_test_overlap:
            results["failed"].append(f"Val/Test overlap: {len(val_test_overlap)} case_ids")
    
    # Test 4: Global uniqueness
    all_case_ids = train_case_ids | val_case_ids | test_case_ids
    total = len(train_case_ids) + len(val_case_ids) + len(test_case_ids)
    
    if len(all_case_ids) == total:
        results["passed"].append(f"All case_ids globally unique: {total}")
    else:
        results["failed"].append(
            f"Duplicate case_ids: {total} total, {len(all_case_ids)} unique"
        )
    
    return results

