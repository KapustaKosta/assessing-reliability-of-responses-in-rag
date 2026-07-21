"""
Constants and configuration for NLI Faithfulness baseline.
"""

import os
from pathlib import Path
from typing import Literal

# Project paths - use environment variable or fallback to relative path
# This allows running from any directory
_PROJECT_ROOT = os.environ.get(
    "PROJECT_ROOT",
    Path(__file__).parent.parent.parent.parent.resolve()
)
PROJECT_ROOT = Path(_PROJECT_ROOT)
PROCESSED_DIR = PROJECT_ROOT / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "stage3_nli_faithfulness"
SRC_DIR = PROJECT_ROOT / "src" / "nli_faithfulness"

# Data columns
CHUNK_COLUMNS = [f"chunk_{i}" for i in range(1, 9)]

# Required columns from processed data
REQUIRED_COLUMNS = [
    "case_id",
    "answer",
    "binary_faithfulness",
    "binary_relevancy",
    "joint_label",
    "chunk_count",
    "retrieval_config",
    *CHUNK_COLUMNS,
]

# NLI model configurations
DEFAULT_MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
LARGE_MODEL_NAME = "joeddav/xlm-roberta-large-xnli"

# Device preference order
DEVICE_PREFERENCE: list[Literal["cuda", "mps", "cpu"]] = ["cuda", "mps", "cpu"]

# NLI label constants (semantic meaning)
NLI_ENTAILMENT = "entailment"
NLI_NEUTRAL = "neutral"
NLI_CONTRADICTION = "contradiction"

# Inference settings
DEFAULT_BATCH_SIZE = 8
MAX_RETRIES = 3
CACHE_DIR = RESULTS_DIR / "cache"

# Segmentation settings
# Split on sentence-ending punctuation: . ? ! and newlines
SENTENCE_SPLIT_PATTERN = r'(?<=[.!?])\s+'
# Minimum sentence length (characters) to consider for standalone
MIN_SENTENCE_LENGTH = 20
# Patterns that indicate important content (should not be merged away)
IMPORTANT_CONTENT_PATTERNS = [
    r'\d+',           # Numbers
    r'\d+[.,]\d+',    # Decimals
    r'\d+%',          # Percentages
    r'\d+\s*(?:руб|р\.|₽|USD|EUR)',  # Currency amounts
    r'(?:вклад|карта|кредит|депозит|ставка|процент)',  # Banking terms
    r'(?:Альфа|банк)',  # Bank names
    r'(?:если|когда|при|то|или|но|и|не)',  # Conjunctions/conditions
]

# Windowing settings
DEFAULT_WINDOW_OVERLAP_TOKENS = 32
MIN_WINDOW_TOKENS = 32
MAX_CHUNK_WINDOWS_PER_CHUNK = 20  # Safety limit

# Aggregation threshold defaults
DEFAULT_ENTAILMENT_THRESHOLD = 0.5
DEFAULT_CONTRADICTION_THRESHOLD = 0.5
DEFAULT_SENTENCE_SUPPORT_THRESHOLD = 0.5

# Evaluation settings
THRESHOLD_RANGE_START = 0.10
THRESHOLD_RANGE_END = 0.90
THRESHOLD_RANGE_STEP = 0.01

# TF-IDF results path for comparison
TFIDF_RESULTS_DIR = PROJECT_ROOT / "results" / "stage2_tfidf"
