"""
Constants for RAGognize dataset adapter.
"""

from pathlib import Path

# Project root (determined at import time)
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()

# Data paths (relative to PROJECT_ROOT)
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "ragognize" / "data"
TRAIN_PARQUET = DATA_RAW_DIR / "train-00000-of-00001.parquet"
TEST_PARQUET = DATA_RAW_DIR / "test-00000-of-00001.parquet"

# Results directory
RESULTS_DIR = PROJECT_ROOT / "results" / "ragognize_data_preparation"

# Source model names (excluding golden_answer)
SOURCE_MODELS = [
    "Llama-2-7b-chat-hf",
    "Llama-3.1-8B-Instruct",
    "Mistral-7B-Instruct-v0.1",
    "Mistral-7B-Instruct-v0.3",
]

# Golden answer model (not included in source models)
GOLDEN_ANSWER_MODEL = "golden_answer"

# All model names including golden
ALL_MODELS = SOURCE_MODELS + [GOLDEN_ANSWER_MODEL]

# Random seed for reproducibility
DEFAULT_SEED = 42

# Validation split ratio
DEFAULT_VAL_RATIO = 0.15
