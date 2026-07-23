"""
Loader for RAGognize dataset from local Parquet files.
"""

import logging
from pathlib import Path
from typing import Optional

import datasets
from datasets import DatasetDict

from .constants import TRAIN_PARQUET, TEST_PARQUET, RESULTS_DIR

logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent.parent.resolve()


def load_ragognize_dataset(
    train_path: Optional[Path] = None,
    test_path: Optional[Path] = None,
    data_dir: Optional[Path] = None,
) -> DatasetDict:
    """
    Load RAGognize dataset from local Parquet files.
    
    Args:
        train_path: Path to train Parquet file. If None, uses default.
        test_path: Path to test Parquet file. If None, uses default.
        data_dir: Alternative to specifying train_path and test_path individually.
                  If provided, constructs paths as data_dir / "train-*.parquet" etc.
    
    Returns:
        DatasetDict with 'train' and 'test' splits.
    
    Raises:
        FileNotFoundError: If Parquet files don't exist.
        ValueError: If required fields are missing.
    """
    # Determine paths
    if data_dir is not None:
        data_dir = Path(data_dir)
        train_file = data_dir / "train-00000-of-00001.parquet"
        test_file = data_dir / "test-00000-of-00001.parquet"
    else:
        if train_path is None:
            train_file = TRAIN_PARQUET
        else:
            train_file = Path(train_path)
        
        if test_path is None:
            test_file = TEST_PARQUET
        else:
            test_file = Path(test_path)
    
    # Verify files exist
    if not train_file.exists():
        raise FileNotFoundError(f"Train file not found: {train_file}")
    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")
    
    logger.info(f"Loading train from: {train_file}")
    logger.info(f"Loading test from: {test_file}")
    
    # Set offline mode
    import os
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    
    # Load dataset
    dataset = datasets.load_dataset(
        "parquet",
        data_files={
            "train": str(train_file),
            "test": str(test_file),
        },
    )
    
    # Log dataset info
    for split_name, split_data in dataset.items():
        logger.info(f"Split: {split_name}")
        logger.info(f"  Rows: {len(split_data)}")
        logger.info(f"  Features: {list(split_data.features.keys())}")
    
    return dataset


def get_dataset_info(dataset: DatasetDict) -> dict:
    """
    Get comprehensive information about the dataset.
    
    Args:
        dataset: The loaded dataset.
    
    Returns:
        Dictionary with dataset statistics.
    """
    info = {
        "splits": {},
        "total_rows": 0,
    }
    
    for split_name, split_data in dataset.items():
        info["splits"][split_name] = {
            "rows": len(split_data),
            "features": list(split_data.features.keys()),
            "column_types": {
                col: str(split_data.features[col]) 
                for col in split_data.features.keys()
            },
        }
        info["total_rows"] += len(split_data)
    
    return info


def verify_required_fields(dataset: DatasetDict) -> list[str]:
    """
    Verify that required fields exist in the dataset.
    
    Args:
        dataset: The loaded dataset.
    
    Returns:
        List of missing required fields (empty if all present).
    """
    # Required top-level fields
    required_fields = [
        "user_prompt_index",
        "user_prompt",
        "answerable",
        "documents",
        "responses",
    ]
    
    # Check in train split
    train_data = dataset.get("train")
    if train_data is None:
        return ["train split not found"]
    
    missing = []
    for field in required_fields:
        if field not in train_data.features:
            missing.append(field)
    
    return missing


def save_dataset_info(dataset: DatasetDict, output_path: Path) -> None:
    """
    Save dataset info to JSON file.
    
    Args:
        dataset: The loaded dataset.
        output_path: Path to save the info JSON.
    """
    import json
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    info = get_dataset_info(dataset)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Dataset info saved to: {output_path}")
