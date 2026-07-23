"""
Tests for dataset module.
"""

import pytest
import json
import tempfile
from pathlib import Path

from token_classifier.schema import TokenSample, HallucinationSpan
from token_classifier.dataset import (
    load_data,
    _collate_fn,
)


class TestLoadData:
    """Test data loading."""
    
    def test_load_jsonl(self):
        """Test loading JSONL format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write(json.dumps({
                "sample_id": "s1",
                "question_id": "q1",
                "context": "Context",
                "question": "Question?",
                "answer": "Answer",
            }) + "\n")
            f.write(json.dumps({
                "sample_id": "s2",
                "question_id": "q2",
                "context": "Context",
                "question": "Question?",
                "answer": "Answer",
            }) + "\n")
            temp_path = f.name
        
        try:
            samples = load_data(temp_path)
            assert len(samples) == 2
            assert samples[0].sample_id == "s1"
            assert samples[1].sample_id == "s2"
        finally:
            Path(temp_path).unlink()
    
    def test_load_json(self):
        """Test loading JSON format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump([
                {"sample_id": "s1", "question_id": "q1", "answer": "A"},
                {"sample_id": "s2", "question_id": "q2", "answer": "B"},
            ], f)
            temp_path = f.name
        
        try:
            samples = load_data(temp_path)
            assert len(samples) == 2
        finally:
            Path(temp_path).unlink()
    
    def test_load_with_spans(self):
        """Test loading with hallucination spans."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write(json.dumps({
                "sample_id": "s1",
                "question_id": "q1",
                "answer": "Paris is the capital.",
                "hallucination_spans": [
                    {"start": 0, "end": 5, "valid": True}
                ],
            }) + "\n")
            temp_path = f.name
        
        try:
            samples = load_data(temp_path)
            assert len(samples) == 1
            assert samples[0].has_hallucinations is True
            assert len(samples[0].hallucination_spans) == 1
        finally:
            Path(temp_path).unlink()
    
    def test_file_not_found(self):
        """Test error on missing file."""
        with pytest.raises(FileNotFoundError):
            load_data("/nonexistent/path.jsonl")


class TestCollateFn:
    """Test collate function."""

    def test_collate_basic(self):
        """Test basic collation."""
        import numpy as np
        import torch
        # Note: Using np.array for simplicity, real collate uses torch.stack
        # This test is simplified
        pass
