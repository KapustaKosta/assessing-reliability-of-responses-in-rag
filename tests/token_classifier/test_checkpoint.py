"""
Tests for checkpoint module.
"""

import pytest
import tempfile
import torch
from pathlib import Path

from token_classifier.checkpoint import CheckpointManager


class MockModel(torch.nn.Module):
    """Mock model for testing."""
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(10, 10)
        self.classifier = torch.nn.Linear(10, 2)


class TestCheckpointManager:
    """Test checkpoint management."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_save_and_load(self, temp_dir):
        """Test saving and loading checkpoint."""
        model = MockModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        from token_classifier.config import TokenClassifierConfig
        config = TokenClassifierConfig(seed=42)

        manager = CheckpointManager(temp_dir)

        # Save
        manager.save(
            model=model,
            optimizer=optimizer,
            scheduler=None,
            epoch=5,
            step=100,
            best_metric=0.85,
            config=config,
            metrics={"f1": 0.85},
        )

        assert manager.checkpoint_path.exists()

        # Create new manager and load
        manager2 = CheckpointManager(temp_dir)
        model2 = MockModel()
        optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.01)

        metadata = manager2.load(model2, optimizer2)

        assert metadata["epoch"] == 5
        assert metadata["step"] == 100
        assert metadata["best_metric"] == 0.85

    def test_load_nonexistent(self, temp_dir):
        """Test loading non-existent checkpoint."""
        manager = CheckpointManager(temp_dir, "nonexistent.pt")

        with pytest.raises(FileNotFoundError):
            manager.load(None)

    def test_exists(self, temp_dir):
        """Test checkpoint exists check."""
        manager = CheckpointManager(temp_dir)

        assert manager.exists() is False

        # Create a dummy checkpoint
        torch.save({"dummy": 1}, manager.checkpoint_path)

        assert manager.exists() is True

    def test_load_config(self, temp_dir):
        """Test loading config from checkpoint."""
        from token_classifier.config import TokenClassifierConfig
        config = TokenClassifierConfig(seed=123, epochs=10)

        model = MockModel()
        manager = CheckpointManager(temp_dir)
        manager.save(
            model=model,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
            scheduler=None,
            epoch=0,
            step=1,
            best_metric=0.0,
            config=config,
        )

        loaded_config = manager.load_config()

        assert loaded_config.seed == 123
        assert loaded_config.epochs == 10
