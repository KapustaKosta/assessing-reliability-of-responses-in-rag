"""
Tests for model module.
"""

import pytest
import os
import torch
from unittest.mock import MagicMock

from token_classifier.model import (
    TokenHallucinationClassifier,
    get_device,
    compute_loss_with_class_weights,
    check_nan_inf,
)


class TestGetDevice:
    """Test device selection."""
    
    def test_auto_npu_available(self):
        if torch.npu.is_available():
            device = get_device("auto")
            assert "npu" in str(device)
        else:
            device = get_device("auto")
            assert "cpu" in str(device)
    
    def test_explicit_cpu(self):
        device = get_device("cpu")
        assert "cpu" in str(device)


class TestCheckNanInf:
    """Test NaN/Inf checking."""
    
    def test_valid_tensor(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        assert check_nan_inf(x, "test") is True
    
    def test_nan_detected(self):
        x = torch.tensor([1.0, float('nan'), 3.0])
        assert check_nan_inf(x, "test") is False
    
    def test_inf_detected(self):
        x = torch.tensor([1.0, float('inf'), 3.0])
        assert check_nan_inf(x, "test") is False


class TestComputeLossWithClassWeights:
    """Test loss computation with class weights."""

    def test_no_weights(self):
        # 3D logits: [batch, seq_len, num_classes]
        logits = torch.randn(2, 4, 2)  # batch=2, seq_len=4, classes=2
        labels = torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0]])  # batch=2, seq_len=4

        loss = compute_loss_with_class_weights(logits, labels)

        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_with_positive_weight(self):
        logits = torch.randn(2, 4, 2)
        labels = torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0]])

        loss_weighted = compute_loss_with_class_weights(
            logits, labels, positive_class_weight=3.0
        )
        loss_unweighted = compute_loss_with_class_weights(
            logits, labels, positive_class_weight=0
        )

        # With more weight on positive class, loss should be different
        assert not torch.isclose(loss_weighted, loss_unweighted, rtol=0.01)


class TestModelForward:
    """Test model forward pass."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock config."""
        from token_classifier.config import TokenClassifierConfig
        config = TokenClassifierConfig(
            model_path="/home/ma-user/work/models/mDeBERTa-v3-base-mnli-xnli",
            dropout=0.1,
        )
        return config
    
    def test_forward_shape(self, mock_config):
        """Test that forward returns expected shapes."""
        # This test requires local model, so skip if not available
        if not os.path.exists(mock_config.model_path):
            pytest.skip("Local model not available")
        
        model = TokenHallucinationClassifier(mock_config)
        model.eval()
        
        batch_size = 2
        seq_len = 128
        
        input_ids = torch.randint(0, 30000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        labels = torch.randint(0, 2, (batch_size, seq_len))
        labels[:, :10] = -100  # Mask some tokens
        
        with torch.no_grad():
            outputs = model(input_ids, attention_mask, labels)
        
        assert "logits" in outputs
        assert outputs["logits"].shape == (batch_size, seq_len, 2)
        assert "loss" in outputs
        assert "valid_token_count" in outputs
    
    def test_forward_no_labels(self, mock_config):
        """Test forward without labels."""
        if not os.path.exists(mock_config.model_path):
            pytest.skip("Local model not available")
        
        model = TokenHallucinationClassifier(mock_config)
        model.eval()
        
        input_ids = torch.randint(0, 30000, (2, 128))
        attention_mask = torch.ones(2, 128)
        
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
        
        assert "logits" in outputs
        assert "loss" not in outputs  # No loss when no labels


class TestModelBackward:
    """Test model backward pass."""
    
    @pytest.fixture
    def model(self):
        """Create model for testing."""
        if not os.path.exists("/home/ma-user/work/models/mDeBERTa-v3-base-mnli-xnli"):
            pytest.skip("Local model not available")
        
        from token_classifier.config import TokenClassifierConfig
        config = TokenClassifierConfig(
            model_path="/home/ma-user/work/models/mDeBERTa-v3-base-mnli-xnli",
        )
        return TokenHallucinationClassifier(config)
    
    def test_backward_produces_gradients(self, model):
        """Test that backward pass produces gradients."""
        model.train()
        
        batch_size = 2
        seq_len = 64
        
        input_ids = torch.randint(0, 30000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        # All valid labels
        labels = torch.randint(0, 2, (batch_size, seq_len))
        
        # Forward
        outputs = model(input_ids, attention_mask, labels)
        loss = outputs["loss"]
        
        # Backward
        loss.backward()
        
        # Check gradients exist
        assert model.classifier.weight.grad is not None
        assert torch.any(model.classifier.weight.grad != 0)
    
    def test_loss_is_finite(self, model):
        """Test that loss is finite."""
        model.train()
        
        input_ids = torch.randint(0, 30000, (2, 64))
        attention_mask = torch.ones(2, 64)
        labels = torch.randint(0, 2, (2, 64))
        
        outputs = model(input_ids, attention_mask, labels)
        loss = outputs["loss"]
        
        assert torch.isfinite(loss)


class TestModelFreezeUnfreeze:
    """Test encoder freezing."""
    
    def test_freeze_encoder(self):
        if not os.path.exists("/home/ma-user/work/models/mDeBERTa-v3-base-mnli-xnli"):
            pytest.skip("Local model not available")
        
        from token_classifier.config import TokenClassifierConfig
        config = TokenClassifierConfig(
            model_path="/home/ma-user/work/models/mDeBERTa-v3-base-mnli-xnli",
        )
        model = TokenHallucinationClassifier(config)
        
        # Check encoder is trainable by default
        for p in model.encoder.parameters():
            assert p.requires_grad is True
        
        # Freeze
        model.freeze_encoder()
        
        for p in model.encoder.parameters():
            assert p.requires_grad is False
        
        # Classifier should still be trainable
        for p in model.classifier.parameters():
            assert p.requires_grad is True
        
        # Unfreeze
        model.unfreeze_encoder()
        
        for p in model.encoder.parameters():
            assert p.requires_grad is True
