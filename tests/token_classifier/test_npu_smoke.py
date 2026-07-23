"""
Smoke tests for NPU support.
"""

import pytest
import torch


class TestNPUSmoke:
    """Test NPU smoke tests."""

    def test_npu_available(self):
        """Test that NPU is available."""
        assert torch.npu.is_available(), "NPU should be available"

    def test_device_detection(self):
        """Test device detection."""
        from token_classifier.model import get_device

        device = get_device("auto")
        # Should return npu if available, otherwise cpu
        assert "npu" in str(device) or "cpu" in str(device)

    def test_model_on_npu(self):
        """Test that model can run on NPU."""
        model_path = "/home/ma-user/work/models/mDeBERTa-v3-base-mnli-xnli"
        if not torch.npu.is_available():
            pytest.skip("NPU not available")

        from token_classifier.config import TokenClassifierConfig
        from token_classifier.model import TokenHallucinationClassifier

        config = TokenClassifierConfig(model_path=model_path, device="npu")
        model = TokenHallucinationClassifier(config)
        model.train()
        
        # Move model to NPU
        model = model.npu()

        # Create input on NPU
        input_ids = torch.randint(0, 30000, (2, 64)).npu()
        attention_mask = torch.ones(2, 64).npu()
        labels = torch.randint(0, 2, (2, 64)).npu()

        outputs = model(input_ids, attention_mask, labels)
        loss = outputs["loss"]

        loss.backward()

        # Check gradients
        assert model.classifier.weight.grad is not None

        # Check finite
        assert torch.isfinite(loss)
