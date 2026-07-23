"""
Integration tests using local model.
"""

import pytest
import os


class TestLocalModelIntegration:
    """Test integration with local model."""
    
    def test_model_path_from_env(self):
        """Test that model path is read from environment."""
        model_path = os.environ.get("CLAIM_MIL_MODEL_PATH", "")
        if not model_path or not os.path.isdir(model_path):
            pytest.skip("CLAIM_MIL_MODEL_PATH not set or invalid")
        
        from token_classifier.config import TokenClassifierConfig
        config = TokenClassifierConfig()
        
        resolved = config.get_resolved_model_path()
        
        assert resolved == model_path
    
    def test_tokenizer_loads(self):
        """Test that tokenizer loads from local path."""
        model_path = os.environ.get("CLAIM_MIL_MODEL_PATH", "")
        if not model_path or not os.path.isdir(model_path):
            pytest.skip("CLAIM_MIL_MODEL_PATH not set or invalid")
        
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            use_safetensors=True,
        )
        
        assert tokenizer is not None
        assert tokenizer.pad_token_id is not None
    
    def test_model_loads(self):
        """Test that model loads from local path."""
        model_path = os.environ.get("CLAIM_MIL_MODEL_PATH", "")
        if not model_path or not os.path.isdir(model_path):
            pytest.skip("CLAIM_MIL_MODEL_PATH not set or invalid")
        
        from token_classifier.config import TokenClassifierConfig
        from token_classifier.model import TokenHallucinationClassifier
        
        config = TokenClassifierConfig(model_path=model_path)
        model = TokenHallucinationClassifier(config)
        
        assert model is not None
        
        # Check encoder and classifier exist
        assert hasattr(model, "encoder")
        assert hasattr(model, "classifier")
    
    def test_model_forward(self):
        """Test model forward pass."""
        model_path = os.environ.get("CLAIM_MIL_MODEL_PATH", "")
        if not model_path or not os.path.isdir(model_path):
            pytest.skip("CLAIM_MIL_MODEL_PATH not set or invalid")
        
        import torch
        from token_classifier.config import TokenClassifierConfig
        from token_classifier.model import TokenHallucinationClassifier
        
        config = TokenClassifierConfig(model_path=model_path)
        model = TokenHallucinationClassifier(config)
        model.eval()
        
        batch_size = 2
        seq_len = 64
        
        input_ids = torch.randint(0, 30000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
        
        assert "logits" in outputs
        assert outputs["logits"].shape == (batch_size, seq_len, 2)
    
    def test_model_backward(self):
        """Test model backward pass."""
        model_path = os.environ.get("CLAIM_MIL_MODEL_PATH", "")
        if not model_path or not os.path.isdir(model_path):
            pytest.skip("CLAIM_MIL_MODEL_PATH not set or invalid")
        
        import torch
        from token_classifier.config import TokenClassifierConfig
        from token_classifier.model import TokenHallucinationClassifier
        
        config = TokenClassifierConfig(model_path=model_path)
        model = TokenHallucinationClassifier(config)
        model.train()
        
        input_ids = torch.randint(0, 30000, (2, 64))
        attention_mask = torch.ones(2, 64)
        labels = torch.randint(0, 2, (2, 64))
        
        outputs = model(input_ids, attention_mask, labels)
        loss = outputs["loss"]
        
        loss.backward()
        
        # Check gradients exist and are non-zero
        assert model.classifier.weight.grad is not None
        assert torch.any(model.classifier.weight.grad != 0)
        
        # Check loss is finite
        assert torch.isfinite(loss)
