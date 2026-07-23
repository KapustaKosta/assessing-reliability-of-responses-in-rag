"""
Tests for data leakage prevention.
"""

import pytest
from token_classifier.schema import (
    TokenSample,
    create_grouped_split,
    audit_split,
)


class TestNoLeakage:
    """Test data leakage prevention."""
    
    def test_question_id_not_in_multiple_splits(self):
        """All samples with same question_id must be in same split."""
        samples = [
            TokenSample(
                sample_id=f"s{i}",
                question_id=f"q{i % 10}",  # 10 unique questions, 100 samples
                context="",
                question="?",
                answer="A",
            )
            for i in range(100)
        ]
        
        result = create_grouped_split(samples, dev_fraction=0.2, seed=42)
        
        # Check that each question is in only one split
        question_splits = {}
        for sample in result["train_samples"] + result["dev_samples"]:
            qid = sample.question_id
            if qid in question_splits:
                assert question_splits[qid] == sample.split, \
                    f"Question {qid} appears in multiple splits!"
            else:
                question_splits[qid] = sample.split
        
        assert len(question_splits) == 10  # All 10 unique questions present
    
    def test_train_dev_no_overlap(self):
        """Train and dev must not share question_ids."""
        samples = [
            TokenSample(
                sample_id=f"s{i}",
                question_id=f"q{i}",
                context="",
                question="?",
                answer="A",
            )
            for i in range(50)
        ]
        
        result = create_grouped_split(samples, dev_fraction=0.2, seed=42)
        
        train_qids = result["train_question_ids"]
        dev_qids = result["dev_question_ids"]
        
        assert len(train_qids & dev_qids) == 0
    
    def test_deterministic_split(self):
        """Same seed must produce same split."""
        samples = [
            TokenSample(
                sample_id=f"s{i}",
                question_id=f"q{i}",
                context="",
                question="?",
                answer="A",
            )
            for i in range(30)
        ]
        
        result1 = create_grouped_split(samples, dev_fraction=0.2, seed=123)
        result2 = create_grouped_split(samples, dev_fraction=0.2, seed=123)
        
        assert result1["train_question_ids"] == result2["train_question_ids"]
        assert result1["dev_question_ids"] == result2["dev_question_ids"]
    
    def test_different_seeds_different_splits(self):
        """Different seeds must produce different splits."""
        samples = [
            TokenSample(
                sample_id=f"s{i}",
                question_id=f"q{i}",
                context="",
                question="?",
                answer="A",
            )
            for i in range(30)
        ]
        
        result1 = create_grouped_split(samples, dev_fraction=0.2, seed=42)
        result2 = create_grouped_split(samples, dev_fraction=0.2, seed=123)
        
        assert result1["train_question_ids"] != result2["train_question_ids"]
    
    def test_audit_detects_leakage(self):
        """Audit must detect leakage."""
        # Create samples with intentional overlap
        samples = []
        for i in range(20):
            samples.append(TokenSample(
                sample_id=f"s{i}",
                question_id="q1",  # Same question in multiple splits
                context="",
                question="?",
                answer="A",
                split="train" if i < 10 else "dev",
            ))
        
        audit = audit_split(samples)
        
        assert audit["has_leakage"] is True
        assert audit["train_dev_overlap"] > 0
    
    def test_audit_no_leakage(self):
        """Audit must confirm no leakage."""
        # Create samples with NO leakage
        # First 15 samples: q0-q7 each appears twice -> train
        # Last 5 samples: q8-q9 each appears twice -> dev
        # Each question_id appears in only ONE split
        samples = []
        # Train samples (q0-q7)
        for i in range(16):
            samples.append(TokenSample(
                sample_id=f"s_train_{i}",
                question_id=f"q{i // 2}",  # q0, q0, q1, q1, ..., q7, q7
                context="",
                question="?",
                answer="A",
                split="train",
            ))
        # Dev samples (q8-q9)
        for i in range(16, 20):
            samples.append(TokenSample(
                sample_id=f"s_dev_{i}",
                question_id=f"q{i // 2}",  # q8, q8, q9, q9
                context="",
                question="?",
                answer="A",
                split="dev",
            ))

        audit = audit_split(samples)

        assert audit["has_leakage"] is False
        assert audit["train_dev_overlap"] == 0
    
    def test_all_models_stay_together(self):
        """All responses for same question must be in same partition."""
        samples = []
        for qid in range(5):
            for model in ["ModelA", "ModelB", "ModelC"]:
                samples.append(TokenSample(
                    sample_id=f"s{qid}_{model}",
                    question_id=f"q{qid}",
                    context="",
                    question="?",
                    answer="A",
                    source_model=model,
                ))
        
        result = create_grouped_split(samples, dev_fraction=0.4, seed=42)
        
        # Group by question_id
        qid_partitions = {}
        for sample in result["train_samples"] + result["dev_samples"]:
            qid = sample.question_id
            if qid not in qid_partitions:
                qid_partitions[qid] = sample.split
            else:
                assert qid_partitions[qid] == sample.split
