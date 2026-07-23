"""
Tests for schema module.
"""

import pytest
from token_classifier.schema import (
    HallucinationSpan,
    TokenSample,
    UnifiedDataSchema,
    ValidationMode,
    validate_span,
    create_grouped_split,
    audit_split,
)


class TestHallucinationSpan:
    """Test HallucinationSpan class."""
    
    def test_create_span(self):
        span = HallucinationSpan(start=5, end=10, text="hello")
        assert span.start == 5
        assert span.end == 10
        assert span.text == "hello"
        assert span.valid is True
    
    def test_to_dict(self):
        span = HallucinationSpan(start=0, end=5)
        d = span.to_dict()
        assert d["start"] == 0
        assert d["end"] == 5
        assert d["valid"] is True
    
    def test_from_dict(self):
        d = {"start": 0, "end": 5, "valid": True}
        span = HallucinationSpan.from_dict(d)
        assert span.start == 0
        assert span.end == 5


class TestTokenSample:
    """Test TokenSample class."""
    
    def test_create_sample(self):
        sample = TokenSample(
            sample_id="test1",
            question_id="q1",
            context="Context",
            question="Question?",
            answer="Answer",
        )
        assert sample.sample_id == "test1"
        assert sample.question_id == "q1"
        assert sample.answer == "Answer"
        assert len(sample.hallucination_spans) == 0
        assert sample.split == "train"
    
    def test_has_hallucinations(self):
        sample = TokenSample(
            sample_id="test1",
            question_id="q1",
            context="Context",
            question="Question?",
            answer="Answer",
            hallucination_spans=[
                HallucinationSpan(start=0, end=3, valid=True),
            ],
        )
        assert sample.has_hallucinations is True
        
        sample2 = TokenSample(
            sample_id="test2",
            question_id="q1",
            context="Context",
            question="Question?",
            answer="Answer",
        )
        assert sample2.has_hallucinations is False
    
    def test_from_dict(self):
        d = {
            "sample_id": "s1",
            "question_id": "q1",
            "context": "Context",
            "question": "Question?",
            "answer": "Answer",
            "hallucination_spans": [
                {"start": 0, "end": 3, "valid": True}
            ],
        }
        sample = TokenSample.from_dict(d)
        assert sample.sample_id == "s1"
        assert sample.answer == "Answer"


class TestValidateSpan:
    """Test span validation."""
    
    def test_valid_span(self):
        span = HallucinationSpan(start=0, end=5)
        answer = "Hello World"
        assert validate_span(span, answer) is True
    
    def test_invalid_negative_start(self):
        span = HallucinationSpan(start=-5, end=5)
        answer = "Hello"
        with pytest.raises(ValueError):
            validate_span(span, answer, mode=ValidationMode.STRICT)
    
    def test_invalid_end_before_start(self):
        span = HallucinationSpan(start=10, end=5)
        answer = "Hello"
        with pytest.raises(ValueError):
            validate_span(span, answer, mode=ValidationMode.STRICT)
    
    def test_invalid_end_exceeds_answer(self):
        span = HallucinationSpan(start=0, end=100)
        answer = "Hello"
        with pytest.raises(ValueError):
            validate_span(span, answer, mode=ValidationMode.STRICT)
    
    def test_lenient_mode_skips(self):
        span = HallucinationSpan(start=-5, end=5)
        answer = "Hello"
        assert validate_span(span, answer, mode=ValidationMode.LENIENT) is False


class TestCreateGroupedSplit:
    """Test grouped split creation."""
    
    def test_split_no_overlap(self):
        samples = [
            TokenSample(sample_id=f"s{i}", question_id=f"q{i//2}",
                       context="", question="?", answer="A", split="train")
            for i in range(20)
        ]
        
        result = create_grouped_split(samples, dev_fraction=0.2, seed=42)
        
        train_qids = result["train_question_ids"]
        dev_qids = result["dev_question_ids"]
        
        # No overlap
        assert len(train_qids & dev_qids) == 0
    
    def test_split_deterministic(self):
        samples = [
            TokenSample(sample_id=f"s{i}", question_id=f"q{i}",
                       context="", question="?", answer="A")
            for i in range(20)
        ]
        
        result1 = create_grouped_split(samples, dev_fraction=0.2, seed=42)
        result2 = create_grouped_split(samples, dev_fraction=0.2, seed=42)
        
        assert result1["train_question_ids"] == result2["train_question_ids"]
        assert result1["dev_question_ids"] == result2["dev_question_ids"]
    
    def test_same_question_stays_together(self):
        # Same question, different samples
        samples = [
            TokenSample(sample_id="s1", question_id="q1", context="", question="?", answer="A1"),
            TokenSample(sample_id="s2", question_id="q1", context="", question="?", answer="A2"),
            TokenSample(sample_id="s3", question_id="q2", context="", question="?", answer="B"),
        ]
        
        result = create_grouped_split(samples, dev_fraction=0.5, seed=42)
        
        # All q1 samples should be in same split
        q1_splits = set()
        for s in result["train_samples"] + result["dev_samples"]:
            if s.question_id == "q1":
                q1_splits.add(s.split)
        
        # q1 should be entirely in train or entirely in dev
        assert len(q1_splits) == 1


class TestAuditSplit:
    """Test split auditing."""
    
    def test_no_leakage(self):
        samples = [
            TokenSample(sample_id=f"s{i}", question_id=f"q{i//3}",
                       context="", question="?", answer="A", split="train" if i < 15 else "dev")
            for i in range(20)
        ]
        
        audit = audit_split(samples)
        
        assert audit["has_leakage"] is False
        assert audit["train_dev_overlap"] == 0
    
    def test_leakage_detected(self):
        # Create samples with overlapping question_ids
        samples = [
            TokenSample(sample_id=f"s{i}", question_id="q1",
                       context="", question="?", answer="A", split="train" if i < 10 else "dev")
            for i in range(20)
        ]
        
        audit = audit_split(samples)
        
        assert audit["has_leakage"] is True
        assert audit["train_dev_overlap"] > 0


class TestUnifiedDataSchema:
    """Test UnifiedDataSchema adapter."""
    
    def test_detect_ragognize_fields(self):
        d = {
            "user_prompt": "What is the capital?",
            "answer": "Paris",
            "chunks": ["Chunk 1", "Chunk 2"],
            "case_id": "test123",
            "user_prompt_index": 1,
            "source_model": "ModelA",
            "hallucination_spans": [{"start": 0, "end": 3, "valid": True}],
        }
        
        sample = UnifiedDataSchema.from_dict(d, log_selection=False)
        
        assert sample is not None
        assert sample.question == "What is the capital?"
        assert sample.answer == "Paris"
    
    def test_detect_csv_fields(self):
        d = {
            "question": "What is 2+2?",
            "answer": "4",
            "context": "Math context",
        }
        
        sample = UnifiedDataSchema.from_dict(d, log_selection=False)
        
        assert sample is not None
        assert sample.question == "What is 2+2?"
        assert sample.answer == "4"
        assert sample.context == "Math context"
