"""
Tests for labeling module.
"""

import pytest
from token_classifier.labeling import (
    span_overlaps_token,
    compute_token_label,
    compute_token_labels,
    AnswerTokenizer,
    create_synthetic_sample,
)
from token_classifier.schema import HallucinationSpan


class TestSpanOverlapsToken:
    """Test span overlap detection."""
    
    def test_no_overlap_before(self):
        # Token: [0, 3), Span: [5, 8)
        assert span_overlaps_token(0, 3, 5, 8) is False
    
    def test_no_overlap_after(self):
        # Token: [5, 8), Span: [0, 3)
        assert span_overlaps_token(5, 8, 0, 3) is False
    
    def test_boundary_touch_not_overlap(self):
        # Token: [0, 3), Span: [3, 6) - touches but doesn't overlap
        assert span_overlaps_token(0, 3, 3, 6) is False
    
    def test_boundary_touch_reverse(self):
        # Token: [3, 6), Span: [0, 3) - touches but doesn't overlap
        assert span_overlaps_token(3, 6, 0, 3) is False
    
    def test_partial_overlap_left(self):
        # Token: [0, 5), Span: [3, 8)
        assert span_overlaps_token(0, 5, 3, 8) is True
    
    def test_partial_overlap_right(self):
        # Token: [3, 8), Span: [0, 5)
        assert span_overlaps_token(3, 8, 0, 5) is True
    
    def test_token_inside_span(self):
        # Token: [3, 5), Span: [0, 10)
        assert span_overlaps_token(3, 5, 0, 10) is True
    
    def test_span_inside_token(self):
        # Token: [0, 10), Span: [3, 5)
        assert span_overlaps_token(0, 10, 3, 5) is True
    
    def test_exact_match(self):
        # Token: [0, 10), Span: [0, 10)
        assert span_overlaps_token(0, 10, 0, 10) is True


class TestComputeTokenLabel:
    """Test token label computation."""
    
    def test_supported_no_overlap(self):
        spans = [HallucinationSpan(start=10, end=15)]
        assert compute_token_label(0, 5, spans) == 0
    
    def test_hallucinated_overlaps(self):
        spans = [HallucinationSpan(start=3, end=7)]
        assert compute_token_label(0, 5, spans) == 1
    
    def test_ignores_invalid_spans(self):
        spans = [HallucinationSpan(start=3, end=7, valid=False)]
        assert compute_token_label(0, 5, spans) == 0
    
    def test_multiple_spans_one_overlaps(self):
        spans = [
            HallucinationSpan(start=0, end=3),
            HallucinationSpan(start=10, end=15),
        ]
        # Token [0, 5) overlaps with first span
        assert compute_token_label(0, 5, spans) == 1
    
    def test_multiple_spans_none_overlap(self):
        spans = [
            HallucinationSpan(start=0, end=3),
            HallucinationSpan(start=10, end=15),
        ]
        # Token [5, 8) overlaps with neither
        assert compute_token_label(5, 8, spans) == 0


class TestComputeTokenLabels:
    """Test batch token labeling."""
    
    def test_empty_spans(self):
        offsets = [(0, 3), (4, 7), (8, 10)]
        labels = compute_token_labels("Hello World", [], offsets)
        assert labels == [0, 0, 0]
    
    def test_all_hallucinated(self):
        offsets = [(0, 5), (6, 11)]
        spans = [HallucinationSpan(start=0, end=11)]
        labels = compute_token_labels("Hello World", spans, offsets)
        assert labels == [1, 1]
    
    def test_mixed_labels(self):
        offsets = [(0, 5), (6, 11)]
        spans = [HallucinationSpan(start=0, end=5)]
        labels = compute_token_labels("Hello World", spans, offsets)
        assert labels == [1, 0]


class TestAnswerTokenizer:
    """Test AnswerTokenizer."""
    
    def test_tokenize_with_offsets(self):
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(
            "/home/ma-user/work/models/mDeBERTa-v3-base-mnli-xnli",
            local_files_only=True,
            use_safetensors=True,
        )
        
        answer_tokenizer = AnswerTokenizer(tokenizer)
        token_ids, offsets = answer_tokenizer._tokenize_with_offsets("Hello World")
        
        assert len(token_ids) == len(offsets)
        # Check offsets map back to original (mDeBERTa uses subword tokenization)
        # So we check that the token decode matches the substring
        for tid, (start, end) in zip(token_ids, offsets):
            decoded = tokenizer.decode([tid])
            # The decoded text should match the substring (or be a subword of it)
            assert "Hello World"[start:end] == decoded or decoded in "Hello World"
    
    def test_tokenize_sample(self):
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(
            "/home/ma-user/work/models/mDeBERTa-v3-base-mnli-xnli",
            local_files_only=True,
            use_safetensors=True,
        )
        
        answer_tokenizer = AnswerTokenizer(tokenizer)
        
        windows = answer_tokenizer.tokenize_sample(
            context="This is the context.",
            question="What is the answer?",
            answer="Paris",
        )
        
        assert len(windows) >= 1
        window = windows[0]
        assert "answer_ids" in window
        assert "answer_offsets" in window


class TestSyntheticData:
    """Test synthetic data generation."""
    
    def test_create_synthetic_sample(self):
        sample = create_synthetic_sample(
            sample_id="test1",
            question_id="q1",
            answer="Paris is the capital.",
            hallucination_spans=[(0, 5)],  # "Paris"
            context="France information",
            question="What is the capital?",
        )
        
        assert sample.sample_id == "test1"
        assert sample.answer == "Paris is the capital."
        assert len(sample.hallucination_spans) == 1
        assert sample.hallucination_spans[0].start == 0
        assert sample.hallucination_spans[0].end == 5
        assert sample.has_hallucinations is True
