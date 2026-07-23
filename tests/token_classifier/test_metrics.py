"""
Tests for metrics module.
"""

import pytest
import numpy as np
from token_classifier.metrics import (
    compute_token_metrics,
    compute_span_metrics,
    spans_to_char_set,
    compute_answer_metrics,
    compute_calibration_metrics,
)


class TestComputeTokenMetrics:
    """Test token-level metrics."""
    
    def test_perfect_predictions(self):
        y_true = [0, 0, 1, 1]
        y_pred = [0, 0, 1, 1]
        
        metrics = compute_token_metrics(y_true, y_pred)
        
        assert metrics["accuracy"] == 1.0
        assert metrics["positive_precision"] == 1.0
        assert metrics["positive_recall"] == 1.0
        assert metrics["positive_f1"] == 1.0
    
    def test_all_supported(self):
        y_true = [0, 0, 0, 0]
        y_pred = [0, 0, 0, 0]
        
        metrics = compute_token_metrics(y_true, y_pred)
        
        assert metrics["accuracy"] == 1.0
        assert metrics["positive_f1"] == 0.0
    
    def test_all_hallucinated(self):
        y_true = [1, 1, 1, 1]
        y_pred = [1, 1, 1, 1]
        
        metrics = compute_token_metrics(y_true, y_pred)
        
        assert metrics["accuracy"] == 1.0
        assert metrics["positive_f1"] == 1.0
    
    def test_partial_predictions(self):
        y_true = [0, 1, 1, 1]
        y_pred = [1, 1, 0, 1]
        
        metrics = compute_token_metrics(y_true, y_pred)
        
        assert metrics["accuracy"] == 0.5
        assert metrics["total_count"] == 4
    
    def test_with_probabilities(self):
        y_true = [0, 1]
        y_pred = [0, 1]
        y_prob = [0.1, 0.9]
        
        metrics = compute_token_metrics(y_true, y_pred, y_prob)
        
        assert "roc_auc" in metrics
        assert "pr_auc" in metrics


class TestSpanMetrics:
    """Test span-level metrics."""
    
    def test_spans_to_char_set(self):
        spans = [
            {"start": 0, "end": 3},
            {"start": 5, "end": 8},
        ]
        
        char_set = spans_to_char_set("HelloWorld", spans)
        
        assert 0 in char_set
        assert 1 in char_set
        assert 2 in char_set
        assert 5 in char_set
        assert 7 in char_set
        assert 3 not in char_set
    
    def test_compute_span_metrics_perfect(self):
        answer = "Hello World"
        gold_spans = [{"start": 0, "end": 5}]  # "Hello"
        pred_spans = [{"start": 0, "end": 5}]
        
        metrics = compute_span_metrics(answer, gold_spans, pred_spans)
        
        assert metrics["character_precision"] == 1.0
        assert metrics["character_recall"] == 1.0
        assert metrics["character_f1"] == 1.0
    
    def test_compute_span_metrics_partial(self):
        answer = "Hello World"
        gold_spans = [{"start": 0, "end": 5}]  # "Hello"
        pred_spans = [{"start": 2, "end": 8}]  # "llo Wo"
        
        metrics = compute_span_metrics(answer, gold_spans, pred_spans)
        
        # "Hello" = positions 0,1,2,3,4
        # "llo Wo" = positions 2,3,4,5,6,7
        # Overlap = positions 2,3,4 = "llo" = 3 chars
        # precision = 3 overlap / 6 predicted = 0.5
        # recall = 3 overlap / 5 gold = 0.6
        assert 0 < metrics["character_precision"] < 1
        assert 0 < metrics["character_recall"] < 1
        assert metrics["character_f1"] > 0
    
    def test_compute_span_metrics_no_overlap(self):
        answer = "Hello World"
        gold_spans = [{"start": 0, "end": 5}]
        pred_spans = [{"start": 6, "end": 11}]
        
        metrics = compute_span_metrics(answer, gold_spans, pred_spans)
        
        assert metrics["character_precision"] == 0.0
        assert metrics["character_recall"] == 0.0
        assert metrics["character_f1"] == 0.0
    
    def test_empty_pred_spans(self):
        answer = "Hello World"
        gold_spans = [{"start": 0, "end": 5}]
        pred_spans = []
        
        metrics = compute_span_metrics(answer, gold_spans, pred_spans)
        
        assert metrics["character_precision"] == 0.0
        assert metrics["character_recall"] == 0.0


class TestAnswerMetrics:
    """Test answer-level metrics."""
    
    def test_perfect_answer(self):
        answers = ["A", "B", "C"]
        gold_labels = [0, 1, 1]
        pred_labels = [0, 1, 1]
        
        metrics = compute_answer_metrics(answers, gold_labels, pred_labels)
        
        assert metrics["accuracy"] == 1.0
        assert metrics["f1"] == 1.0
    
    def test_answer_with_probs(self):
        answers = ["A", "B"]
        gold_labels = [0, 1]
        pred_labels = [0, 1]
        pred_probs = [0.2, 0.8]
        
        metrics = compute_answer_metrics(answers, gold_labels, pred_labels, pred_probs)
        
        assert "roc_auc" in metrics
        assert "pr_auc" in metrics


class TestCalibrationMetrics:
    """Test calibration metrics."""
    
    def test_perfect_calibration(self):
        # Well-calibrated probabilities
        y_true = [0, 0, 0, 0, 1, 1, 1, 1]
        y_prob = [0.2, 0.2, 0.2, 0.2, 0.8, 0.8, 0.8, 0.8]
        
        metrics = compute_calibration_metrics(y_true, y_prob)
        
        assert "brier_score" in metrics
        assert "expected_calibration_error" in metrics
        assert metrics["brier_score"] < 0.1
    
    def test_poor_calibration(self):
        # Overconfident predictions
        y_true = [0, 1, 0, 1]
        y_prob = [0.99, 0.99, 0.01, 0.01]
        
        metrics = compute_calibration_metrics(y_true, y_prob)
        
        # Should have higher ECE
        assert metrics["expected_calibration_error"] > 0.3
