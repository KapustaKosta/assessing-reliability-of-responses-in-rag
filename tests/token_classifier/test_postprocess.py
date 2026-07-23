"""
Tests for postprocess module.
"""

import pytest
import numpy as np
from token_classifier.postprocess import (
    tokens_to_spans,
    TokenPrediction,
    compute_answer_score,
    predict_answer_hallucination,
)


class TestTokensToSpans:
    """Test token to span conversion."""

    def test_no_hallucinated_tokens(self):
        tokens = [
            TokenPrediction(text="Hello", start=0, end=5, p_hallucination=0.1, predicted_label=0),
            TokenPrediction(text=" World", start=5, end=11, p_hallucination=0.2, predicted_label=0),
        ]

        spans = tokens_to_spans(tokens, threshold=0.5)

        assert len(spans) == 0

    def test_single_hallucinated_token(self):
        tokens = [
            TokenPrediction(text="Hello", start=0, end=5, p_hallucination=0.9, predicted_label=1),
            TokenPrediction(text=" World", start=5, end=11, p_hallucination=0.1, predicted_label=0),
        ]

        spans = tokens_to_spans(tokens, threshold=0.5)

        assert len(spans) == 1
        assert spans[0]["start"] == 0
        assert spans[0]["end"] == 5

    def test_adjacent_hallucinated_tokens_merged(self):
        tokens = [
            TokenPrediction(text="Hello", start=0, end=5, p_hallucination=0.9, predicted_label=1),
            TokenPrediction(text=" ", start=5, end=6, p_hallucination=0.8, predicted_label=1),
            TokenPrediction(text="World", start=6, end=11, p_hallucination=0.7, predicted_label=1),
        ]

        spans = tokens_to_spans(tokens, threshold=0.5)

        assert len(spans) == 1
        assert spans[0]["start"] == 0
        assert spans[0]["end"] == 11
        assert spans[0]["text"] == "Hello World"

    def test_separate_hallucinated_spans(self):
        tokens = [
            TokenPrediction(text="Hello", start=0, end=5, p_hallucination=0.9, predicted_label=1),
            TokenPrediction(text=" ", start=5, end=6, p_hallucination=0.1, predicted_label=0),
            TokenPrediction(text="World", start=6, end=11, p_hallucination=0.8, predicted_label=1),
        ]

        spans = tokens_to_spans(tokens, threshold=0.5)

        assert len(spans) == 2
        assert spans[0]["start"] == 0
        assert spans[0]["end"] == 5
        assert spans[1]["start"] == 6
        assert spans[1]["end"] == 11

    def test_merge_gap(self):
        tokens = [
            TokenPrediction(text="Hello", start=0, end=5, p_hallucination=0.9, predicted_label=1),
            TokenPrediction(text=" ", start=5, end=6, p_hallucination=0.1, predicted_label=0),
            TokenPrediction(text="World", start=6, end=11, p_hallucination=0.8, predicted_label=1),
        ]

        spans = tokens_to_spans(tokens, threshold=0.5, merge_gap=2)

        assert len(spans) == 1
        assert spans[0]["start"] == 0
        assert spans[0]["end"] == 11

    def test_score_max_and_mean(self):
        # "Hello" = positions 0-5, "World" = positions 5-10 (no space)
        tokens = [
            TokenPrediction(text="Hello", start=0, end=5, p_hallucination=0.6, predicted_label=1),
            TokenPrediction(text="World", start=5, end=10, p_hallucination=0.9, predicted_label=1),
        ]

        spans = tokens_to_spans(tokens, threshold=0.5)

        assert len(spans) == 1  # Both tokens are hallucinated and adjacent
        assert spans[0]["score_max"] == 0.9


class TestTokensToSpansRussianAndASCII:
    """Test token to span conversion with Unicode (Russian) and ASCII."""

    def test_russian_percentage_span(self):
        """Test Russian text with percentage - matches the 10% span."""
        # Answer: "Ставка составляет 10% годовых."
        # Gold span: "10%" at positions 18-21
        answer = "Ставка составляет 10% годовых."
        span_start = 18
        span_end = 21  # "10%"
        span_text = answer[span_start:span_end]

        tokens = [
            TokenPrediction(text="Ста", start=0, end=2, p_hallucination=0.1, predicted_label=0),
            TokenPrediction(text="вка", start=2, end=5, p_hallucination=0.1, predicted_label=0),
            TokenPrediction(text=" со", start=5, end=7, p_hallucination=0.1, predicted_label=0),
            TokenPrediction(text="ст", start=7, end=9, p_hallucination=0.1, predicted_label=0),
            TokenPrediction(text="а", start=9, end=10, p_hallucination=0.1, predicted_label=0),
            TokenPrediction(text="вл", start=10, end=12, p_hallucination=0.1, predicted_label=0),
            TokenPrediction(text="и", start=12, end=13, p_hallucination=0.1, predicted_label=0),
            TokenPrediction(text="ва", start=13, end=15, p_hallucination=0.1, predicted_label=0),
            TokenPrediction(text="ет ", start=15, end=18, p_hallucination=0.1, predicted_label=0),
            TokenPrediction(text="10", start=18, end=20, p_hallucination=0.9, predicted_label=1),
            TokenPrediction(text="%", start=20, end=21, p_hallucination=0.9, predicted_label=1),
            TokenPrediction(text=" го", start=21, end=24, p_hallucination=0.1, predicted_label=0),
            TokenPrediction(text="до", start=24, end=26, p_hallucination=0.1, predicted_label=0),
            TokenPrediction(text="вы", start=26, end=28, p_hallucination=0.1, predicted_label=0),
            TokenPrediction(text="х.", start=28, end=30, p_hallucination=0.1, predicted_label=0),
        ]

        spans = tokens_to_spans(tokens, threshold=0.5)

        # Should find the "10%" span
        assert len(spans) == 1
        assert spans[0]["start"] == span_start
        assert spans[0]["end"] == span_end
        assert spans[0]["text"] == span_text
        assert answer[spans[0]["start"]:spans[0]["end"]] == spans[0]["text"]

    def test_ascii_with_percentage(self):
        """Test ASCII text with percentage."""
        answer = "The rate is 15% per year."
        # "15%" at positions 12-15
        span_start = 12
        span_end = 15
        span_text = answer[span_start:span_end]

        tokens = [
            TokenPrediction(text="The", start=0, end=3, p_hallucination=0.1, predicted_label=0),
            TokenPrediction(text=" rate", start=3, end=8, p_hallucination=0.1, predicted_label=0),
            TokenPrediction(text=" is ", start=8, end=12, p_hallucination=0.1, predicted_label=0),
            TokenPrediction(text="15", start=12, end=14, p_hallucination=0.9, predicted_label=1),
            TokenPrediction(text="%", start=14, end=15, p_hallucination=0.9, predicted_label=1),
            TokenPrediction(text=" per", start=15, end=19, p_hallucination=0.1, predicted_label=0),
            TokenPrediction(text=" year", start=19, end=24, p_hallucination=0.1, predicted_label=0),
            TokenPrediction(text=".", start=24, end=25, p_hallucination=0.1, predicted_label=0),
        ]

        spans = tokens_to_spans(tokens, threshold=0.5)

        assert len(spans) == 1
        assert spans[0]["start"] == span_start
        assert spans[0]["end"] == span_end
        assert spans[0]["text"] == span_text
        assert answer[spans[0]["start"]:spans[0]["end"]] == spans[0]["text"]

    def test_adjacent_positive_tokens(self):
        """Test two adjacent positive tokens are merged."""
        tokens = [
            TokenPrediction(text="Hello", start=0, end=5, p_hallucination=0.9, predicted_label=1),
            TokenPrediction(text=" ", start=5, end=6, p_hallucination=0.9, predicted_label=1),
            TokenPrediction(text="World", start=6, end=11, p_hallucination=0.9, predicted_label=1),
            TokenPrediction(text="!", start=11, end=12, p_hallucination=0.1, predicted_label=0),
        ]

        spans = tokens_to_spans(tokens, threshold=0.5)

        assert len(spans) == 1
        assert spans[0]["start"] == 0
        assert spans[0]["end"] == 11
        assert spans[0]["text"] == "Hello World"

    def test_positive_token_with_original_space(self):
        """Test that spaces between positive tokens are preserved."""
        answer = "Hello World"
        tokens = [
            TokenPrediction(text="Hello", start=0, end=5, p_hallucination=0.9, predicted_label=1),
            TokenPrediction(text=" ", start=5, end=6, p_hallucination=0.9, predicted_label=1),
            TokenPrediction(text="World", start=6, end=11, p_hallucination=0.9, predicted_label=1),
        ]

        spans = tokens_to_spans(tokens, threshold=0.5)

        assert len(spans) == 1
        assert spans[0]["text"] == "Hello World"
        assert answer[spans[0]["start"]:spans[0]["end"]] == spans[0]["text"]

    def test_punctuation_not_swallowed(self):
        """Test that punctuation is not swallowed."""
        answer = "Hello, World!"
        tokens = [
            TokenPrediction(text="Hello", start=0, end=5, p_hallucination=0.9, predicted_label=1),
            TokenPrediction(text=",", start=5, end=6, p_hallucination=0.9, predicted_label=1),
            TokenPrediction(text=" ", start=6, end=7, p_hallucination=0.1, predicted_label=0),
            TokenPrediction(text="World", start=7, end=12, p_hallucination=0.9, predicted_label=1),
            TokenPrediction(text="!", start=12, end=13, p_hallucination=0.9, predicted_label=1),
        ]

        spans = tokens_to_spans(tokens, threshold=0.5)

        assert len(spans) == 2
        assert spans[0]["start"] == 0
        assert spans[0]["end"] == 6
        assert spans[0]["text"] == "Hello,"
        assert spans[1]["start"] == 7
        assert spans[1]["end"] == 13
        assert spans[1]["text"] == "World!"

    def test_boundary_touch_not_overlap(self):
        """Test that boundary touch (end == start) is NOT considered overlap."""
        # Token at [0,5) and gold span at [5,10) - they touch but don't overlap
        tokens = [
            TokenPrediction(text="Hello", start=0, end=5, p_hallucination=0.9, predicted_label=1),
            TokenPrediction(text="World", start=5, end=10, p_hallucination=0.1, predicted_label=0),
        ]

        # With threshold 0.5, first token is hallucinated, second is not
        spans = tokens_to_spans(tokens, threshold=0.5)

        # Should only find span [0, 5)
        assert len(spans) == 1
        assert spans[0]["start"] == 0
        assert spans[0]["end"] == 5

    def test_multiple_independent_spans(self):
        """Test multiple independent hallucination spans."""
        answer = "The quick brown fox jumps."
        tokens = [
            TokenPrediction(text="The", start=0, end=3, p_hallucination=0.1, predicted_label=0),
            TokenPrediction(text=" quick", start=3, end=9, p_hallucination=0.9, predicted_label=1),
            TokenPrediction(text=" brown", start=9, end=15, p_hallucination=0.9, predicted_label=1),
            TokenPrediction(text=" fox", start=15, end=19, p_hallucination=0.1, predicted_label=0),
            TokenPrediction(text=" jumps", start=19, end=25, p_hallucination=0.9, predicted_label=1),
            TokenPrediction(text=".", start=25, end=26, p_hallucination=0.1, predicted_label=0),
        ]

        spans = tokens_to_spans(tokens, threshold=0.5)

        # Should find 2 spans: " quick brown" and " jumps"
        assert len(spans) == 2
        assert spans[0]["start"] == 3
        assert spans[0]["end"] == 15
        assert spans[0]["text"] == " quick brown"
        assert spans[1]["start"] == 19
        assert spans[1]["end"] == 25
        assert spans[1]["text"] == " jumps"
        # Verify text matches original offsets
        assert answer[spans[0]["start"]:spans[0]["end"]] == spans[0]["text"]
        assert answer[spans[1]["start"]:spans[1]["end"]] == spans[1]["text"]

    def test_unicode_cyrillic(self):
        """Test Russian Cyrillic characters."""
        answer = "Привет мир"
        tokens = [
            TokenPrediction(text="При", start=0, end=3, p_hallucination=0.1, predicted_label=0),
            TokenPrediction(text="вет", start=3, end=6, p_hallucination=0.9, predicted_label=1),
            TokenPrediction(text=" ", start=6, end=7, p_hallucination=0.1, predicted_label=0),
            TokenPrediction(text="мир", start=7, end=10, p_hallucination=0.9, predicted_label=1),
        ]

        spans = tokens_to_spans(tokens, threshold=0.5)

        assert len(spans) == 2
        assert spans[0]["start"] == 3
        assert spans[0]["end"] == 6
        assert spans[0]["text"] == "вет"
        assert spans[1]["start"] == 7
        assert spans[1]["end"] == 10
        assert spans[1]["text"] == "мир"
        # Verify text matches
        assert answer[spans[0]["start"]:spans[0]["end"]] == spans[0]["text"]
        assert answer[spans[1]["start"]:spans[1]["end"]] == spans[1]["text"]

    def test_answer_span_text_consistency(self):
        """Test that answer[start:end] always equals span['text']."""
        answer = "Paris is the capital of France"
        tokens = [
            TokenPrediction(text="Paris", start=0, end=5, p_hallucination=0.9, predicted_label=1),
            TokenPrediction(text=" is", start=5, end=8, p_hallucination=0.1, predicted_label=0),
            TokenPrediction(text=" the", start=8, end=12, p_hallucination=0.1, predicted_label=0),
            TokenPrediction(text=" cap", start=12, end=16, p_hallucination=0.9, predicted_label=1),
            TokenPrediction(text="ital", start=16, end=20, p_hallucination=0.9, predicted_label=1),
            TokenPrediction(text=" of", start=20, end=23, p_hallucination=0.1, predicted_label=0),
            TokenPrediction(text=" France", start=23, end=30, p_hallucination=0.9, predicted_label=1),
        ]

        spans = tokens_to_spans(tokens, threshold=0.5)

        for span in spans:
            extracted = answer[span["start"]:span["end"]]
            assert extracted == span["text"], f"Mismatch: answer[{span['start']}:{span['end']}]='{extracted}' != span['text']='{span['text']}'"


class TestComputeAnswerScore:
    """Test answer-level scoring."""
    
    def test_max_score(self):
        probs = [0.1, 0.9, 0.3, 0.7]
        score = compute_answer_score(probs, mode="max")
        assert score == 0.9
    
    def test_noisy_or(self):
        probs = [0.1, 0.2, 0.3]
        score = compute_answer_score(probs, mode="noisy_or")
        expected = 1 - (0.9 * 0.8 * 0.7)
        assert abs(score - expected) < 0.01
    
    def test_ratio(self):
        probs = [0.1, 0.6, 0.3, 0.7]
        score = compute_answer_score(probs, mode="ratio")
        assert score == 0.5  # 2 out of 4 above 0.5
    
    def test_empty_probs(self):
        score = compute_answer_score([], mode="max")
        assert score == 0.0


class TestPredictAnswerHallucination:
    """Test answer-level prediction."""

    def test_any_mode(self):
        # With threshold=0.5, [0.1, 0.6, 0.2] has 1 token >= 0.5, so any=True
        probs = [0.1, 0.6, 0.2]
        result = predict_answer_hallucination(probs, threshold=0.5, mode="any")
        assert result == True  # numpy.bool_ compatibility

    def test_any_mode_all_low(self):
        probs = [0.1, 0.2, 0.3]
        assert predict_answer_hallucination(probs, threshold=0.5, mode="any") == False

    def test_majority_mode(self):
        probs = [0.6, 0.7, 0.5]
        assert predict_answer_hallucination(probs, threshold=0.5, mode="majority") == True

    def test_majority_mode_not_enough(self):
        probs = [0.6, 0.3, 0.2]
        assert predict_answer_hallucination(probs, threshold=0.5, mode="majority") == False

    def test_all_mode(self):
        probs = [0.6, 0.7, 0.8]
        assert predict_answer_hallucination(probs, threshold=0.5, mode="all") == True

    def test_all_mode_one_low(self):
        probs = [0.6, 0.3, 0.8]
        assert predict_answer_hallucination(probs, threshold=0.5, mode="all") == False
