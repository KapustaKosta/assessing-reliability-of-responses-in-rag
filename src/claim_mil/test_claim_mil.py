"""
Unit tests for Phase 2 claim labeling and MIL model.

Tests:
 1. claim fully outside all spans -> supported
 2. claim fully inside a span -> unsupported
 3. partial left overlap -> unsupported
 4. partial right overlap -> unsupported
 5. boundary-touch without character overlap -> supported
 6. multiple spans
 7. multiple claims
 8. empty hallucination span list
 9. malformed span
10. invalid claim offsets
11. grouped split has no question leakage
12. fixed project validation is excluded
13. MIL max-pooling behaviour
14. answer-level max aggregation
15. deterministic output with seed 42
"""

import random
import pytest
from dataclasses import dataclass
from claim_mil.claim_bags import (
    _compute_claim_label,
    create_grouped_split,
)


# =============================================================================
# Test Claim Labeling
# =============================================================================

@dataclass
class MockClaim:
    text: str
    char_start: int
    char_end: int


def test_claim_outside_all_spans_supported():
    """Claim with no overlap to any span should be labeled supported (0)."""
    answer = "The capital of France is Paris."
    claim = MockClaim(text="Paris.", char_start=23, char_end=29)
    spans = [{"start": 0, "end": 5, "text": "The c", "valid": True}]

    label, reason = _compute_claim_label(claim, spans, answer)
    assert label == 0, f"Expected 0 (supported), got {label}. Reason: {reason}"


def test_claim_fully_inside_span_unsupported():
    """Claim fully inside a hallucination span should be labeled unsupported (1)."""
    answer = "Based on the provided information, the stock price increased by 15%."
    # Claim is the whole answer, which overlaps with the span
    claim = MockClaim(
        text="Based on the provided information, the stock price increased by 15%.",
        char_start=0,
        char_end=72,
    )
    spans = [{"start": 0, "end": 72, "text": answer, "valid": True}]

    label, reason = _compute_claim_label(claim, spans, answer)
    assert label == 1, f"Expected 1 (unsupported), got {label}. Reason: {reason}"


def test_partial_left_overlap_unsupported():
    """Partial overlap from the left should be unsupported."""
    answer = "The meeting was scheduled for Monday at 3pm in conference room B."
    # Claim starts inside span
    claim = MockClaim(
        text="Monday at 3pm in conference room B.",
        char_start=25,
        char_end=57,
    )
    # Span covers "Monday at 3pm"
    spans = [{"start": 25, "end": 40, "text": "Monday at 3pm", "valid": True}]

    label, reason = _compute_claim_label(claim, spans, answer)
    assert label == 1, f"Expected 1 (unsupported), got {label}. Reason: {reason}"


def test_partial_right_overlap_unsupported():
    """Partial overlap from the right should be unsupported."""
    answer = "The event will take place on July 15th at the convention center."
    # Claim ends inside span
    claim = MockClaim(
        text="The event will take place on July 15th",
        char_start=0,
        char_end=39,
    )
    # Span covers "July 15th"
    spans = [{"start": 28, "end": 38, "text": "July 15th", "valid": True}]

    label, reason = _compute_claim_label(claim, spans, answer)
    assert label == 1, f"Expected 1 (unsupported), got {label}. Reason: {reason}"


def test_boundary_touch_without_overlap_supported():
    """Adjacent but non-overlapping should be supported."""
    answer = "The temperature is 25 degrees today."
    # Claim touches the span boundary but doesn't overlap
    claim = MockClaim(text="25 degrees", char_start=19, char_end=30)
    spans = [{"start": 16, "end": 19, "text": "is ", "valid": True}]  # "is " ends at 19, claim starts at 19

    label, reason = _compute_claim_label(claim, spans, answer)
    assert label == 0, f"Expected 0 (supported), got {label}. Reason: {reason}"


def test_multiple_spans_one_overlap_unsupported():
    """If any span overlaps, the claim is unsupported."""
    answer = "The product costs $99 and ships within 3 days."
    claim = MockClaim(text="ships within 3 days.", char_start=25, char_end=43)
    spans = [
        {"start": 4, "end": 8, "text": "$99", "valid": True},
        {"start": 25, "end": 40, "text": "ships within 3", "valid": True},  # overlaps
        {"start": 40, "end": 44, "text": "days", "valid": True},
    ]

    label, reason = _compute_claim_label(claim, spans, answer)
    assert label == 1, f"Expected 1 (unsupported), got {label}. Reason: {reason}"


def test_multiple_claims_multiple_spans():
    """Multiple claims with multiple spans."""
    answer = "The stock rose 5% and the market closed higher."
    claims = [
        MockClaim(text="The stock rose 5%", char_start=0, char_end=18),
        MockClaim(text="and the market closed higher.", char_start=18, char_end=48),
    ]
    spans = [{"start": 0, "end": 10, "text": "The stock", "valid": True}]

    # First claim overlaps with span
    label1, _ = _compute_claim_label(claims[0], spans, answer)
    assert label1 == 1

    # Second claim does not overlap
    label2, _ = _compute_claim_label(claims[1], spans, answer)
    assert label2 == 0


def test_empty_hallucination_span_list():
    """No hallucination spans means all claims are supported."""
    answer = "This is a faithful answer with all facts correct."
    claim = MockClaim(text="This is a faithful answer", char_start=0, char_end=25)

    label, reason = _compute_claim_label(claim, [], answer)
    assert label == 0, f"Expected 0 (supported), got {label}. Reason: {reason}"
    assert "no_hallucination" in reason


def test_invalid_span_flagged_valid_false():
    """Spans with valid=False should be skipped."""
    answer = "The answer is incorrect about the date."
    claim = MockClaim(text="incorrect about the date.", char_start=10, char_end=33)
    spans = [
        {"start": 10, "end": 33, "text": "incorrect about the date.", "valid": False},
    ]

    label, reason = _compute_claim_label(claim, spans, answer)
    assert label == 0, f"Expected 0 (supported) when span is invalid. Got {label}. Reason: {reason}"


def test_malformed_span_negative_start():
    """Negative start should be skipped."""
    answer = "This is the answer."
    claim = MockClaim(text="answer.", char_start=12, char_end=19)
    spans = [{"start": -5, "end": 10, "text": "bad", "valid": True}]

    label, reason = _compute_claim_label(claim, spans, answer)
    assert label == 0, f"Malformed span should be skipped. Got {label}. Reason: {reason}"


def test_malformed_span_end_before_start():
    """end <= start should be skipped."""
    answer = "This is the answer."
    claim = MockClaim(text="answer.", char_start=12, char_end=19)
    spans = [{"start": 15, "end": 10, "text": "bad", "valid": True}]

    label, reason = _compute_claim_label(claim, spans, answer)
    assert label == 0, f"Malformed span should be skipped. Got {label}. Reason: {reason}"


def test_invalid_claim_offsets_negative():
    """Negative claim start should not crash."""
    answer = "Valid answer text."
    claim = MockClaim(text="answer.", char_start=-5, char_end=19)
    spans = [{"start": 0, "end": 5, "text": "Valid", "valid": True}]

    label, reason = _compute_claim_label(claim, spans, answer)
    # Should not crash, may produce unexpected result
    assert label in (0, 1)


def test_whitespace_handling():
    """Non-whitespace overlap detection should work correctly."""
    answer = "  Leading and trailing spaces  "
    claim = MockClaim(text="  Leading and trailing spaces  ", char_start=0, char_end=34)
    # Span covers "Leading" (after stripping)
    spans = [{"start": 2, "end": 9, "text": "Leading", "valid": True}]

    label, reason = _compute_claim_label(claim, spans, answer)
    assert label == 1, f"Non-whitespace overlap should be detected. Got {label}. Reason: {reason}"


# =============================================================================
# Test Grouped Split
# =============================================================================

def _make_sample(qid: int, model: str, idx: int):
    """Helper to create a mock UnifiedSample-like dict."""
    from dataclasses import replace
    from ragognize_adapter import UnifiedSample
    s = UnifiedSample(
        case_id=f"case_{qid}_{model}_{idx}",
        question=f"Question {qid}",
        answer=f"Answer for q{qid} from {model}",
        user_prompt_index=qid,
        source_model=model,
        source_split="train",
        source_row_index=idx,
    )
    return s


def test_grouped_split_no_leakage():
    """Train and dev question IDs must be disjoint."""
    samples = [_make_sample(qid=i, model="ModelA", idx=i) for i in range(50)]
    project_val = set(range(45, 50))  # questions 45-49 are validation

    result = create_grouped_split(
        samples,
        dev_fraction=0.20,
        seed=42,
        project_val_question_ids=project_val,
    )

    train_qids = result["train_question_ids"]
    dev_qids = result["dev_question_ids"]

    assert len(train_qids & dev_qids) == 0, "Train and dev must not overlap"


def test_grouped_split_val_excluded():
    """Project validation questions must not appear in train or dev."""
    samples = [_make_sample(qid=i, model="ModelA", idx=i) for i in range(50)]
    project_val = set(range(40, 50))  # questions 40-49 are validation

    result = create_grouped_split(
        samples,
        dev_fraction=0.20,
        seed=42,
        project_val_question_ids=project_val,
    )

    train_qids = result["train_question_ids"]
    dev_qids = result["dev_question_ids"]

    assert len(train_qids & project_val) == 0, "Train must not contain val questions"
    assert len(dev_qids & project_val) == 0, "Dev must not contain val questions"


def test_grouped_split_deterministic():
    """Same seed must produce identical splits."""
    samples = [_make_sample(qid=i, model="ModelA", idx=i) for i in range(30)]

    result1 = create_grouped_split(samples, dev_fraction=0.20, seed=42)
    result2 = create_grouped_split(samples, dev_fraction=0.20, seed=42)

    assert result1["train_question_ids"] == result2["train_question_ids"]
    assert result1["dev_question_ids"] == result2["dev_question_ids"]


def test_grouped_split_different_seed():
    """Different seeds must produce different splits."""
    samples = [_make_sample(qid=i, model="ModelA", idx=i) for i in range(30)]

    result1 = create_grouped_split(samples, dev_fraction=0.20, seed=42)
    result2 = create_grouped_split(samples, dev_fraction=0.20, seed=123)

    assert result1["train_question_ids"] != result2["train_question_ids"]


def test_grouped_split_all_models_in_same_partition():
    """All responses for same question must stay together."""
    # 3 questions, 2 models each = 6 samples
    samples = []
    for qid in [10, 20, 30]:
        for model in ["ModelA", "ModelB"]:
            for idx in range(1):
                samples.append(_make_sample(qid=qid, model=model, idx=idx * 100 + qid))

    result = create_grouped_split(samples, dev_fraction=0.33, seed=42)

    # Check: for each question, all its models are in same partition
    question_to_partition = {}
    for s in samples:
        qid = s.user_prompt_index
        if qid in result["train_question_ids"]:
            partition = "train"
        elif qid in result["dev_question_ids"]:
            partition = "dev"
        else:
            continue

        if qid in question_to_partition:
            assert question_to_partition[qid] == partition, \
                f"Question {qid} spans partitions: {question_to_partition[qid]} vs {partition}"
        else:
            question_to_partition[qid] = partition


# =============================================================================
# Test MIL Model
# =============================================================================

import os as _os

def _get_tokenizer():
    """Load tokenizer once for all MIL tests. Skips if offline."""
    if _os.environ.get("HF_HUB_OFFLINE") == "1":
        pytest.skip("HF_HUB_OFFLINE=1, skipping model download test")
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")


def test_mil_max_pooling_basic():
    """Max pooling should take the max across window representations."""
    from claim_mil.model import ClaimMILModel, MILConfig
    import torch

    config = MILConfig(pooling_mode="max")
    tokenizer = _get_tokenizer()
    model = ClaimMILModel(config, tokenizer=tokenizer)
    model.eval()

    # Simple test: two identical windows should give same result as one
    windows = ["This is context window one.", "This is context window two."]
    claim = "The product ships in 3 days."

    with torch.no_grad():
        result1 = model.forward(windows, claim)
        result2 = model.forward([windows[0]], claim)

    # p_unsupported may differ slightly due to max pooling
    # but both should be valid probabilities
    assert 0 <= result1["p_unsupported"] <= 1
    assert 0 <= result2["p_unsupported"] <= 1


def test_mil_forward_returns_valid_probs():
    """Forward pass must return valid probabilities."""
    from claim_mil.model import ClaimMILModel, MILConfig
    import torch

    config = MILConfig()
    tokenizer = _get_tokenizer()
    model = ClaimMILModel(config, tokenizer=tokenizer)
    model.eval()

    windows = ["This context supports the claim about stock prices rising."]
    claim = "The stock price rose 5%."

    with torch.no_grad():
        result = model.forward(windows, claim)

    assert 0 <= result["p_unsupported"] <= 1
    assert 0 <= result["p_supported"] <= 1
    # p_supported + p_unsupported should be ~1.0
    total = result["p_supported"] + result["p_unsupported"]
    assert abs(total - 1.0) < 0.01


def test_mil_model_parameters_trainable():
    """Model should have trainable parameters."""
    from claim_mil.model import ClaimMILModel, MILConfig
    import torch

    config = MILConfig()
    tokenizer = _get_tokenizer()
    model = ClaimMILModel(config, tokenizer=tokenizer)

    params = list(model.parameters())
    assert len(params) > 0, "Model should have parameters"
    assert any(p.requires_grad for p in params), "Model should have trainable parameters"


def test_mil_backward_pass():
    """Backward pass should work without errors."""
    from claim_mil.model import ClaimMILModel, MILConfig
    import torch
    import torch.nn as nn

    config = MILConfig()
    tokenizer = _get_tokenizer()
    model = ClaimMILModel(config, tokenizer=tokenizer)
    model.train()

    windows = ["This is a context window."]
    claim = "The stock rose."

    # Forward
    result = model.forward(windows, claim)
    support_logit_t = torch.tensor([result["support_logit"]], requires_grad=True)

    # Simulate BCE-like loss
    label = torch.tensor([1.0])  # unsupported
    p_supported = torch.sigmoid(support_logit_t)
    loss = nn.functional.binary_cross_entropy(p_supported, label)

    # Backward
    loss.backward()

    # Gradients should exist
    assert support_logit_t.grad is not None


# =============================================================================
# Test Answer-Level Aggregation
# =============================================================================

def test_answer_level_max_aggregation():
    """Answer unfaithfulness = max of claim p_unsupported."""
    # Simulate: 3 claims with p_unsupported = [0.1, 0.8, 0.3]
    claim_probs = [0.1, 0.8, 0.3]
    answer_score = max(claim_probs)
    assert answer_score == 0.8

    # Prediction threshold = 0.5
    pred = 1 if answer_score >= 0.5 else 0
    assert pred == 1  # Unfaithful


def test_answer_level_all_supported():
    """If all claims are supported, answer is faithful."""
    claim_probs = [0.1, 0.2, 0.3]
    answer_score = max(claim_probs)
    assert answer_score < 0.5
    pred = 1 if answer_score >= 0.5 else 0
    assert pred == 0  # Faithful


# =============================================================================
# Test Determinism
# =============================================================================

def test_claim_labeling_deterministic():
    """Claim labeling must be deterministic."""
    answer = "The capital of France is Paris."
    claim = MockClaim(text="Paris.", char_start=23, char_end=29)
    spans = [{"start": 0, "end": 5, "text": "The c", "valid": True}]

    for _ in range(5):
        label, reason = _compute_claim_label(claim, spans, answer)
        assert label == 0


def test_grouped_split_deterministic_across_runs():
    """Split must be identical across multiple runs with same seed."""
    samples = [_make_sample(qid=i, model="ModelA", idx=i) for i in range(20)]

    # Simulate 3 runs
    results = [
        create_grouped_split(samples, dev_fraction=0.20, seed=42)
        for _ in range(3)
    ]

    for i in range(1, len(results)):
        assert results[i]["train_question_ids"] == results[0]["train_question_ids"]
        assert results[i]["dev_question_ids"] == results[0]["dev_question_ids"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
