# Phase 2 Plan: Faithfulness Encoder Fine-tuning

**Status**: NOT STARTED
**Prerequisites**: Relevance gold labels acquisition (see Section E)

---

## Overview

This document outlines the experimental plan for Phase 2 of the Faithfulness Encoder
project. Phase 2 focuses on **fine-tuning** an Encoder model for claim-level
Faithfulness detection.

**IMPORTANT**: This plan covers Faithfulness-only experiments. Full Reliability
evaluation requires Relevance gold labels, which are currently unavailable.

---

## A. Faithfulness Token-level Encoder: Data Preprocessing

### A.1 Data Source

Use RAGognize hallucination spans for Token-level Faithfulness supervision.

### A.2 Span-to-Token Conversion

```python
# Input: Hallucination spans (character positions)
hallucination_spans = [
    {"start": 10, "end": 25, "text": "hallucinated text"},
    {"start": 50, "end": 65, "text": "more hallucination"},
]

# Output: Token-level labels
# For each token in answer:
#   - Label 0: hallucinated token
#   - Label 1: faithful token
```

### A.3 Tokenization Strategy

```python
# Use mDeBERTa tokenizer
tokenizer = AutoTokenizer.from_pretrained("MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")

# For each answer:
# 1. Tokenize answer
tokens = tokenizer.tokenize(answer, add_special_tokens=False)

# 2. Convert character spans to token positions
# 3. Assign labels: 0 = hallucinated, 1 = faithful
token_labels = span_to_token_labels(hallucination_spans, tokens, answer)
```

### A.4 Training Format

```python
{
    "input_ids": [...],           # [CLS] premise [SEP] hypothesis [SEP]
    "attention_mask": [...],
    "labels": [...],              # Token-level: 0=hallucinated, 1=faithful
    "case_id": "case_abc123",
    "source_model": "Llama-2-7b-chat-hf",
}
```

---

## B. Character Span to Token Label Conversion

### B.1 Algorithm

```python
def char_span_to_token_labels(
    text: str,
    spans: list[dict],  # [{"start": int, "end": int}, ...]
    tokenizer,
) -> list[int]:
    """
    Convert character-level hallucination spans to token-level labels.
    
    Args:
        text: Original answer text
        spans: Hallucination spans as character positions
        tokenizer: Tokenizer for word-to-token alignment
    
    Returns:
        List of token-level labels (0=hallucinated, 1=faithful)
    """
    # Tokenize with word-level alignment
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    
    # Initialize all tokens as faithful (1)
    token_labels = [1] * len(encoding.word_ids())
    
    # Mark hallucinated tokens as 0
    for span in spans:
        start_char = span["start"]
        end_char = span["end"]
        
        for idx, (token_start, token_end) in enumerate(encoding.offset_mapping):
            if token_start >= start_char and token_end <= end_char:
                token_labels[idx] = 0
    
    return token_labels
```

### B.2 Edge Cases

| Case | Handling |
|------|----------|
| Empty hallucination spans | All tokens = 1 (faithful) |
| Partial token overlap | Mark token as hallucinated (0) |
| Spans outside text | Skip, log warning |
| Empty answer | Skip sample, log |

---

## C. Train/Validation Split and Leakage Control

### C.1 Split Strategy

```python
# IMPORTANT: Split at question level, NOT at response level
# Same question asked to different models should NOT appear in both splits

SPLIT_CONFIG = {
    "val_size": 0.15,
    "seed": 42,
    "stratify_by": ["source_model", "faithfulness_label"],
}

# Split by user_prompt_index (question ID)
# All responses to the same question go to the same split
```

### C.2 Leakage Prevention

| Leakage Type | Prevention |
|--------------|------------|
| Same question in train/val | Split by question ID |
| Same model in train/val | Stratified split across models |
| Duplicate case_ids | Verify uniqueness |
| Shared context | Ensure context chunks don't overlap |

### C.3 Verification

```python
def verify_no_leakage(train_df, val_df):
    """Verify train/val split has no leakage."""
    
    # Check question-level separation
    train_questions = set(train_df["user_prompt_index"])
    val_questions = set(val_df["user_prompt_index"])
    assert len(train_questions & val_questions) == 0, "Question overlap detected!"
    
    # Check case_id uniqueness
    all_ids = list(train_df["case_id"]) + list(val_df["case_id"])
    assert len(all_ids) == len(set(all_ids)), "Duplicate case_ids!"
```

---

## D. Zero-shot Baseline vs Fine-tuned Model Comparison

### D.1 Experiments

| Experiment | Description | Metric |
|-----------|-------------|--------|
| **D.1.1** | Zero-shot NLI (current) | Macro-F1 = 0.488 |
| **D.1.2** | Claim-level NLI fine-tune | Target: > 0.55 |
| **D.1.3** | Token-level span detection | Target: > 0.60 |
| **D.1.4** | Joint claim + span | Target: > 0.65 |

### D.2 Model Options

| Model | Size | Notes |
|-------|------|-------|
| mDeBERTa-v3-base-mnli-xnli | 278M | Current zero-shot baseline |
| mDeBERTa-v3-large-mnli-xnli | 434M | Larger variant |
| DeBERTa-v3-base | 86M | Smaller, faster |
| XLM-RoBERTa-base | 278M | Multilingual |

### D.3 Training Configuration

```python
TRAINING_CONFIG = {
    "epochs": 3,
    "batch_size": 16,
    "learning_rate": 2e-5,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "max_seq_length": 512,
    "gradient_accumulation_steps": 2,
}

# For token classification:
LOSS_CONFIG = {
    "type": "cross_entropy",
    "ignore_index": -100,  # For padding tokens
    "class_weights": [1.0, 1.0],  # Balance hallucinated vs faithful
}
```

### D.4 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Macro-F1 | Primary metric |
| Balanced Accuracy | Class balance |
| AUROC | Ranking quality |
| Per-class P/R/F | Unfaithful, Faithful |
| Confusion Matrix | TN, FP, FN, TP |

---

## E. Relevance Data Source Options

### E.1 Why Relevance is Needed

Current evaluation computes **Faithfulness only**.
Full **Reliability = Faithfulness AND Relevance** requires Relevance labels.

### E.2 Options

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| **E.2.1** | RAGognize `addressed_user_prompt` | Native to dataset | Currently unavailable (0/1100) |
| **E.2.2** | Russian Banking Dataset | Has relevance annotations | Different domain |
| **E.2.3** | Manual annotation | Custom labels | Expensive, time-consuming |
| **E.2.4** | Synthetic generation | Unlimited data | May not match real distribution |

### E.3 Recommended Path

**Priority 1**: Check RAGognize updates
- Monitor for new dataset version with `addressed_user_prompt`
- Check HuggingFace Hub periodically

**Priority 2**: Use Russian Banking Dataset
- Has explicit `binary_relevancy` labels
- Different domain but validates approach
- Can establish Relevance methodology

**Priority 3**: Partial annotation
- Annotate subset of RAGognize for Relevance
- 100-200 samples sufficient for methodology validation

### E.4 Semantic Mapping

```python
# If using answerable as proxy:
RELEVANCE_MAPPING = {
    "answerable=True": "relevant",    # Question can be answered
    "answerable=False": "irrelevant"  # Question cannot be answered
}

# NOT the same as:
# addressed_user_prompt = Did the answer address the question?
```

---

## F. When Can We Compute Reliability?

### F.1 Requirements

| Component | Status | Notes |
|-----------|--------|-------|
| Faithfulness NLI | ✅ Available | Zero-shot baseline done |
| Faithfulness fine-tune | ⏳ Planned | Phase 2 |
| Relevance gold labels | ❌ Unavailable | Must acquire |
| Reliability formula | ✅ Defined | Faithful AND Relevant |

### F.2 Decision Tree

```
                    ┌─────────────────────────────┐
                    │ Need Reliability evaluation?│
                    └─────────────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │ YES                              │ NO
                    ▼                                  ▼
    ┌───────────────────────────┐         ┌─────────────────────────┐
    │ Acquire Relevance labels │         │ Faithfulness-only OK    │
    │ (see Section E)         │         │ Continue Phase 2        │
    └───────────────────────────┘         └─────────────────────────┘
                    │
        ┌───────────┴───────────┐
        │ Labels acquired?       │
        └───────────┬───────────┘
               YES   │   NO
                ▼         ▼
    ┌─────────────────┐   ┌─────────────────────┐
    │ Compute         │   │ Report Faithfulness  │
    │ Reliability     │   │ only, note Limitation│
    └─────────────────┘   └─────────────────────┘
```

### F.3 Timeline

| Phase | Task | Status |
|-------|------|--------|
| 1 | Zero-shot Faithfulness baseline | ✅ Complete |
| 2a | Faithfulness fine-tuning prep | 📋 This document |
| 2b | Relevance data acquisition | ⏳ Awaiting |
| 3 | Reliability evaluation | ⏳ Awaiting 2b |

---

## G. Summary: Phase 2 Experiments

### G.1 Faithfulness-only Track

| # | Experiment | Status | Target |
|---|------------|--------|--------|
| 1 | Token-level data preprocessing | Planned | Pipeline ready |
| 2 | Span-to-token conversion | Planned | >95% coverage |
| 3 | Fine-tune mDeBERTa (claim-level) | Planned | Macro-F1 > 0.55 |
| 4 | Fine-tune mDeBERTa (token-level) | Planned | Macro-F1 > 0.60 |
| 5 | Compare zero-shot vs fine-tuned | Planned | Document improvement |

### G.2 Prerequisites Check

| Prerequisite | Status | Action |
|--------------|--------|--------|
| Data pipeline | Ready | Implement Section A |
| Train/val split | Ready | Reuse Phase 1 splits |
| Tokenization | Ready | mDeBERTa tokenizer |
| Evaluation metrics | Ready | Use Phase 1 metrics |
| Relevance data | ❌ Not ready | See Section E |

### G.3 What Phase 2 Will NOT Include

- Full Reliability evaluation (awaiting Relevance data)
- LoRA fine-tuning (full fine-tune first)
- External LLM API calls
- SelfCheckGPT comparison

---

## H. Next Actions

### Immediate (Before Training)

1. [ ] Verify hallucination span coverage in dataset
2. [ ] Implement span-to-token conversion (Section B)
3. [ ] Create token-level training dataset
4. [ ] Verify no data leakage in splits (Section C)
5. [ ] Set up training infrastructure

### Medium-term (Before Reliability)

1. [ ] Monitor RAGognize for Relevance updates
2. [ ] Consider Russian Banking Dataset adapter
3. [ ] Design Relevance annotation schema
4. [ ] Acquire/verify Relevance labels

### Long-term (After Relevance)

1. [ ] Compute full Reliability = Faithful AND Relevant
2. [ ] Compare Faithfulness-only vs full Reliability
3. [ ] Joint training for both tasks

---

## I. Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-07-20 | Initial plan |

---

**End of Phase 2 Plan**
