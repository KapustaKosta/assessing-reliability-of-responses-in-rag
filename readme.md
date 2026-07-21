# Zero-shot Claim-level NLI Faithfulness Baseline

**NOT a full Reliability evaluation. This is a Faithfulness-only baseline.**

## Project Name

**Zero-shot Claim-level NLI Faithfulness Baseline on RAGognize Validation**

## Current Evaluation Scope

| Component | Status |
|-----------|--------|
| **Faithfulness NLI** | ✅ Complete |
| **Relevance Evaluation** | ❌ NOT AVAILABLE |
| **Reliability** | ❌ NOT AVAILABLE |

> **Important**: This project does NOT currently support full Reliability evaluation
> because `addressed_user_prompt` (the Relevance gold label) is not available in the
> RAGognize dataset version being used.

## Label Semantics

| Label | Meaning | Description |
|-------|---------|-------------|
| `0` | Unfaithful | Hallucination present in answer |
| `1` | Faithful | No hallucination, answer supported by context |

## Dataset: RAGognize (F4biian/RAGognize)

| Property | Value |
|----------|-------|
| Raw train split | 1,842 questions |
| Raw test split | 2,781 questions |
| Models per question | 4 |
| Validation questions | 277 (15% of train) |
| Theoretical max validation responses | 277 × 4 = 1,108 |
| Source data missing responses | 8 |
| **Actual valid samples** | **1,100** |
| Runtime silent skipped | 0 |

### Sample Count Explanation

- **277 validation questions** × 4 models = 1,108 theoretical responses
- **8 model responses missing** from source data (some questions have < 4 models)
- **1,100 valid samples** = 1,108 - 8 = 1,100
- **0 runtime silent skipped** = all available samples were processed

## Relevance Evaluation Status

| Field | Availability |
|-------|-------------|
| `addressed_user_prompt` | **0 / 1,100** (NOT AVAILABLE) |
| `answerable` | Available (but different semantics) |

**Conclusion**: Relevance formal evaluation is NOT possible with current data.

## Architecture

```
Question + Retrieved Context + Answer
    │
    ├─→ Answer Split into Claims
    │       │
    │       └─→ Claim 1, Claim 2, ..., Claim N
    │
    ├─→ Long Context → Overlapping Windows
    │       │
    │       └─→ Window 1, Window 2, ..., Window M
    │
    └─→ Faithfulness NLI (Current Scope)
            │
            └─→ For each (Claim, Window) pair:
                    premise = context window
                    hypothesis = answer claim
                    → entailment / neutral / contradiction
                    → aggregate → faithfulness prediction

    [Relevance NLI - NOT IMPLEMENTED - awaiting gold labels]
    
    [Reliability = Faithful AND Relevant - NOT COMPUTED]
```

## Model

**MoritzLaurer/mDeBERTa-v3-base-mnli-xnli**
- Multilingual (100+ languages, including English)
- Zero-shot NLI (no fine-tuning in Phase 1)
- Label indices read dynamically from `model.config.id2label`

## Aggregation Strategies

| Strategy | Formula |
|----------|---------|
| `max_entail` | score = max(p_entailment) |
| `entail_minus_contradiction` | score = max(p_entail) - max(p_contrad) |
| `claim_min_support` | score = min(claim_scores) |
| `contradiction_penalized_support` | score = min(p_entail) - penalty × max(p_contrad) |

## Validation Results (Zero-shot mDeBERTa)

### Overall Metrics

| Metric | Value |
|--------|-------|
| **Accuracy** | 0.4955 |
| **Balanced Accuracy** | 0.5518 |
| **Macro-F1** | 0.4880 |
| **AUROC** | ~0.56 |

### Per-Class Metrics

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Unfaithful (0) | 0.311 | 0.678 | 0.426 |
| Faithful (1) | 0.776 | 0.426 | 0.550 |

### Confusion Matrix

```
                 Predicted
              unfaithful  faithful
Actual unfaithful    206       98
        faithful    457      339
```

### Baseline Comparisons

| Baseline | Accuracy | Balanced Acc | Macro-F1 |
|----------|----------|-------------|----------|
| Always Faithful | 0.724 | 0.500 | 0.418 |
| Always Unfaithful | 0.276 | 0.500 | 0.418 |
| Stratified Random | ~0.724 | ~0.500 | ~0.418 |
| Majority (Faithful) | 0.724 | 0.500 | 0.418 |
| **NLI (best)** | **0.496** | **0.552** | **0.488** |

### By Source Model

| Model | N | Unfaithful | Acc | Macro-F1 |
|-------|---|------------|-----|----------|
| Llama-2-7b-chat-hf | 275 | 142 | 0.564 | 0.524 |
| Llama-3.1-8B-Instruct | 275 | 15 | 0.295 | 0.267 |
| Mistral-7B-Instruct-v0.1 | 275 | 99 | 0.487 | 0.478 |
| Mistral-7B-Instruct-v0.3 | 275 | 48 | 0.364 | 0.359 |

## What is NOT Included

- Relevance evaluation (gold labels unavailable)
- Reliability = Faithfulness AND Relevance
- Fine-tuned models (Phase 2 not started)
- Test evaluation (not run)

## Current Stage

- [x] RAGognize adapter
- [x] Data validation and split manifest (277 questions, 1100 samples)
- [x] Sentence splitting with claim positions
- [x] Context windowing
- [x] Faithfulness NLI (zero-shot)
- [x] Aggregation strategies
- [x] Validation evaluation
- [ ] **Relevance gold-label acquisition**
- [ ] **Reliability evaluation**
- [ ] Faithfulness Encoder fine-tuning (Phase 2)
- [ ] Test evaluation

## Phase 2 Considerations

See [PHASE2_PLAN.md](PHASE2_PLAN.md) for detailed experimental plan.

**Key prerequisites before Phase 2:**
1. Acquire Relevance gold labels (or document why unavailable)
2. Enable full Reliability evaluation
3. Design Token-level Faithfulness training data

## Not Implemented

- LLM-as-judge approaches
- SelfCheckGPT-style methods
- LoRA fine-tuning (Phase 2)
- Marker-CoT
- External LLM API calls
- Relevance evaluation (awaiting data)
