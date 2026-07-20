# Answer Reliability Classification

## Project Type

**Local Multilingual Encoder / NLI Classifier for Answer Reliability**

This project implements a claim-level faithfulness and relevance classifier using
encoder-based NLI models. It is NOT:
- LLM-as-judge
- SelfCheckGPT
- LoRA / marker-CoT
- Full LLM fine-tuning

## Architecture

```
Question + Retrieved Context + Answer
    │
    ├─→ Answer Split into Claims
    │       │
    │       └─→ Claim 1, Claim 2, ..., Claim N
    │           (each with claim_id, char positions, text)
    │
    ├─→ Long Context → Overlapping Windows
    │       │
    │       └─→ Window 1, Window 2, ..., Window M
    │           (overlapping, preserves claim boundaries)
    │
    ├─→ Faithfulness NLI
    │       │
    │       └─→ For each (Claim, Window) pair:
    │               premise = context window
    │               hypothesis = answer claim
    │               → entailment / neutral / contradiction
    │
    ├─→ Relevance NLI
    │       │
    │       └─→ For each claim:
    │               premise = question
    │               hypothesis = claim
    │               → claim_relevance_score
    │
    ├─→ Aggregation
    │       │
    │       ├─→ claim-level → answer-level faithfulness
    │       └─→ claim-level → answer-level relevance
    │
    └─→ Reliability Decision
            │
            └─→ reliable = (faithful AND relevant)
```

## Reliability Definition

```
reliability = faithfulness AND relevance

- faithful = 1: All claims are supported by context
- relevant = 1: All claims address the question
- reliable = 1: faithful=1 AND relevant=1
```

An answer is **reliable** only when it is both:
1. **Faithful**: All claims are consistent with the retrieved context
2. **Relevant**: All claims directly address the user's question

## Pipeline Components

### 1. Answer Splitting

Splits answers into atomic claims for fine-grained evaluation.

- Sentence splitting (base implementation)
- Atomic claim extraction (interface preserved for future enhancement)
- Each claim stores: `claim_id`, `char_start`, `char_end`, `claim_text`

### 2. Context Windowing

Handles long contexts with overlapping windows.

- Overlapping windows with configurable stride
- Claims are never truncated
- `pair truncation only_first` for premise length control
- Preserves: `window_id`, `token_range`, `doc_source`

### 3. Faithfulness NLI

Evaluates whether each claim is supported by the context.

```python
premise = retrieved_context_window
hypothesis = answer_claim
→ p(entailment), p(neutral), p(contradiction)
```

Label indices are read dynamically from `model.config.id2label`.

### 4. Relevance NLI

Evaluates whether each claim addresses the question.

```python
premise = question
hypothesis = answer_claim
→ claim_relevance_score
```

Phase 1 uses zero-shot NLI; interface preserved for training a binary classifier.

### 5. Aggregation Strategies

Combines claim-level scores into answer-level predictions.

**Faithfulness aggregations:**
- `max_entail`: score = max(p_entailment)
- `entail_minus_contradiction`: score = max(p_entail) - max(p_contrad)
- `claim_min_support`: score = min(p_entail) per claim
- `contradiction_penalized_support`: score = mean(p_entail) - std(p_contrad)

### 6. Prediction

```python
faithfulness_pred = (faithfulness_score >= faithfulness_threshold)
relevance_pred = (relevance_score >= relevance_threshold)
reliability_pred = faithfulness_pred AND relevance_pred
```

## Output Format

Answer-level output includes:

| Field | Description |
|-------|-------------|
| `case_id` | Unique case identifier |
| `faithfulness_score` | Aggregated faithfulness metric |
| `faithfulness_prediction` | Binary: 1=faithful, 0=unfaithful |
| `relevance_score` | Aggregated relevance metric |
| `relevance_prediction` | Binary: 1=relevant, 0=irrelevant |
| `reliability_prediction` | Binary: 1=reliable, 0=unreliable |

Claim-window level output includes:

| Field | Description |
|-------|-------------|
| `case_id` | Unique case identifier |
| `claim_id` | Claim identifier within answer |
| `claim_text` | Text of the claim |
| `window_id` | Context window identifier |
| `entailment_probability` | P(entailment) |
| `neutral_probability` | P(neutral) |
| `contradiction_probability` | P(contradiction) |
| `claim_faithfulness_score` | Claim-level faithfulness |
| `claim_relevance_score` | Claim-level relevance |
| `faithfulness_prediction` | Binary prediction |
| `relevance_prediction` | Binary prediction |

## Current Model

**Phase 1: Zero-shot Baseline**

Model: `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`
- Multilingual (100+ languages)
- No fine-tuning in phase 1
- Full fine-tuning planned for phase 2

## Datasets

### RAGognize (F4biian/RAGognize)

English RAG dataset with hallucination annotations.

- 1,842 train samples
- 2,781 test samples
- 4 model responses per sample
- Binary faithfulness labels from annotations

### Russian Banking Dataset

Legacy Russian customer service dataset with error markers.

## Metrics

Classification metrics computed for:

1. **Faithfulness**: faithful vs unfaithful
2. **Relevance**: relevant vs irrelevant
3. **Reliability**: reliable vs unreliable

Per-class metrics:
- Precision
- Recall
- F1-score

Aggregate metrics:
- Accuracy
- Balanced Accuracy
- Macro F1
- AUROC
- AUPRC

## Current Stage

- [x] RAGognize adapter
- [x] Data validation and split manifest
- [x] Sentence splitting
- [x] Context windowing
- [x] Faithfulness NLI (zero-shot)
- [x] Relevance NLI (zero-shot)
- [x] Aggregation strategies
- [x] Full validation evaluation
- [ ] Encoder fine-tuning (Phase 2)
- [ ] Test evaluation

## Not Implemented

- LLM-as-judge approaches
- SelfCheckGPT-style methods
- LoRA fine-tuning
- Marker-CoT
- External LLM API calls
