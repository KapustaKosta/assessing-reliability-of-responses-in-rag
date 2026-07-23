# Token-Level Hallucination Detection (Scheme 3)

## Overview

This document describes the Token-level Encoder hallucination detection scheme (Scheme 3), which is independent from the Claim-MIL scheme (Scheme 1/2).

## Input and Output

### Input
```
Context + Question + Answer
```

### Output
```
- Token-level: p_hallucination for each Answer token
- Span-level: hallucination character spans
- Answer-level: hallucination probability
```

## Key Differences from Claim-MIL

| Aspect | Claim-MIL (Scheme 1/2) | Token-Level (Scheme 3) |
|--------|-------------------------|------------------------|
| Granularity | Claim-level | Token-level |
| Supervision | Bag of claims | Character spans |
| Aggregation | MIL pooling | Token classification |
| Output | p_unsupported per claim | p_hallucination per token |

## Character Span to Token Label Rules

Given a character span `[start, end)` in the answer and a token with character range `[token_start, token_end)`:

```
token_is_hallucinated = max(token_start, span_start) < min(token_end, span_end)
```

**Important**: Boundary touch does NOT count as overlap. If `token_end == span_start`, the token is NOT hallucinated.

A token is labeled as hallucinated (label=1) if it overlaps with ANY hallucination span.

## Model Architecture

```
AutoModel Encoder (pretrained mDeBERTa-v3-base-mnli-xnli)
    ↓
Dropout
    ↓
Linear(hidden_size, 2) → logits [supported, hallucinated]
```

## Loss Function

```
CrossEntropyLoss(ignore_index=-100)
```

Only Answer tokens contribute to loss. Non-Answer tokens (special tokens, Context, Question, Padding) have label=-100.

Optional positive class weight for imbalanced data:
```
class_weights = [1.0, positive_class_weight]
```

## Context Window Strategy

Three-part encoding:
```
[CLS] Context [SEP] Question [SEP] Answer [SEP]
```

Budget priority:
1. Full Answer (never truncated)
2. Full Question
3. Remaining budget for Context

When context exceeds budget, sliding windows are used:
- Each window contains full Question + Answer
- Context portion slides with configurable stride
- Multiple windows aggregated via `max` or `mean`

## Threshold Selection

Threshold is tuned on the **development set only** (never on test).

Search range: `[threshold_min, threshold_max]` with configurable step.

Metric for selection: `token_f1`, `span_f1`, or `answer_f1`.

## Metrics (Three Levels)

### Token-Level
- Precision/Recall/F1 for positive class
- Accuracy
- Macro F1
- Confusion Matrix

### Character Span-Level
- Character Precision/Recall/F1
- Compares predicted spans with gold spans at character granularity

### Answer-Level
- Precision/Recall/F1
- Accuracy
- ROC-AUC / PR-AUC
- Brier Score
- Expected Calibration Error (ECE)

## Offline Model Configuration

Set the model path via environment variable:
```bash
export CLAIM_MIL_MODEL_PATH="/home/ma-user/work/models/mDeBERTa-v3-base-mnli-xnli"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

Or via CLI argument:
```bash
--model_path /home/ma-user/work/models/mDeBERTa-v3-base-mnli-xnli
```

## CPU and Ascend NPU Commands

### CPU
```bash
export CLAIM_MIL_MODEL_PATH="/path/to/model"
python -m token_classifier.train --device cpu --model_path "$CLAIM_MIL_MODEL_PATH" ...
```

### NPU
```bash
export CLAIM_MIL_MODEL_PATH="/path/to/model"
python -m token_classifier.train --device npu --model_path "$CLAIM_MIL_MODEL_PATH" ...
```

## Smoke, Overfit, and Training Commands

### Smoke Test
```bash
python -m token_classifier.train \
  --smoke_test \
  --device npu \
  --model_path "$CLAIM_MIL_MODEL_PATH" \
  --epochs 1 \
  --batch_size 2 \
  --max_train_samples 4 \
  --max_dev_samples 4 \
  --results_dir results/token_classifier_npu_smoke
```

### Tiny Overfit
```bash
python -m token_classifier.train \
  --overfit_diagnostic \
  --device npu \
  --model_path "$CLAIM_MIL_MODEL_PATH" \
  --epochs 30 \
  --batch_size 2 \
  --results_dir results/token_classifier_npu_overfit
```

### Full Training
```bash
python -m token_classifier.train \
  --data_path data/processed/train.csv \
  --model_path "$CLAIM_MIL_MODEL_PATH" \
  --results_dir results/token_classifier \
  --epochs 10 \
  --batch_size 8 \
  --learning_rate 2e-5
```

### Evaluation
```bash
python -m token_classifier.evaluate \
  --data_path data/processed/test.csv \
  --results_dir results/token_classifier \
  --output_dir results/token_classifier/evaluation
```

### Single Sample Prediction
```bash
python -m token_classifier.predict \
  --checkpoint_path results/token_classifier/best_checkpoint.pt \
  --context "Context text..." \
  --question "Question?" \
  --answer "Answer..." \
  --threshold 0.5
```

## Data Leakage Prevention

1. **Grouped Split**: All samples with the same `question_id` stay in the same split
2. **No Question ID Overlap**: Train/dev/test have disjoint question IDs
3. **Threshold Tuning Only on Dev**: Never tune threshold on test
4. **Checkpoint Selection Only on Dev**: Never select best checkpoint using test

## Known Limitations

1. Tokenization may cause slight misalignment between character spans and token boundaries
2. Very long answers may be truncated if exceeding `max_length`
3. Context windowing may miss information at window boundaries
4. Unicode handling depends on tokenizer behavior

## Directory Structure

```
src/token_classifier/
    __init__.py
    config.py          # Configuration dataclass
    schema.py         # Data schema and validation
    labeling.py       # Token labeling and offset tracking
    dataset.py        # PyTorch Dataset
    model.py          # TokenHallucinationClassifier
    metrics.py        # Token/Span/Answer metrics
    postprocess.py    # Token-to-span conversion
    checkpoint.py     # Checkpoint management
    train.py          # Training script
    evaluate.py       # Evaluation script
    predict.py        # Prediction script

tests/token_classifier/
    test_schema.py
    test_labeling.py
    test_dataset.py
    test_model.py
    test_metrics.py
    test_postprocess.py
    test_checkpoint.py
    test_no_leakage.py
    test_local_model_integration.py
    test_npu_smoke.py
```

## References

- Related to `src/claim_mil/` which implements Claim-level MIL
- Uses same underlying encoder: `mDeBERTa-v3-base-mnli-xnli`
- Data format compatible with `src/ragognize_adapter/`
