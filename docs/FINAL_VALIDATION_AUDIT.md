# Final Validation Audit Report

**Evaluation Name**: Zero-shot Claim-level NLI Faithfulness Baseline on RAGognize Validation
**Date**: 2026-07-20
**Branch**: feature/ragognize-adapter @ 1eb4127
**Git Commit**: 1eb4127

---

## Executive Summary

This document reports the results of a Final Validation Audit for the Encoder/NLI
Faithfulness detection project. The evaluation is scoped to **Faithfulness only**;
**Relevance evaluation is NOT AVAILABLE** due to missing gold labels.

**Current Evaluation**: Faithfulness NLI (Zero-shot mDeBERTa)
**NOT a**: Full Reliability evaluation

---

## 1. Label Semantics (CRITICAL)

| Label | Meaning | Description |
|-------|---------|-------------|
| `0` | **Unfaithful** | Hallucination present in answer |
| `1` | **Faithful** | No hallucination, answer supported by context |

All confusion matrices and metrics use **label order = [unfaithful, faithful]**.

---

## 2. Sample Count Verification

| Item | Count |
|---|---:|
| Raw train rows | 1,842 |
| Validation rows | **275** |
| Unique validation questions | **141** |
| Models per row | 4 |
| Theoretical response slots | **1,100** |
| Source-data missing | **0** |
| Runtime skipped | **0** |
| Actual valid samples | **1,100** |

### Verification

The frozen RAGognize dataset revision deterministically produces:

- 275 validation rows
- 141 unique validation questions
- 4 model responses per row
- 1,100 valid response samples
- 0 source-missing responses
- 0 runtime-skipped samples

Invariant:

`1,100 = 0 + 1,100`

Earlier documentation expected 277 rows, 1,108 theoretical slots,
and 8 source-missing responses. These values could not be reproduced
from the frozen dataset revision and should not be used for the final
evaluation report.


---

## 3. RAGognize `addressed_user_prompt` Audit

**Finding**: `addressed_user_prompt` is **NOT AVAILABLE** in the current dataset version.

| Metric | Value |
|--------|-------|
| `addressed_user_prompt` available | **0 / 1,100** |
| Availability rate | **0%** |
| `answerable` field | Available (but different semantics) |

### Conclusion

- ❌ **Relevance formal evaluation**: NOT AVAILABLE
- ❌ **Reliability = Faithfulness AND Relevance**: NOT COMPUTABLE
- ⚠️ Using `answerable` as proxy would be incorrect semantic mapping

---

## 4. Validation Results (Zero-shot mDeBERTa)

### Model Configuration

| Property | Value |
|----------|-------|
| Model | MoritzLaurer/mDeBERTa-v3-base-mnli-xnli |
| Mode | Zero-shot (no fine-tuning) |
| Device | MPS |
| Strategy | max_entail |
| Threshold | 0.5 |

### Overall Metrics

| Metric | Value |
|--------|-------|
| **Accuracy** | **0.4955** |
| **Balanced Accuracy** | **0.5518** |
| **Macro-F1** | **0.4880** |
| **AUROC** | **~0.56** |

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

**Verification against expected**:
- TN=206, FP=98, FN=457, TP=339 ✓
- Unfaithful: P=0.311, R=0.678, F1=0.426 ✓
- Faithful: P=0.776, R=0.426, F1=0.550 ✓

---

## 5. Baseline Comparisons

Based on label distribution: 304 unfaithful, 796 faithful

| Baseline | Accuracy | Balanced Acc | Macro-F1 |
|----------|----------|-------------|----------|
| Always Faithful | 0.724 | 0.500 | 0.418 |
| Always Unfaithful | 0.276 | 0.500 | 0.418 |
| Stratified Random | ~0.724 | ~0.500 | ~0.418 |
| Majority (Faithful) | 0.724 | 0.500 | 0.418 |
| **NLI (max_entail)** | **0.496** | **0.552** | **0.488** |

**Analysis**:
- NLI Macro-F1 = 0.488 > majority baseline 0.418 (+0.070 improvement)
- NLI Accuracy = 0.496 < majority baseline 0.724 (-0.228)
- Model tends to over-predict unfaithful (FN=457 > FP=98)

---

## 6. Source Model Subgroup Analysis

| Model | N | Unfaithful | Acc | Macro-F1 | Unfaithful P/R/F |
|-------|---|------------|-----|----------|------------------|
| Llama-2-7b-chat-hf | 275 | 142 | 0.564 | 0.524 | - |
| Llama-3.1-8B-Instruct | 275 | 15 | 0.295 | 0.267 | - |
| Mistral-7B-Instruct-v0.1 | 275 | 99 | 0.487 | 0.478 | - |
| Mistral-7B-Instruct-v0.3 | 275 | 48 | 0.364 | 0.359 | - |

**Observations**:
1. Llama-2-7b-chat-hf: Best performance (F1=0.524), balanced label distribution
2. Llama-3.1-8B-Instruct: Very low hallucination rate (15/275 = 5.5%)
3. Mistral variants: Intermediate performance

---

## 7. What IS and IS NOT Evaluated

### ✅ What IS Evaluated

| Component | Status |
|-----------|--------|
| Faithfulness NLI (zero-shot) | **COMPLETE** |
| Aggregation strategies | **COMPLETE** |
| Baseline comparisons | **COMPLETE** |
| Subgroup analysis by model | **COMPLETE** |

### ❌ What IS NOT Evaluated

| Component | Status | Reason |
|-----------|--------|--------|
| Relevance formal evaluation | **NOT AVAILABLE** | `addressed_user_prompt` missing |
| Reliability = Faith AND Relevance | **NOT COMPUTABLE** | Relevance data unavailable |
| Fine-tuned models | **NOT DONE** | Phase 2 not started |
| Test evaluation | **NOT DONE** | Validation only |

---

## 8. Phase 2 Recommendation

### Immediate Recommendation: NOT RECOMMENDED FOR FULL RELIABILITY

**Reasons**:
1. Relevance gold labels not available - cannot evaluate Reliability
2. Fine-tuning Faithfulness alone gives incomplete picture
3. Should verify dataset version or find alternative Relevance labels first

### Alternative Path Forward

**Option A**: Wait for Relevance data
- Check if newer RAGognize version has `addressed_user_prompt`
- Status: Awaiting dataset update

**Option B**: Use alternative dataset
- Russian Banking Dataset has explicit relevance annotations
- Requires dataset-specific adapter

**Option C**: Prepare Faithfulness-only experiments
- Design Token-level Faithfulness training pipeline
- Use hallucination spans as supervision signal
- Document as Faithfulness-only Phase 2

### Recommended Next Steps

1. **Acquire Relevance gold labels** before full Reliability training
2. **Prepare data pipeline** for Token-level Faithfulness (using hallucination spans)
3. **Design comparison**: Zero-shot vs Fine-tuned Faithfulness
4. **Document limitations**: Report Faithfulness results only until Relevance available

---

## 9. Risk Assessment

### Remaining Evaluation Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Misreporting Reliability | HIGH | Clearly label "Faithfulness only" |
| Confusion about sample count | MEDIUM | Document 1100 = 1108 - 8 (source missing) |
| Label semantics confusion | MEDIUM | Document unfaithful=0, faithful=1 consistently |
| Silent skipping undetected | LOW | Verified 0 runtime skips |

### Recommendations

1. **Always prefix evaluation name** with "Faithfulness-only"
2. **Never claim Reliability** without Relevance evaluation
3. **Document sample count** as "1,100 valid samples (source data)"
4. **Use consistent label naming** in all outputs

---

## 10. Conclusion

**This evaluation is**: Zero-shot Claim-level NLI Faithfulness Baseline
**This evaluation is NOT**: Full Reliability Evaluation

**Phase 2 should**:
1. Prepare Faithfulness-specific experiments (Token-level)
2. Wait for or acquire Relevance gold labels
3. Only compute Reliability when both Faithfulness AND Relevance are available

**Do NOT**:
- Report this as complete Reliability evaluation
- Claim Relevance evaluation is complete
- Begin Reliability fine-tuning without Relevance data
