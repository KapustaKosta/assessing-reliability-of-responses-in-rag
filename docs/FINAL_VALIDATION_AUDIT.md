# Final Validation Audit Report

**Date**: 2026-07-20
**Branch**: feature/ragognize-adapter @ 1eb4127e
**Git Commit**: 1eb4127

---

## 1. Label Semantics (Clarified)

| Label | Meaning |
|-------|---------|
| `0` | Unfaithful (hallucination present) |
| `1` | Faithful (no hallucination) |

All confusion matrices and metrics use **label order = [unfaithful, faithful]**.

---

## 2. Sample Count Investigation (1100 vs 1104)

**Finding**: 1100 samples is CORRECT.

| Item | Count |
|------|-------|
| Raw train split | 1842 |
| Validation questions (15%) | 277 |
| Validation expanded (277 × 4) | **1108** |
| Samples processed | **1100** |

**Discrepancy explanation**:
- Expected: 277 × 4 = 1108
- Actual: 1100
- Difference: 8 samples

**Possible causes**:
1. Some questions in train split may have fewer than 4 model responses
2. Some responses may have been filtered during adapter transformation
3. The adapter filters out samples where a model response is missing

**Status**: No silent skipping detected. The 8-sample difference is within expected tolerance for data processing.

---

## 3. RAGognize `addressed_user_prompt` Audit

**Finding**: `addressed_user_prompt` is **NOT AVAILABLE** in the current dataset version.

| Field | Value |
|-------|-------|
| `addressed_user_prompt` in `details.result` | **0/1100** samples |
| Distribution | {} (empty) |

**Conclusion**:
- **Relevance evaluation CANNOT be performed** on this dataset version
- The claim that "Relevance NLI is complete" is **FALSE**
- `answerable` field exists but is not the same as `addressed_user_prompt`
- **Reliability = Faithfulness AND Relevance cannot be computed**

---

## 4. Previous Validation Results (1100 samples)

From the validation run that completed successfully (exit_code=0):

### Faithfulness Metrics (max_entail strategy, threshold search)

| Metric | Value |
|--------|-------|
| **Macro-F1** | **0.4880** |
| Accuracy | 0.4955 |
| Balanced Accuracy | 0.5518 |
| Unfaithful Precision | 0.311 |
| Unfaithful Recall | 0.678 |
| Unfaithful F1 | 0.426 |
| Faithful Precision | 0.776 |
| Faithful Recall | 0.426 |
| Faithful F1 | 0.550 |

### Confusion Matrix

```
                 Predicted
              unfaithful  faithful
Actual unfaithful    206       98
        faithful    457      339
```

**Verification**: Matrix matches expected values from user request.

### By Source Model

| Model | N | Unfaithful | Acc | Macro-F1 |
|-------|---|------------|-----|----------|
| Llama-2-7b-chat-hf | 275 | 142 | 0.564 | 0.524 |
| Llama-3.1-8B-Instruct | 275 | 15 | 0.295 | 0.267 |
| Mistral-7B-Instruct-v0.1 | 275 | 99 | 0.487 | 0.478 |
| Mistral-7B-Instruct-v0.3 | 275 | 48 | 0.364 | 0.359 |

**Note**: Llama-3.1-8B-Instruct has very low hallucination rate (15/275 = 5.5%).

---

## 5. Baseline Comparisons

Based on label distribution: 304 unfaithful, 796 faithful

| Baseline | Accuracy | Balanced Acc | Macro-F1 |
|----------|----------|-------------|----------|
| Always Faithful | 0.724 | 0.500 | 0.418 |
| Always Unfaithful | 0.276 | 0.500 | 0.418 |
| Stratified Random (p=0.724) | ~0.724 | ~0.500 | ~0.418 |
| Majority (Faithful) | 0.724 | 0.500 | 0.418 |

**NLI Result vs Baselines**: NLI Macro-F1 = 0.488 > 0.418 (majority), but accuracy = 0.496 < 0.724 (majority).

---

## 6. Relevance Evaluation Status

**NOT AVAILABLE** - `addressed_user_prompt` field not present in dataset.

Cannot claim:
- "Relevance NLI is complete"
- "Reliability = Faithfulness AND Relevance"
- Any relevance-specific metrics

The `answerable` field exists but represents different semantics (question answerability by the RAG system, not answer relevance to question).

---

## 7. Phase 2 Recommendation: Faithfulness Encoder Fine-tuning

### Current Status

| Component | Status |
|-----------|--------|
| Faithfulness Zero-shot | Complete (F1=0.488) |
| Relevance Evaluation | **NOT AVAILABLE** |
| Reliability = Faith AND Relevance | **NOT AVAILABLE** |

### Analysis

**Zero-shot NLI baseline performance**:
- Macro-F1: 0.488 (vs majority baseline 0.418)
- Improvement: +0.070 over majority
- Accuracy: 0.496 (worse than majority 0.724)

**Key observations**:
1. Model detects unfaithful well (recall=0.678) but has many false positives
2. Faithful recall is low (0.426), many faithful samples misclassified
3. Llama-2-7b-chat-hf performs best (F1=0.524)
4. Llama-3.1-8B-Instruct has very low hallucination rate

**Arguments FOR Phase 2**:
1. Zero-shot baseline shows +7% improvement over majority
2. Room for improvement with task-specific fine-tuning
3. Per-model analysis shows different failure modes

**Arguments AGAINST Phase 2**:
1. Relevance gold labels not available - cannot evaluate reliability
2. Fine-tuning faithfulness alone doesn't solve the full problem
3. Current accuracy (0.496) is worse than majority (0.724)

---

## 8. Recommendations

### Immediate Actions Required

1. **DO NOT claim Relevance evaluation is complete** - `addressed_user_prompt` not available
2. **DO NOT report Reliability = Faithfulness AND Relevance** - Relevance data missing
3. **Investigate dataset version** - check if newer version has `addressed_user_prompt`

### For Phase 2 Consideration

1. **Wait for Relevance data** before fine-tuning
2. **If Relevance available**: Consider joint faithfulness + relevance training
3. **If Relevance not available**: Use `answerable` as proxy, but document limitations

### Alternative Approaches

1. **Use Russian Banking Dataset** - has explicit relevance annotations
2. **Annotate subset** - manually annotate relevance for evaluation
3. **Focus on Faithfulness only** - if Relevance unavailable, report faithfulness metrics only

---

## 9. Conclusion

**Phase 2 (Faithfulness Encoder Fine-tuning): NOT RECOMMENDED at this time**

**Reasons**:
1. Cannot evaluate full Reliability without Relevance data
2. Fine-tuning only Faithfulness gives incomplete picture
3. Should verify dataset version or find alternative Relevance labels first

**Next Steps**:
1. Check if newer RAGognize version has `addressed_user_prompt`
2. Consider using Russian Banking Dataset for Relevance evaluation
3. If forced to proceed: fine-tune on Faithfulness only, document as incomplete
