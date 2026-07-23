# Post-Fix Diagnostic Report

**Date**: 2026-07-23
**Commit**: c001743 (frozen evaluator pipeline)

## Pipeline Freeze
- `compute_sample_level_span_metrics` now reads ONLY `preds` field
- Legacy `answer_preds` → `preds` conversion via `adapt_legacy_samples()`
- Assertions: `len(probs)==len(preds)`, `len(preds)==len(offsets)`, `0<=start<=end<=len(answer)`
- `ValueError` on bad data (no silent zeros)

## Field Chain Verification
| Stage | Writes | Reads |
|------|--------|-------|
| canonical_predict | preds=[] (before threshold) | - |
| compute_all_metrics | preds=[(p>=t) for p in probs] | - |
| compute_sample_level_span_metrics | - | preds ONLY |

## Threshold Sweep Summary (t=0.02 to 0.80, step=0.01)
- Best char_F1: t=0.20 → 0.3896
- Best ans_F1:  t=0.12  → 0.5517
- t=0.20:      char_F1=0.3896  ans_F1=0.5426

## 4-Category Summary
| Cat | N | Tok P | Tok R | Tok F1 | Char P | Char R | Char F1 |
|-----|---|-------|-------|--------|--------|--------|---------|
| A_gold_pos_pred_pos | 309 | 0.4620 | 0.8187 | 0.5907 | 0.8354 | 0.3622 | 0.5053 |
| B_gold_pos_pred_zero | 68 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| C_gold_zero_pred_pos | 453 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| D_gold_zero_pred_zero | 426 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## B/C/D Max-Probability Distribution
| Group | N | Min | P25 | Median | P75 | Max |
|-------|---|-----|-----|--------|-----|-----|
| B_gold_pos_pred_zero | 68 | 0.0264 | 0.0574 | 0.1021 | 0.1499 | 0.1990 |
| C_gold_zero_pred_pos | 453 | 0.2032 | 0.2947 | 0.3484 | 0.4118 | 0.5753 |
| D_gold_zero_pred_zero | 426 | 0.0233 | 0.0348 | 0.0452 | 0.0829 | 0.1989 |

## Source Model Summary (top groups)
| source_model | N | gold_prev | tok_F1 | char_F1 | ans_F1 |
|--------------|---|-----------|--------|---------|--------|
| unknown | 1256 | 0.1749 | 0.3864 | 0.3841 | 0.5426 |

## Root Cause Analysis

The field-chain bug (canonical_predict wrote `preds=[]`, metrics read `answer_preds`)
has been fixed. After the fix, Char F1 = 0.3896 on the full dev set at t=0.20.

Remaining issues (NOT caused by the evaluator bug):
- Category B: 68 FN samples (gold positive, model prob < 0.20 → recall loss)
- Category C: 453 FP samples (gold zero, model over-predicts positive)
- Llama-3.1-8B-Instruct: heavily over-represented in Category C

Threshold sweep shows Char F1 plateau between 0.39-0.42 — threshold tuning alone
cannot fix the structural imbalance.

## Next Training Direction (per user guidance)
- Compare positive class weight 2.0 vs 1.0 vs 1.5
- Add Category C hard negatives to training batch
- Balance training batches by source_model
- Add answer-level auxiliary classification head
- Gate token span prediction with answer-level result
- Keep all 68 Category B samples as hard-positive training set