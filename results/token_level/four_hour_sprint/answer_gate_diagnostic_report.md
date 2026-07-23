# Two-Stage Answer Gate + Token Span Diagnostic Report

**Date**: 2026-07-23
**Commit**: c001743 (frozen evaluator)

## 1. Metadata Fix

| Dataset | Samples | source_model known | sample_id coverage |
|---------|---------|---------------------|--------------------|
| Dev (hold-out) | 1256 | 1256/1256 (100.0%) | 100% |
| Train (portion) | 5012 | 5012/5012 (100.0%) | 100% |

## 2. Reachability Analysis (Dev, t=0.20)

| Category | N | Vis Token Ratio | Vis Char Ratio | Gold Char Recall | Fully Vis Span | Partially Vis | Fully Trunc |
|----------|---|-----------------|----------------|-----------------|----------------|---------------|-------------|
| A | 309 | 1.0000 | 0.9987 | 0.9997 | 0.9996 | 0.0004 | 0.0000 |
| B | 68 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| C | 453 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| D | 426 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## 3. Gate Feature Engineering

Features (21):
`['prob_max', 'prob_mean', 'prob_p90', 'prob_p95', 'prob_p99', 'prob_top1_mean', 'prob_top3_mean', 'prob_top5_mean', 'prob_sum', 'prob_sum_norm', 'count_above_005', 'count_above_015', 'ratio_above_005', 'ratio_above_015', 'n_pred_spans', 'max_consecutive_high', 'context_len', 'question_len', 'truncation_ratio', 'prob_std', 'prob_min']`

## 4. Gate Training Results

| Model | Train Acc | Dev Acc | Dev Prob Range |
|-------|-----------|---------|----------------|
| LogisticRegression | 0.7289 | 0.7205 | [0.1750, 0.9966] |
| HistGradientBoosting | 0.8939 | 0.7094 | [0.0006, 0.9978] |

## 5. Two-Stage Comparison (Character F1 Primary)

| Method | Char F1 | Char P | Char R | Ans F1 | Ans P | Ans R | Tok F1 | Gate t | Token t |
|--------|---------|--------|--------|--------|--------|--------|--------|--------|--------|
| Baseline t=0.20 | 0.3841 | 0.4751 | 0.3223 | 0.5426 | 0.4055 | 0.8196 | 0.3864 | N/A | 0.20 |
| Raw Best CharF1 | 0.4310 | 0.4297 | 0.4324 | 0.5279 | 0.3633 | 0.9655 | 0.3633 | N/A | 0.05 |
| LR Gate Best | 0.4341 | 0.4376 | 0.4306 | 0.5476 | 0.3865 | 0.9390 | 0.3676 | 0.25 | 0.05 |
| HG Gate Best | 0.4318 | 0.4318 | 0.4319 | 0.5355 | 0.3713 | 0.9602 | 0.3647 | 0.10 | 0.05 |

## 6. Category Breakdown (Best Gate Results)

| Method | A (TP) | B (FN) | C (FP) | D (TN) |
|--------|--------|--------|--------|--------|
| Baseline t=0.20 | 309 | 68 | 453 | 426 |
| LR Gate Best | 354 | 23 | 562 | 317 |
| HG Gate Best | 362 | 15 | 613 | 266 |

## 7. Key Findings

- Baseline (t=0.20): Char F1=0.3841, Ans F1=0.5426
- Best raw token (no gate): Char F1=0.4310 @ t=0.05
- LR Gate best: Char F1=0.4341 @ gate_t=0.25 tok_t=0.05
- HG Gate best: Char F1=0.4318 @ gate_t=0.10 tok_t=0.05

## 8. Output Files

- `answer_gate_features_train.parquet` — Train gate features
- `answer_gate_features_dev.parquet` — Dev gate features
- `answer_gate_results.csv` — Per-sample gate probabilities
- `two_stage_threshold_sweep.csv` — Full 2D sweep results
- `two_stage_best_metrics.json` — Best metrics per method
- `reachability_analysis.csv` — Per-sample reachability
- `answer_gate_diagnostic_report.md` — This report