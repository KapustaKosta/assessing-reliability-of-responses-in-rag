# Data Exploration Summary

**Date:** July 17, 2026
**Dataset:** RAG Response Reliability Assessment
**Analyst:** Data Exploration Notebook (`notebooks/chengyi_data_exploration.ipynb`)

---

## Executive Summary

This document summarizes the exploratory data analysis (EDA) performed on the RAG reliability dataset. The dataset contains 2,030 annotated samples across train (1,450), validation (290), and test (290) splits, with binary labels for **Faithfulness**, **Relevancy**, and **Reliability**.

**Key Finding:** Reliability is defined as `Reliability = Faithfulness AND Relevancy`. All labels are internally consistent with zero mismatches.

---

## Dataset Structure

### Split Sizes

| Split | Samples | Proportion |
|-------|---------|------------|
| Train | 1,450 | 70.0% |
| Validation | 290 | 14.1% |
| Test | 290 | 14.1% |
| **Total** | **2,030** | 100% |

### Column Detection

| Role | Detected Column | Notes |
|------|----------------|-------|
| Question | `question` | Extracted from `full_dialog` via `extract_last_substantive_request()` |
| Answer | `answer` | Generated response from RAG system |
| Context | `chunk_1` | Primary retrieved chunk; chunks 1-5 always present, 6-8 optional |
| Faithfulness | `binary_faithfulness` | Binary label |
| Relevancy | `binary_relevancy` | Binary label |
| Reliability | `joint_label` | Format: `{relevancy}_{faithfulness}`, e.g., "1_1" |

### Context Storage

The dataset stores retrieved context across multiple chunk columns:
- **Always present:** `chunk_1`, `chunk_2`, `chunk_3`, `chunk_4`, `chunk_5`
- **Optional (top-8 retrieval):** `chunk_6`, `chunk_7`, `chunk_8`

The `build_combined_context()` helper concatenates all available chunks for analysis.

---

## Data Quality

### Missing Values

| Column | Missing Ratio | Expected? |
|--------|--------------|-----------|
| `markers` | 93–95% | Yes — used for partial supervision only |
| `chunk_6/7/8` | ~82% | Yes — top-5 retrieval config |
| `question` | <1% | Acceptable — only 7 total across all splits |

### Duplicates & Leakage

| Check | Result |
|-------|--------|
| Exact duplicate rows | **0** |
| Cross-split overlap | **0** |
| Group leakage | **None** |

The StratifiedGroupKFold split ensures no dialogue groups leak across splits.

---

## Label Distribution

### Binary Label Statistics (Train Set)

| Label | Positive (1) | Negative (0) | Positive Ratio |
|-------|-------------|--------------|----------------|
| Faithfulness | 1,059 | 391 | **73.0%** |
| Relevancy | 1,267 | 183 | **87.4%** |
| Reliability | 1,042 | 408 | **71.9%** |

### Joint Label Distribution (Train Set)

| Faithfulness | Relevancy | Count | Ratio | Reliability |
|--------------|-----------|-------|-------|-------------|
| 0 | 0 | 166 | 11.4% | unreliable |
| 0 | 1 | 225 | 15.5% | unreliable |
| 1 | 0 | 17 | 1.2% | unreliable |
| 1 | 1 | 1,042 | 71.9% | **reliable** |

**Most common combination:** faithfulness=1, relevancy=1 (71.9% of samples)

**Most critical minority:** faithfulness=1, relevancy=0 (only 17 samples, 1.2%)

---

## Text Characteristics

| Field | Median Chars | Median Tokens | P95 Chars | Max Chars |
|-------|-------------|---------------|-----------|-----------|
| Question | 58 | 9 | 160 | 653 |
| Answer | 416 | 59 | 603 | 1,124 |
| Context (chunk_1) | 5,447 | 857 | 8,591 | 10,191 |

### Key Observations

1. **Questions are brief** — median 9 tokens, suitable for semantic search/embedding
2. **Answers are moderate length** — median 59 tokens, consistent (P95=603 vs median=416)
3. **Context chunks are substantial** — median 857 tokens per chunk, typical for banking domain

---

## Reliability Verification

**Status:** ✅ **All labels are internally consistent**

We verified that `Reliability = Faithfulness AND Relevancy` holds for all samples:

```
Reliability label mismatches: 0
```

### Mapping Applied

The `joint_label` format `{relevancy}_{faithfulness}` is normalized to binary reliability:

| joint_label | Reliability | Interpretation |
|-------------|-------------|---------------|
| "1_1" | 1 | Both relevancy AND faithfulness are true |
| "0_0" | 0 | Both are false |
| "0_1" | 0 | Relevancy is false |
| "1_0" | 0 | Faithfulness is false |

---

## Implications for Modeling

1. **Is class weighting needed?** Yes, but the approach depends on the model type:
   - Traditional classifiers (e.g., SVM, LogisticRegression): use `class_weight="balanced"` or `sample_weight`
   - Neural network classifiers: consider weighted cross-entropy or focal loss

2. **Report macro-F1 as the primary metric**, while also reporting per-class precision, recall, F1, and confusion matrix for detailed analysis.

3. **Compare EDA findings with existing TF-IDF baseline results** in `results/stage2_tfidf/` to identify which label is the weakest performer.

4. **Focus on the "relevant but unfaithful" subgroup** (225 samples, 15.5%), which is the largest minority failure class — these cases have good retrieval but suffer from hallucinations.

5. **Future models should leverage the full context**: question + answer + complete retrieved context (multiple chunks via `build_combined_context()`).

---

## Output Files

All analysis results are saved to `results/data_exploration/`:

| File | Description |
|------|-------------|
| `eda_summary.json` | Machine-readable summary with all key metrics |
| `split_overview.csv` | Split sizes and duplicate counts |
| `label_distribution.csv` | Label value counts per split |
| `missing_values.csv` | Missing value analysis |
| `text_length_summary.csv` | Text length statistics |
| `cross_split_overlap.csv` | Overlap check results |
| `reliability_label_mismatches.csv` | Label consistency verification (empty = no mismatches) |

---

## Reproducibility

To reproduce this analysis:

```bash
# Activate the project environment
conda activate rag-reliability

# Navigate to the project root
cd /path/to/assessing-reliability-of-responses-in-rag

# Re-run the notebook
jupyter nbconvert --to notebook --execute notebooks/chengyi_data_exploration.ipynb --inplace
```

All outputs will be regenerated and saved to `results/data_exploration/`.
