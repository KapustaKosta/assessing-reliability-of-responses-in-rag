# Environment Audit Report - Token-Level Hallucination Detection

**Date**: 2026-07-22
**Branch**: feat/ragognize-token-level

---

## Executive Summary

**Status**: BLOCKED - Cannot proceed with Token-Level Hallucination Detection

**Reason**: No access to RAGognize dataset with character-level span annotations

---

## Git Status

```
Branch: feat/ragognize-token-level (newly created)
Main branch: main
Working branch: chengyi/claim-mil-correctness-fixes
```

### Recent Commits
- `ebcaab9` - chore: ignore local Phase 2 checkpoints
- `6af0384` - feat: train claim-level MIL faithfulness model
- `cc2d6e9` - feat: build supervised faithfulness claim bags
- `7a9c6a2` - results: add Stage 3 NLI faithfulness baseline artifacts
- `ff1f182` - fix: finalize RAGognize manifest and relevance extraction

---

## Python Environment

| Component | Version |
|-----------|---------|
| Python | 3.11.10 |
| PyTorch | 2.7.1 |
| Transformers | 4.57.1 |
| Datasets | 3.0.1 |
| Accelerate | 1.0.1 |

---

## Hardware

| Component | Status |
|-----------|--------|
| GPU (nvidia-smi) | Not available |
| NPU (Huawei Ascend) | Available |
| Disk Space | 50GB total, 50GB available (2% used) |

---

## Network Status

### HuggingFace Access Test

```
ConnectionError: Couldn't reach 'F4biian/RAGognize' on the Hub (ProxyError)
```

**Result**: Network access to HuggingFace Hub is blocked.

---

## Data Availability Investigation

### 1. Searched Locations

| Location | Result |
|----------|--------|
| HuggingFace Cache (`~/.cache/huggingface/`) | Empty |
| Project data directories | Manifests only |
| Work directory (`/home/ma-user/work/`) | No RAGognize JSON/JSONL/Parquet files |
| Downloads folder | Empty |
| Any `*ragognize*` directories | Only adapter code directory |

### 2. Local Processed Data Analysis

**File**: `processed/ragognize_split_manifest.csv`

Contains only:
- `question_id`, `source_model`, `faithfulness_label`
- **No character-level span annotations**

**File**: `results/overnight_autorun/diagnostics/data_supervision_audit.json`

Key findings:
```json
{
  "has_character_spans": false,
  "reason": "markers column contains categorical labels like reason_hallucinated_fact, 
            NOT character positions with start/end"
}
```

**Marker Types Available**:
- `reason_hallucinated_fact`
- `reason_incomplete_answer`
- `reason_false_verification`
- `reason_reveals_ai_identity`
- `reason_chunk_fact_mixup`
- `reason_wrong_navigation`
- `reason_irrelevant_chunk_used`
- `reason_missed_chunk_conditions`
- `reason_answer_for_operator`

**Conclusion**: Local data has **answer-level categorical labels only**, not token/character-level spans.

### 3. Adapter Code Analysis

**File**: `src/ragognize_adapter/adapter.py`

The adapter **expects** character-level spans:
```python
@dataclass
class HallucinationSpan:
    text: str
    start: int
    end: int
    valid: bool
```

But this depends on the raw HuggingFace dataset being available.

---

## Comparison: What Exists vs What Is Needed

| Aspect | Existing Data | Required for Token-Level |
|--------|---------------|------------------------|
| **Span Fields** | None (only categorical markers) | `text/start/end/valid` per span |
| **Granularity** | Answer-level (faithful vs unfaithful) | Character-level (which tokens are hallucinated) |
| **Labels** | 304 unfaithful / 796 faithful | Token-level labels for each token |
| **Span Source** | N/A | Original RAGognize annotations |

---

## Attempted Solutions

1. **Load from HuggingFace**: Failed (network blocked)
2. **Search local filesystem**: No RAGognize data files found
3. **Check HuggingFace cache**: Empty
4. **Check alternative sources**: None found

---

## GATE 1 Status: NOT PASSED

Cannot validate hallucination span text match rate because:
1. No network access to HuggingFace
2. No local RAGognize data with character spans
3. Cannot proceed to training without data

---

## Next Steps Required

1. **Obtain RAGognize data offline** from a team member who has access
2. **Verify the data contains** `responses[model_name].hallucinations` with `text/start/end/valid` fields
3. **Provide the local file path** so I can audit and process it

---

## What NOT To Do

- Do NOT derive pseudo-spans from existing claim data
- Do NOT use regex matching to create spans
- Do NOT use LLM inference to identify hallucinated tokens
- Do NOT run the official test set

---

## Branch Status

The branch `feat/ragognize-token-level` has been created and is ready for work once data is available.

```
git checkout -b feat/ragognize-token-level
```

All directory structure has been created:
- `reports/token_level/`
- `data/processed/`
- `outputs/token_level/`
- `configs/token_level/`
- `logs/token_level/`
- `scripts/`
- `third_party/`
