# Data Integrity Validation Report

**File**: `results/token_level/data/val_data.jsonl`

## Statistics

- Total samples: 100
- Samples with hallucination spans: 36 (36.0%)
- Samples with empty spans: 64 (64.0%)
- Total span count: 54
- Total gold characters: 6742
- Positive rate: 36.0%
- Span length: mean=124.9, median=92.5, std=138.1
- Span length range: [17, 757]
- Samples with validation issues: 0 (0 issues total)

## By Source Model

- Llama-2-7b-chat-hf: 25 samples, 14 with spans (56.0%), 3818 total chars
- Llama-3.1-8B-Instruct: 25 samples, 3 with spans (12.0%), 487 total chars
- Mistral-7B-Instruct-v0.1: 25 samples, 13 with spans (52.0%), 1400 total chars
- Mistral-7B-Instruct-v0.3: 25 samples, 6 with spans (24.0%), 1037 total chars

## Token Label Validation

- Samples validated: 100
- Total tokens: 9664
- Samples with label issues: 0
- Total label issues: 0
- Token label accuracy: 100.00%

## Span Recovery (Gold Labels as Predictions)

- Samples validated: 100
- Total gold chars: 6742
- Total pred chars: 6773
- Total overlap chars: 6737
- **Character Precision: 0.9947**
- **Character Recall: 0.9993**
- **Character F1: 0.9970**
