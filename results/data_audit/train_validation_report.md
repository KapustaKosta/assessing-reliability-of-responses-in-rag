# Data Integrity Validation Report

**File**: `results/token_level/data/train_data.jsonl`

## Statistics

- Total samples: 100
- Samples with hallucination spans: 41 (41.0%)
- Samples with empty spans: 59 (59.0%)
- Total span count: 52
- Total gold characters: 8249
- Positive rate: 41.0%
- Span length: mean=158.6, median=91.5, std=285.7
- Span length range: [4, 1778]
- Samples with validation issues: 0 (0 issues total)

## By Source Model

- Llama-2-7b-chat-hf: 25 samples, 18 with spans (72.0%), 6405 total chars
- Llama-3.1-8B-Instruct: 25 samples, 2 with spans (8.0%), 182 total chars
- Mistral-7B-Instruct-v0.1: 25 samples, 11 with spans (44.0%), 993 total chars
- Mistral-7B-Instruct-v0.3: 25 samples, 10 with spans (40.0%), 669 total chars

## Token Label Validation

- Samples validated: 100
- Total tokens: 9506
- Samples with label issues: 0
- Total label issues: 0
- Token label accuracy: 100.00%

## Span Recovery (Gold Labels as Predictions)

- Samples validated: 100
- Total gold chars: 8249
- Total pred chars: 8272
- Total overlap chars: 8236
- **Character Precision: 0.9956**
- **Character Recall: 0.9984**
- **Character F1: 0.9970**
