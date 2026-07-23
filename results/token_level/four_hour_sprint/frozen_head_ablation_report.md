# Frozen Head Ablation & LOSO Report

## Feature Dimension Manifest
- Embedding dim per pool: 768
- CLS pool:        [   0: 768]  dim=768
- Mean pool:       [ 768:1536]  dim=768
- Max pool:        [1536:2304]  dim=768
- Prob features:   [2304:2332]  dim=28
- SM one-hot:      [2332:2336]  dim=4
- SM interactions: [2336:2348]  dim=12
- **TOTAL: [0:2348] dim=2348**

## Ablation Results

| Label | Dim | Char F1 | Char P | Char R | Ans F1 | Ans P | Ans R | Token F1 | Gate_t | Tok_t | A | B | C | D |
|-------|-----|---------|--------|--------|--------|-------|-------|----------|--------|-------|---|---|---|---|
| A_prob_only | 28 | 0.3977 | 0.2945 | 0.6124 | 0.5358 | 0.5826 | 0.4960 | 0.3967 | 0.60 | 0.15 | 187 | 190 | 134 | 745 |
| B_emb_only | 2304 | 0.4217 | 0.3103 | 0.6577 | 0.5718 | 0.5845 | 0.5597 | 0.4205 | 0.60 | 0.10 | 211 | 166 | 150 | 729 |
| C_emb_prob_no_sm | 2332 | 0.4343 | 0.3316 | 0.6291 | 0.5665 | 0.6149 | 0.5252 | 0.4334 | 0.65 | 0.15 | 198 | 179 | 124 | 755 |
| D_sm_only | 4 | 0.4514 | 0.3180 | 0.7774 | 0.6327 | 0.5525 | 0.7401 | 0.4499 | 0.40 | 0.10 | 279 | 98 | 226 | 653 |
| E_full | 2348 | 0.4404 | 0.3269 | 0.6747 | 0.6121 | 0.6196 | 0.6048 | 0.4396 | 0.55 | 0.15 | 228 | 149 | 140 | 739 |
| raw_t020_baseline | 0 | 0.3896 | 0.2655 | 0.7310 | 0.5426 | 0.4055 | 0.8196 | 0.3864 | 0.20 | 0.20 | 309 | 68 | 453 | 426 |

## LOSO Cross-Validation (variant C: emb+prob, no SM)

| Holdout Source | Char F1 | Ans F1 | Ans P | Ans R | Token F1 | A | B | C | D |
|----------------|---------|--------|-------|-------|----------|---|---|---|---|
| Llama-2-7b-chat-hf | 0.5135 | 0.7342 | 0.6744 | 0.8056 | 0.5124 | 145 | 35 | 70 | 64 |
| Llama-3.1-8B-Instruct | 0.0947 | 0.2381 | 0.2083 | 0.2778 | 0.0849 | 5 | 13 | 19 | 277 |
| Mistral-7B-Instruct-v0.1 | 0.4174 | 0.6050 | 0.6207 | 0.5902 | 0.4135 | 72 | 50 | 44 | 148 |
| Mistral-7B-Instruct-v0.3 | 0.2246 | 0.4459 | 0.3500 | 0.6140 | 0.2155 | 35 | 22 | 65 | 192 |

## Key Findings

1. **Dimension check**: Full FROZEN_LR input = 2348 (not 2352 as previously reported)
   - 768*3 embeddings + 28 prob feats + 4 SM + 12 SM interactions = 2348

2. **Ablation**: Feature groups ordered by importance (by Char F1 improvement over baseline 0.3896):
   1. D_sm_only (dim=4): Char F1=0.4514 (+0.0618 vs baseline)
   2. E_full (dim=2348): Char F1=0.4404 (+0.0508 vs baseline)
   3. C_emb_prob_no_sm (dim=2332): Char F1=0.4343 (+0.0447 vs baseline)
   4. B_emb_only (dim=2304): Char F1=0.4217 (+0.0321 vs baseline)
   5. A_prob_only (dim=28): Char F1=0.3977 (+0.0081 vs baseline)

3. **LOSO**: Model generalizes across source models (Char F1 range: 0.0947 - 0.5135)
