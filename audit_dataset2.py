#!/usr/bin/env python
"""Deep dataset audit - check details, splits, and addressed_user_prompt."""

import sys
sys.path.insert(0, 'src')

from datasets import load_dataset
import json

print("Loading RAGognize dataset...")

ds = load_dataset("F4biian/RAGognize", cache_dir="data/cache/huggingface")

# Check details column structure
print("\n=== Checking details column ===")
test_sample = ds['test'][0]
details = test_sample.get('details', {})
print(f"details type: {type(details)}")
if isinstance(details, dict):
    print(f"details keys: {list(details.keys())[:20]}")
    
    # Check if addressed_user_prompt is inside details
    if 'addressed_user_prompt' in details:
        print("addressed_user_prompt FOUND in details")
    else:
        print("addressed_user_prompt NOT in details")
        print(f"Available detail keys: {list(details.keys())}")

# Check responses structure
print("\n=== Checking responses column ===")
responses = test_sample.get('responses', [])
print(f"responses type: {type(responses)}")
if isinstance(responses, list):
    print(f"responses length: {len(responses)}")
    if responses:
        print(f"First response keys: {list(responses[0].keys()) if isinstance(responses[0], dict) else responses[0]}")
        # Check for model names
        for r in responses[:4]:
            if isinstance(r, dict):
                model = r.get('model', 'UNKNOWN')
                print(f"  Model: {model}")

# Now let's manually compute the validation split
print("\n=== Manual validation split ===")
from sklearn.model_selection import train_test_split
import numpy as np

# Get all train indices
train_indices = list(range(len(ds['train'])))
np.random.seed(42)
train_idx, val_idx = train_test_split(train_indices, test_size=0.15, random_state=42)

print(f"Train indices: {len(train_idx)}")
print(f"Val indices: {len(val_idx)}")

# But we need to account for the fact that RAGognize has 4 responses per question
# The split should be done at the question level, then expanded
# Let's check how the adapter does it

print("\n=== Expected validation samples ===")
# If 276 questions in val (15% of 1842), and 4 models per question
n_val_questions = len(val_idx)
n_val_expanded = n_val_questions * 4
print(f"Val questions: {n_val_questions}")
print(f"Val expanded: {n_val_expanded}")

# Check addressed_user_prompt in all responses
print("\n=== Checking addressed_user_prompt in train details ===")
train_sample = ds['train'][0]
train_details = train_sample.get('details', {})
if isinstance(train_details, dict):
    if 'addressed_user_prompt' in train_details:
        print("addressed_user_prompt FOUND in train details")
        # Sample values
        vals = []
        for i in range(min(100, len(ds['train']))):
            d = ds['train'][i].get('details', {})
            if isinstance(d, dict) and 'addressed_user_prompt' in d:
                vals.append(d['addressed_user_prompt'])
        if vals:
            unique = set(vals)
            print(f"Unique values: {unique}")
            dist = {}
            for v in vals:
                dist[str(v)] = dist.get(str(v), 0) + 1
            print(f"Distribution: {dist}")
    else:
        print("addressed_user_prompt NOT in train details")
        print(f"Available detail keys: {list(train_details.keys())}")
