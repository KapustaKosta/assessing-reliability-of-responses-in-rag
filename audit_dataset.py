#!/usr/bin/env python
"""Quick dataset audit script."""

import sys
sys.path.insert(0, 'src')

from datasets import load_dataset
import json

print("Loading RAGognize dataset from cache...")

# Load from cache
ds = load_dataset("F4biian/RAGognize", cache_dir="data/cache/huggingface")
print(f"Loaded! Keys: {list(ds.keys())}")

for split_name, split_ds in ds.items():
    print(f"\n{split_name}: {len(split_ds)} samples")
    print(f"Columns: {split_ds.column_names[:15]}...")

# Check test split
test_ds = ds['test']
print(f"\nTest split size: {len(test_ds)}")

# Expanded responses
print("\nExpanded responses:")
print(f"  Test expanded: {len(test_ds) * 4} (4 models per question)")

# Check addressed_user_prompt in test
if 'addressed_user_prompt' in test_ds.column_names:
    print("\nChecking addressed_user_prompt in test...")
    vals = [test_ds[i].get('addressed_user_prompt') for i in range(min(100, len(test_ds)))]
    non_null = [v for v in vals if v is not None]
    print(f"  Non-null in first 100: {len(non_null)}")
    if non_null:
        unique = set(non_null[:50])
        print(f"  Sample values: {list(unique)[:5]}")
        dist = {}
        for v in non_null:
            dist[v] = dist.get(v, 0) + 1
        print(f"  Distribution: {dist}")
else:
    print("\naddressed_user_prompt NOT in test columns")
    print(f"  Available: {[c for c in test_ds.column_names if 'prompt' in c.lower() or 'address' in c.lower()]}")

# Save manifest
manifest = {
    "splits": {name: len(ds[name]) for name in ds},
    "expanded_responses": {name: len(ds[name]) * 4 for name in ds},
    "columns": ds['test'].column_names,
}
with open("data/cache/ragognize_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)
print("\nManifest saved to data/cache/ragognize_manifest.json")
