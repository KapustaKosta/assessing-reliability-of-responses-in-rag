#!/usr/bin/env python
"""
Generate RAGognize Split Manifest CSV.

This script creates a detailed manifest of all expanded responses in the RAGognize dataset,
tracking which responses are present vs missing at the source level.
"""

import sys
sys.path.insert(0, 'src')

import hashlib
import pandas as pd
from pathlib import Path

from ragognize_adapter import (
    RAGognizeAdapter, load_ragognize_dataset,
    create_train_val_split, apply_split, AVAILABLE_MODELS,
)

def compute_file_hash(filepath: str) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()[:16]

def main():
    print("=" * 70)
    print("GENERATING RAGOGNIZE SPLIT MANIFEST")
    print("=" * 70)
    
    # Load dataset
    print("\n1. Loading RAGognize dataset...")
    raw = load_ragognize_dataset()
    split_info = create_train_val_split(raw, val_size=0.15, seed=42)
    adapter = RAGognizeAdapter(models=AVAILABLE_MODELS)
    raw_split = apply_split(raw, split_info)
    unified = adapter.transform_dataset(raw_split)
    
    print(f"   Raw train: {len(raw['train'])} questions")
    print(f"   Raw test: {len(raw['test'])} questions")
    print(f"   Val questions (from split): {len(split_info['val_indices'])}")
    
    # Get raw validation questions
    val_raw = [raw['train'][i] for i in split_info['val_indices']]
    
    # Generate manifest for validation split by checking RAW data
    print("\n2. Generating validation split manifest from RAW data...")
    
    records = []
    theoretical_count = 0
    missing_count = 0
    present_count = 0
    
    # For each raw validation question, check all 4 models
    for row_idx in split_info['val_indices']:
        raw_sample = raw['train'][row_idx]
        question_id = raw_sample['user_prompt_index']
        responses = raw_sample.get('responses', {})
        
        # Check which models are present
        present_models = set(responses.keys()) & set(AVAILABLE_MODELS)
        
        for model_idx, model_name in enumerate(AVAILABLE_MODELS):
            theoretical_count += 1
            
            if model_name in present_models:
                present_count += 1
                # Get faithfulness label from the hallucination spans
                model_response = responses[model_name]
                hallucinations = model_response.get('hallucinations', [])
                # Count valid hallucinations (not marked as invalid)
                valid_hallucinations = [h for h in hallucinations if h.get('valid', True)]
                faithfulness_label = 1 if len(valid_hallucinations) == 0 else 0
                
                # Generate case_id (must match what adapter generates)
                import hashlib
                raw_id = f"val_{row_idx}_{question_id}_{model_name}"
                case_id = f"case_{hashlib.md5(raw_id.encode()).hexdigest()[:16]}"
                
                records.append({
                    'question_id': question_id,
                    'source_row_index': row_idx,
                    'dataset_original_split': 'train',
                    'project_split': 'validation',
                    'source_model': model_name,
                    'response_index': model_idx,
                    'expanded_sample_id': case_id,
                    'faithfulness_label': faithfulness_label,
                    'response_present': True,
                    'missing_reason': '',
                    'split_seed': 42,
                    'n_hallucinations': len(valid_hallucinations),
                })
            else:
                missing_count += 1
                records.append({
                    'question_id': question_id,
                    'source_row_index': row_idx,
                    'dataset_original_split': 'train',
                    'project_split': 'validation',
                    'source_model': model_name,
                    'response_index': model_idx,
                    'expanded_sample_id': '',
                    'faithfulness_label': -1,
                    'response_present': False,
                    'missing_reason': 'source_data_missing',
                    'split_seed': 42,
                    'n_hallucinations': -1,
                })
    
    # Create DataFrame
    df = pd.DataFrame(records)
    
    # Save manifest
    manifest_dir = Path('processed')
    manifest_dir.mkdir(exist_ok=True)
    manifest_path = manifest_dir / 'ragognize_split_manifest.csv'
    df.to_csv(manifest_path, index=False)
    
    # Compute hash
    manifest_hash = compute_file_hash(str(manifest_path))
    
    # Print summary
    print(f"\n3. Manifest Summary:")
    print(f"   Val questions: {len(split_info['val_indices'])}")
    print(f"   Theoretical slots: {theoretical_count}")
    print(f"   Source missing: {missing_count}")
    print(f"   Present responses: {present_count}")
    print(f"   File: {manifest_path}")
    print(f"   SHA-256 (first 16): {manifest_hash}")
    
    # Verify counts
    print(f"\n4. Verification:")
    print(f"   Expected: 277 questions × 4 models = 1108 theoretical")
    print(f"   Actual: {theoretical_count} theoretical, {missing_count} missing, {present_count} present")
    
    # Model breakdown
    print(f"\n5. Missing by model:")
    missing_df = df[df['response_present'] == False]
    if len(missing_df) > 0:
        for model in AVAILABLE_MODELS:
            count = len(missing_df[missing_df['source_model'] == model])
            if count > 0:
                print(f"   {model}: {count}")
    else:
        print("   No missing responses!")
    
    # Faithfulness distribution
    print(f"\n6. Faithfulness distribution:")
    faithful_count = df[df['response_present'] == True]['faithfulness_label'].sum()
    total_present = df[df['response_present'] == True]['faithfulness_label'].count()
    print(f"   Faithful (1): {int(faithful_count)}")
    print(f"   Unfaithful (0): {int(total_present - faithful_count)}")
    
    # Save summary as JSON for easy reading
    summary = {
        'theoretical_slots': int(theoretical_count),
        'source_missing': int(missing_count),
        'actual_valid': int(present_count),
        'val_questions': len(split_info['val_indices']),
        'manifest_path': str(manifest_path),
        'manifest_hash': manifest_hash,
        'split_seed': 42,
        'by_model': {
            model: {
                'total': int(len(df[df['source_model'] == model])),
                'present': int(len(df[(df['source_model'] == model) & (df['response_present'] == True)])),
                'missing': int(len(df[(df['source_model'] == model) & (df['response_present'] == False)])),
            }
            for model in AVAILABLE_MODELS
        },
        'faithfulness': {
            'faithful': int(faithful_count),
            'unfaithful': int(total_present - faithful_count),
        }
    }
    
    import json
    summary_path = manifest_dir / 'ragognize_split_manifest_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n7. Summary saved to: {summary_path}")
    
    return summary

if __name__ == '__main__':
    main()
