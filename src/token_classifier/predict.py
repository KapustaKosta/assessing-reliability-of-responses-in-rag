"""
Prediction script for single samples.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from token_classifier.config import TokenClassifierConfig
from token_classifier.model import (
    TokenHallucinationClassifier,
    load_tokenizer_and_model,
    get_device,
)
from token_classifier.labeling import AnswerTokenizer
from token_classifier.postprocess import (
    tokens_to_spans,
    TokenPrediction,
    extract_answer_tokens_from_offsets,
    compute_answer_score,
    predict_answer_hallucination,
)
from token_classifier.checkpoint import CheckpointManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Prediction
# =============================================================================

@torch.no_grad()
def predict_sample(
    model: TokenHallucinationClassifier,
    tokenizer,
    answer_tokenizer: AnswerTokenizer,
    context: str,
    question: str,
    answer: str,
    threshold: float = 0.5,
    device: torch.device = None,
) -> dict:
    """
    Predict hallucination for a single sample.
    
    Args:
        model: Trained model
        tokenizer: HuggingFace tokenizer
        answer_tokenizer: Answer tokenizer with offset tracking
        context: Context text
        question: Question text
        answer: Answer text
        threshold: Classification threshold
        device: Device to use
    
    Returns:
        Prediction dict with:
        - answer: original answer
        - threshold: used threshold
        - answer_score: answer-level hallucination score
        - tokens: list of token predictions
        - predicted_spans: merged hallucination spans
    """
    if device is None:
        device = model.device
    
    model.eval()
    
    # Tokenize sample
    windows = answer_tokenizer.tokenize_sample(context, question, answer)
    
    # Collect predictions from all windows
    all_token_probs = []
    all_answer_offsets = []
    
    for window in windows:
        # Create input
        input_ids, _ = answer_tokenizer.create_input_ids(
            window["context_ids"],
            window["question_ids"],
            window["answer_ids"],
        )
        
        # Pad/truncate
        max_len = model.config.max_length if hasattr(model.config, 'max_length') else 512
        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len]
        
        padding_length = max_len - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
        
        attention_mask = [1 if tid != tokenizer.pad_token_id else 0 for tid in input_ids]
        
        # Predict
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
        mask_tensor = torch.tensor([attention_mask], dtype=torch.long).to(device)
        
        outputs = model(input_tensor, mask_tensor)
        logits = outputs["logits"].cpu()[0]  # [seq_len, 2]
        
        # Get answer token probabilities
        answer_start = window["answer_start_idx"]
        answer_count = window["answer_token_count"]
        answer_offsets = window["answer_offsets"]
        
        probs = torch.softmax(logits, dim=-1)[:, 1].numpy()
        
        # Extract answer token info
        for i, (start, end) in enumerate(answer_offsets):
            token_start = answer_start + i
            if token_start < len(probs):
                all_token_probs.append(float(probs[token_start]))
                all_answer_offsets.append((start, end))
    
    # Aggregate across windows (simple max for now)
    if not all_token_probs:
        return {
            "answer": answer,
            "threshold": threshold,
            "answer_score": 0.0,
            "tokens": [],
            "predicted_spans": [],
        }
    
    # For multiple windows, we'd need more sophisticated aggregation
    # For now, just use the last window's predictions
    # (A proper implementation would track which answer chars were in which window)
    
    # Create token predictions
    tokens = []
    for i, (start, end) in enumerate(all_answer_offsets):
        if i < len(all_token_probs):
            tokens.append(TokenPrediction(
                text=answer[start:end],
                start=start,
                end=end,
                p_hallucination=all_token_probs[i],
                predicted_label=0,  # Will be set by tokens_to_spans
            ))
    
    # Convert to spans
    predicted_spans = tokens_to_spans(tokens, threshold=threshold)
    
    # Compute answer score
    answer_score = compute_answer_score(all_token_probs, mode="max")
    
    return {
        "answer": answer,
        "threshold": threshold,
        "answer_score": float(answer_score),
        "tokens": [
            {
                "text": t.text,
                "start": t.start,
                "end": t.end,
                "p_hallucination": float(t.p_hallucination),
                "predicted_label": int(t.predicted_label),
            }
            for t in tokens
        ],
        "predicted_spans": predicted_spans,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Predict hallucination for samples")
    
    # Model
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to checkpoint")
    parser.add_argument("--device", type=str, default="auto")
    
    # Input
    parser.add_argument("--context", type=str, help="Context text")
    parser.add_argument("--question", type=str, help="Question text")
    parser.add_argument("--answer", type=str, help="Answer text")
    parser.add_argument("--input_json", type=str, help="JSON file with samples")
    
    # Output
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", type=str, help="Output JSON file")
    
    args = parser.parse_args()
    
    # Load checkpoint
    checkpoint_path = Path(args.checkpoint_path)
    checkpoint_manager = CheckpointManager(str(checkpoint_path.parent))
    checkpoint_manager.checkpoint_path = checkpoint_path
    
    config = checkpoint_manager.load_config()
    config.device = args.device
    
    # Load model
    logger.info("Loading model...")
    tokenizer, model = load_tokenizer_and_model(config)
    device = get_device(config.device)
    model = model.to(device)
    checkpoint_manager.load(model)
    
    # Create answer tokenizer
    answer_tokenizer = AnswerTokenizer(
        tokenizer,
        max_length=config.max_length,
        context_max_length=config.context_max_length,
        context_stride=config.context_stride,
    )
    
    # Process input
    if args.input_json:
        # Load from JSON file
        with open(args.input_json, "r") as f:
            data = json.load(f)
        
        samples = data if isinstance(data, list) else [data]
        
        results = []
        for sample in samples:
            result = predict_sample(
                model, tokenizer, answer_tokenizer,
                context=sample.get("context", ""),
                question=sample.get("question", ""),
                answer=sample.get("answer", ""),
                threshold=args.threshold,
                device=device,
            )
            results.append(result)
    
    elif args.context and args.question and args.answer:
        # Single sample from CLI
        result = predict_sample(
            model, tokenizer, answer_tokenizer,
            context=args.context,
            question=args.question,
            answer=args.answer,
            threshold=args.threshold,
            device=device,
        )
        results = [result]
    
    else:
        raise ValueError("Must provide --input_json or --context, --question, --answer")
    
    # Output
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output}")
    else:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    
    return results


if __name__ == "__main__":
    main()
