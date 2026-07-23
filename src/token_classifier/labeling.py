"""
Token labeling utilities for hallucination detection.

Converts character-level hallucination spans to token-level labels.
"""

from __future__ import annotations

import logging
from typing import Optional

from .schema import TokenSample, HallucinationSpan

logger = logging.getLogger(__name__)


# =============================================================================
# Span Overlap Detection
# =============================================================================

def span_overlaps_token(
    token_start: int,
    token_end: int,
    span_start: int,
    span_end: int,
) -> bool:
    """
    Check if a token overlaps with a hallucination span.
    
    Uses the rule:
        token_is_hallucinated = max(token_start, span_start) < min(token_end, span_end)
    
    IMPORTANT: Boundary touch does NOT count as overlap.
    For example, if token_end == span_start, the token is NOT hallucinated.
    
    Args:
        token_start: Start character position of token (relative to answer)
        token_end: End character position of token (exclusive)
        span_start: Start of hallucination span
        span_end: End of hallucination span (exclusive)
    
    Returns:
        True if token overlaps with span, False otherwise
    """
    # Boundary touch (token_end == span_start or token_start == span_end) is NOT overlap
    return max(token_start, span_start) < min(token_end, span_end)


def compute_token_label(
    token_start: int,
    token_end: int,
    spans: list[HallucinationSpan],
) -> int:
    """
    Compute the label for a single token.
    
    Label:
        0 = supported (no overlap with any hallucination span)
        1 = hallucinated (overlaps with at least one hallucination span)
    
    Args:
        token_start: Start character position of token
        token_end: End character position of token
        spans: List of hallucination spans
    
    Returns:
        0 for supported, 1 for hallucinated
    """
    for span in spans:
        if not span.valid:
            continue
        if span_overlaps_token(token_start, token_end, span.start, span.end):
            return 1
    return 0


# =============================================================================
# Token Labeling
# =============================================================================

def compute_token_labels(
    answer: str,
    spans: list[HallucinationSpan],
    answer_offsets: list[tuple[int, int]],
) -> list[int]:
    """
    Compute token-level labels for all tokens.
    
    Args:
        answer: The answer text
        spans: List of hallucination spans (relative to answer)
        answer_offsets: List of (start, end) character offsets for each answer token
    
    Returns:
        List of labels (0=supported, 1=hallucinated) for each token
    """
    labels = []
    for token_start, token_end in answer_offsets:
        label = compute_token_label(token_start, token_end, spans)
        labels.append(label)
    return labels


# =============================================================================
# Tokenization with Offset Tracking
# =============================================================================

class AnswerTokenizer:
    """
    Tokenizer that tracks character offsets for answer tokens.
    
    Implements three-part encoding:
        [CLS] Context [SEP] Question [SEP] Answer [SEP]
    
    With priority:
        1. Full Answer (required)
        2. Full Question (preferred)
        3. Remaining budget for Context
    """
    
    def __init__(
        self,
        tokenizer,
        max_length: int = 512,
        answer_max_length: Optional[int] = None,
        context_max_length: int = 400,
        context_stride: int = 128,
        answer_essential: bool = True,
    ):
        """
        Initialize tokenizer.
        
        Args:
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            answer_max_length: Max tokens for answer (None = no limit)
            context_max_length: Max tokens for context
            context_stride: Stride for context sliding window
            answer_essential: If True, never truncate answer
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.answer_max_length = answer_max_length
        self.context_max_length = context_max_length
        self.context_stride = context_stride
        self.answer_essential = answer_essential
    
    def _tokenize_with_offsets(
        self,
        text: str,
        add_special_tokens: bool = False,
    ) -> tuple[list[int], list[tuple[int, int]]]:
        """
        Tokenize text and return token IDs with character offsets.
        
        Returns:
            Tuple of (token_ids, offsets) where offsets are (start, end) character positions
        """
        # Tokenize with offset mapping
        encoding = self.tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            return_offsets_mapping=True,
        )
        
        token_ids = encoding["input_ids"]
        offset_mapping = encoding["offset_mapping"]
        
        # Filter out special tokens if add_special_tokens=False
        if not add_special_tokens:
            # offset (0, 0) indicates special token
            filtered_ids = []
            filtered_offsets = []
            for tid, (start, end) in zip(token_ids, offset_mapping):
                if start != 0 or end != 0:  # Not a special token
                    filtered_ids.append(tid)
                    filtered_offsets.append((start, end))
            return filtered_ids, filtered_offsets
        
        return token_ids, offset_mapping
    
    def tokenize_sample(
        self,
        context: str,
        question: str,
        answer: str,
    ) -> list[dict]:
        """
        Tokenize a sample with three-part encoding.
        
        Budget priority:
            1. Full Answer (never truncated)
            2. Full Question
            3. Context (may be truncated or windowed)
        
        Args:
            context: Context text
            question: Question text
            answer: Answer text
        
        Returns:
            List of windows, each with:
            - input_ids: Token IDs for the window
            - attention_mask: Attention mask
            - answer_token_ids: Token IDs belonging to answer
            - answer_offsets: Character offsets for answer tokens (relative to answer)
            - answer_start_idx: Index where answer tokens start in the sequence
            - window_id: Window index
        """
        # Tokenize answer first (must be complete)
        answer_ids, answer_offsets = self._tokenize_with_offsets(answer)
        num_answer_tokens = len(answer_ids)
        
        if self.answer_max_length and num_answer_tokens > self.answer_max_length:
            if self.answer_essential:
                logger.warning(
                    f"Answer has {num_answer_tokens} tokens, exceeding max {self.answer_max_length}. "
                    "Truncating answer may cause incorrect labels!"
                )
            answer_ids = answer_ids[:self.answer_max_length]
            answer_offsets = answer_offsets[:self.answer_max_length]
            num_answer_tokens = self.answer_max_length
        
        # Tokenize question
        question_ids, question_offsets = self._tokenize_with_offsets(question)
        num_question_tokens = len(question_ids)
        
        # Calculate budget for context
        # Reserve space for special tokens: [CLS] ... [SEP] ... [SEP] ... [SEP]
        # At minimum: [CLS] + answer + [SEP] + question + [SEP] + context + [SEP]
        # Rough estimate: 4 special tokens + answer + question
        reserved = 4 + num_answer_tokens + num_question_tokens
        remaining_budget = self.max_length - reserved
        
        if remaining_budget < 0:
            raise ValueError(
                f"Answer ({num_answer_tokens}) + Question ({num_question_tokens}) exceed "
                f"max_length ({self.max_length}). Reduce max_length or truncate question."
            )
        
        # Tokenize context
        context_ids, context_offsets = self._tokenize_with_offsets(context)
        
        # Check if context needs windowing
        if len(context_ids) <= remaining_budget:
            # Context fits, no windowing needed
            windows = [{
                "context_ids": context_ids,
                "context_offsets": context_offsets,
                "question_ids": question_ids,
                "question_offsets": question_offsets,
                "answer_ids": answer_ids,
                "answer_offsets": answer_offsets,
                "window_id": 0,
                "num_windows": 1,
            }]
        else:
            # Need to create context windows
            windows = self._create_context_windows(
                context_ids, context_offsets,
                question_ids, question_offsets,
                answer_ids, answer_offsets,
                remaining_budget,
            )
        
        return windows
    
    def _create_context_windows(
        self,
        context_ids: list[int],
        context_offsets: list[tuple[int, int]],
        question_ids: list[int],
        question_offsets: list[tuple[int, int]],
        answer_ids: list[int],
        answer_offsets: list[tuple[int, int]],
        budget: int,
    ) -> list[dict]:
        """
        Create sliding window for context.
        
        Each window includes the full question and answer, with a sliding
        context portion.
        """
        windows = []
        num_context_tokens = len(context_ids)
        num_answer_tokens = len(answer_ids)
        num_question_tokens = len(question_ids)
        
        # Calculate window size (must leave room for question + answer + special tokens)
        # Special tokens per window: [CLS] context [SEP] question [SEP] answer [SEP] = 4
        window_size = budget  # Max tokens per window
        
        stride = self.context_stride
        
        start = 0
        window_id = 0
        while start < num_context_tokens:
            end = min(start + window_size, num_context_tokens)
            
            # Create window
            windows.append({
                "context_ids": context_ids[start:end],
                "context_offsets": context_offsets[start:end],
                "question_ids": question_ids,
                "question_offsets": question_offsets,
                "answer_ids": answer_ids,
                "answer_offsets": answer_offsets,
                "window_id": window_id,
                "num_windows": 0,  # Will be updated
                "context_start_char": context_offsets[start][0] if start < len(context_offsets) else 0,
            })
            
            window_id += 1
            start += stride
        
        # Update num_windows for all windows
        num_windows = len(windows)
        for w in windows:
            w["num_windows"] = num_windows
        
        return windows
    
    def create_input_ids(
        self,
        context_ids: list[int],
        question_ids: list[int],
        answer_ids: list[int],
    ) -> tuple[list[int], list[int]]:
        """
        Combine token IDs into a single sequence.
        
        Format: [CLS] context [SEP] question [SEP] answer [SEP]
        
        Returns:
            (input_ids, answer_token_mask) where answer_token_mask[i] = 1 if token i is from answer
        """
        tokenizer = self.tokenizer
        
        # Build sequence: [CLS] + context + [SEP] + question + [SEP] + answer + [SEP]
        sep_token = tokenizer.sep_token or "[SEP]"
        cls_token = tokenizer.cls_token or "[CLS]"
        
        sep_id = tokenizer.convert_tokens_to_ids([sep_token])[0]
        cls_id = tokenizer.convert_tokens_to_ids([cls_token])[0]
        
        input_ids = [cls_id] + context_ids + [sep_id] + question_ids + [sep_id] + answer_ids + [sep_id]
        
        # Create mask: 0 for special/context/question, 1 for answer, -100 for ignore
        answer_start = 1 + len(context_ids) + 1 + len(question_ids) + 1  # After [CLS] context [SEP] question [SEP]
        labels = []
        for i in range(len(input_ids)):
            if i < answer_start:
                labels.append(-100)  # Special token, context, or question
            elif i < answer_start + len(answer_ids):
                labels.append(0)  # Answer token (will be replaced with actual labels)
            else:
                labels.append(-100)  # Padding or ending special token
        
        return input_ids, labels
    
    def tokenize_sample_with_labels(
        self,
        sample: TokenSample,
    ) -> list[dict]:
        """
        Tokenize a sample and compute labels for answer tokens.
        
        Returns list of windows, each with tokenization and labels.
        """
        windows = self.tokenize_sample(
            sample.context,
            sample.question,
            sample.answer,
        )
        
        # Compute labels for each window
        for window in windows:
            # Build full sequence
            input_ids, labels = self.create_input_ids(
                window["context_ids"],
                window["question_ids"],
                window["answer_ids"],
            )
            
            # Replace answer labels with actual hallucination labels
            answer_offsets = window["answer_offsets"]
            answer_start_in_seq = 1 + len(window["context_ids"]) + 1 + len(window["question_ids"]) + 1
            
            for i, (ans_start, ans_end) in enumerate(answer_offsets):
                label = compute_token_label(ans_start, ans_end, sample.hallucination_spans)
                labels[answer_start_in_seq + i] = label
            
            window["input_ids"] = input_ids
            window["labels"] = labels
            window["answer_start_idx"] = answer_start_in_seq
            window["answer_token_count"] = len(answer_offsets)
            
            # Add token details for analysis
            window["answer_tokens"] = [
                {
                    "text": sample.answer[ans_start:ans_end],
                    "start": ans_start,
                    "end": ans_end,
                    "label": compute_token_label(ans_start, ans_end, sample.hallucination_spans),
                }
                for ans_start, ans_end in answer_offsets
            ]
        
        return windows


# =============================================================================
# Synthetic Data Generation for Testing
# =============================================================================

def create_synthetic_sample(
    sample_id: str,
    question_id: str,
    answer: str,
    hallucination_spans: list[tuple[int, int]],
    context: str = "This is the context.",
    question: str = "What is the answer?",
    source_model: str = "synthetic",
) -> TokenSample:
    """
    Create a synthetic sample for testing.
    
    Args:
        sample_id: Unique sample ID
        question_id: Question ID
        answer: Answer text
        hallucination_spans: List of (start, end) for hallucination spans
        context: Context text
        question: Question text
        source_model: Source model name
    
    Returns:
        TokenSample with hallucination spans
    """
    spans = [
        HallucinationSpan(
            start=start,
            end=end,
            text=answer[start:end],
            valid=True,
        )
        for start, end in hallucination_spans
    ]
    
    return TokenSample(
        sample_id=sample_id,
        question_id=question_id,
        context=context,
        question=question,
        answer=answer,
        hallucination_spans=spans,
        split="train",
        source_model=source_model,
    )
