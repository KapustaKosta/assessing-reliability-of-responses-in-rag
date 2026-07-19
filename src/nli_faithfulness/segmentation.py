"""
Answer segmentation into sentence units.

This module splits Russian answers into sentence-level units for granular
evidence assessment. Note: These are sentence units, not necessarily atomic
claims. A single claim may span multiple sentences, and a single sentence
may contain multiple claims.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .constants import (
    SENTENCE_SPLIT_PATTERN,
    MIN_SENTENCE_LENGTH,
    IMPORTANT_CONTENT_PATTERNS,
)


@dataclass
class SentenceUnit:
    """Represents a sentence-level unit extracted from an answer."""
    case_id: str
    sentence_id: int  # 0-indexed within the answer
    text: str
    has_important_content: bool = False
    # Metadata for debugging
    original_position: int = 0  # Character position in original answer


@dataclass 
class AnswerSegments:
    """Container for all sentence units from one answer."""
    case_id: str
    answer_text: str
    units: list[SentenceUnit] = field(default_factory=list)
    
    def __len__(self) -> int:
        return len(self.units)
    
    def __iter__(self):
        return iter(self.units)


def _has_important_content(text: str) -> bool:
    """Check if text contains important content markers."""
    for pattern in IMPORTANT_CONTENT_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def _clean_sentence(text: str) -> str:
    """Clean a sentence by removing excessive whitespace."""
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def _should_merge(short_sentence: str, prev_sentence: str, next_sentence: str) -> bool:
    """
    Determine if a short sentence should be merged with adjacent sentences.
    
    We want to preserve sentences with important content (numbers, banking terms,
    conditions, etc.) even if they're short. But truly empty or meaningless
    sentences should be merged.
    """
    # If it's whitespace only, always merge
    if not short_sentence.strip():
        return True
    
    # If it has important content, don't merge
    if _has_important_content(short_sentence):
        return False
    
    # Check if the sentence is meaningful even if short
    # Russian sentences typically have at least a verb and subject
    word_count = len(short_sentence.split())
    
    # Very short sentences (< 3 words) without important content should merge
    if word_count < 3:
        return True
    
    # Sentences with pronouns or short responses might be valid
    # e.g., "Да.", "Нет.", "Хорошо."
    short_affirmative = re.match(r'^(?:да|нет|ладно|хорошо|конечно)\.?$', 
                                  short_sentence.strip().lower())
    if short_affirmative:
        return False  # Keep short affirmative responses
    
    return True


def split_answer_into_units(
    case_id: str,
    answer: str,
) -> AnswerSegments:
    """
    Split an answer into sentence-level units.
    
    The splitting is done on sentence-ending punctuation (. ? !), with
    newline handling. Short sentences may be merged with adjacent ones if
    they don't contain important content (numbers, conditions, etc.).
    
    Args:
        case_id: Unique identifier for this sample
        answer: The answer text to split
        
    Returns:
        AnswerSegments containing all sentence units
    """
    if not answer or not answer.strip():
        return AnswerSegments(case_id=case_id, answer_text=answer, units=[])
    
    # Split on sentence boundaries
    # Keep the delimiter attached to the sentence
    parts = re.split(SENTENCE_SPLIT_PATTERN, answer)
    
    units = []
    current_position = 0
    
    for i, part in enumerate(parts):
        original_position = answer.find(part, current_position)
        if original_position == -1:
            original_position = current_position
        
        cleaned = _clean_sentence(part)
        current_position = original_position + len(part)
        
        if not cleaned:
            continue
            
        units.append(SentenceUnit(
            case_id=case_id,
            sentence_id=i,
            text=cleaned,
            has_important_content=_has_important_content(cleaned),
            original_position=original_position,
        ))
    
    # Merge short sentences with neighbors
    merged_units = _merge_short_sentences(units)
    
    # Re-index sentence_ids after merging
    for i, unit in enumerate(merged_units):
        unit.sentence_id = i
    
    return AnswerSegments(
        case_id=case_id,
        answer_text=answer,
        units=merged_units,
    )


def _merge_short_sentences(units: list[SentenceUnit]) -> list[SentenceUnit]:
    """
    Merge short sentences without important content into adjacent sentences.
    
    This uses a forward-fill approach: short sentences are merged with the
    previous sentence, unless they are the first sentence.
    """
    if len(units) <= 1:
        return units
    
    merged = []
    i = 0
    
    while i < len(units):
        current = units[i]
        
        # Check if current should be merged with previous
        if (merged and 
            _should_merge(current.text, merged[-1].text, 
                         units[i + 1].text if i + 1 < len(units) else "")):
            # Merge with previous
            prev = merged.pop()
            
            # Preserve original position of the first sentence
            merged_text = f"{prev.text} {current.text}"
            has_important = prev.has_important_content or current.has_important_content
            
            merged.append(SentenceUnit(
                case_id=current.case_id,
                sentence_id=prev.sentence_id,  # Will be re-indexed later
                text=merged_text,
                has_important_content=has_important,
                original_position=prev.original_position,
            ))
        else:
            merged.append(current)
        
        i += 1
    
    return merged


def segment_dataset(samples: list) -> dict[str, AnswerSegments]:
    """
    Segment all answers in a dataset.
    
    Args:
        samples: List of SampleData objects
        
    Returns:
        Dictionary mapping case_id to AnswerSegments
    """
    segments = {}
    for sample in samples:
        segments[sample.case_id] = split_answer_into_units(
            case_id=sample.case_id,
            answer=sample.answer,
        )
    return segments


def get_segment_stats(segments: dict[str, AnswerSegments]) -> dict:
    """Calculate statistics about segmented answers."""
    if not segments:
        return {"total_answers": 0, "total_units": 0}
    
    total_answers = len(segments)
    total_units = sum(len(seg) for seg in segments.values())
    
    units_per_answer = [len(seg) for seg in segments.values()]
    
    return {
        "total_answers": total_answers,
        "total_units": total_units,
        "avg_units_per_answer": total_units / total_answers if total_answers > 0 else 0,
        "min_units": min(units_per_answer) if units_per_answer else 0,
        "max_units": max(units_per_answer) if units_per_answer else 0,
    }
