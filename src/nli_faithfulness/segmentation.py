"""
Answer segmentation into sentence/claim units.

This module splits answers into sentence-level or claim-level units for granular
evidence assessment. Sentence splitting is the base implementation; atomic claim
extraction interface is preserved for future enhancement.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .constants import (
    SENTENCE_SPLIT_PATTERN,
    MIN_SENTENCE_LENGTH,
    IMPORTANT_CONTENT_PATTERNS,
)


@dataclass
class ClaimUnit:
    """
    Represents a claim-level unit extracted from an answer.
    
    Each claim has a unique identifier, character positions in the original answer,
    and the claim text itself.
    """
    case_id: str
    claim_id: int  # 0-indexed within the answer
    char_start: int  # Character start position in original answer
    char_end: int    # Character end position in original answer
    text: str       # The claim text
    has_important_content: bool = False


@dataclass 
class AnswerSegments:
    """Container for all claim units from one answer."""
    case_id: str
    answer_text: str
    claims: list[ClaimUnit] = field(default_factory=list)
    
    def __len__(self) -> int:
        return len(self.claims)
    
    def __iter__(self):
        return iter(self.claims)


def _has_important_content(text: str) -> bool:
    """Check if text contains important content markers."""
    for pattern in IMPORTANT_CONTENT_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def _clean_sentence(text: str) -> str:
    """Clean a sentence by removing excessive whitespace."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _should_merge(short_sentence: str, prev_sentence: str, next_sentence: str) -> bool:
    """
    Determine if a short sentence should be merged with adjacent sentences.
    """
    if not short_sentence.strip():
        return True
    
    if _has_important_content(short_sentence):
        return False
    
    word_count = len(short_sentence.split())
    
    if word_count < 3:
        return True
    
    short_affirmative = re.match(r'^(?:да|нет|ладно|хорошо|конечно)\.?$', 
                                  short_sentence.strip().lower())
    if short_affirmative:
        return False
    
    return True


def split_answer_into_units(
    case_id: str,
    answer: str,
) -> AnswerSegments:
    """
    Split an answer into claim-level units.
    
    The splitting is done on sentence-ending punctuation (. ? !), with
    newline handling. Short sentences may be merged with adjacent ones if
    they don't contain important content (numbers, conditions, etc.).
    
    Args:
        case_id: Unique identifier for this sample
        answer: The answer text to split
        
    Returns:
        AnswerSegments containing all claim units with character positions
    """
    if not answer or not answer.strip():
        return AnswerSegments(case_id=case_id, answer_text=answer, claims=[])
    
    # Split on sentence boundaries
    parts = re.split(SENTENCE_SPLIT_PATTERN, answer)
    
    claims = []
    current_position = 0
    
    for i, part in enumerate(parts):
        # Find the actual position of this part in the original answer
        # Handle leading whitespace/newlines
        original_position = answer.find(part.lstrip(), current_position)
        if original_position == -1:
            original_position = current_position
        
        cleaned = _clean_sentence(part)
        
        # Calculate end position
        if cleaned:
            # Find where this cleaned text appears in the original
            search_start = original_position
            end_position = answer.find(cleaned, search_start)
            if end_position == -1:
                end_position = original_position + len(cleaned)
            else:
                end_position = end_position + len(cleaned)
        else:
            end_position = original_position
        
        current_position = original_position + len(part)
        
        if not cleaned:
            continue
            
        claims.append(ClaimUnit(
            case_id=case_id,
            claim_id=i,
            char_start=original_position,
            char_end=end_position,
            text=cleaned,
            has_important_content=_has_important_content(cleaned),
        ))
    
    # Merge short sentences with neighbors
    merged_claims = _merge_short_sentences(claims)
    
    # Re-index claim_ids after merging
    for i, claim in enumerate(merged_claims):
        claim.claim_id = i
    
    return AnswerSegments(
        case_id=case_id,
        answer_text=answer,
        claims=merged_claims,
    )


def _merge_short_sentences(claims: list[ClaimUnit]) -> list[ClaimUnit]:
    """
    Merge short sentences without important content into adjacent sentences.
    
    Uses forward-fill: short sentences merge with the previous sentence.
    """
    if len(claims) <= 1:
        return claims
    
    merged = []
    i = 0
    
    while i < len(claims):
        current = claims[i]
        
        if (merged and 
            _should_merge(current.text, merged[-1].text, 
                         claims[i + 1].text if i + 1 < len(claims) else "")):
            # Merge with previous
            prev = merged.pop()
            
            merged_text = f"{prev.text} {current.text}"
            has_important = prev.has_important_content or current.has_important_content
            
            merged.append(ClaimUnit(
                case_id=current.case_id,
                claim_id=prev.claim_id,
                char_start=prev.char_start,
                char_end=current.char_end,
                text=merged_text,
                has_important_content=has_important,
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
        return {"total_answers": 0, "total_claims": 0}
    
    total_answers = len(segments)
    total_claims = sum(len(seg) for seg in segments.values())
    
    claims_per_answer = [len(seg) for seg in segments.values()]
    
    return {
        "total_answers": total_answers,
        "total_claims": total_claims,
        "avg_claims_per_answer": total_claims / total_answers if total_answers > 0 else 0,
        "min_claims": min(claims_per_answer) if claims_per_answer else 0,
        "max_claims": max(claims_per_answer) if claims_per_answer else 0,
    }
