"""
Shared parsing helpers for RAGognize response annotations.

Implements the correct nested path:
    details -> annotations -> result -> addressed_user_prompt

Exports one shared helper used by both the adapter and audit scripts,
ensuring consistent extraction across all code paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


AddressedPromptValue = Literal["true", "false", "missing", "invalid"]


@dataclass
class AnnotationResult:
    """Parsed result from details.annotations.result."""
    addressed_user_prompt: AddressedPromptValue = "missing"
    all_valid: bool = True
    cluelessness: bool = False
    completely_hallucinated: bool = False


# Sentinel for absent field
_MISSING = "__MISSING__"


def _normalize_addressed_user_prompt(value) -> AddressedPromptValue:
    """
    Normalise an addressed_user_prompt value from the dataset.

    Rules:
        True / "true" / 1        -> "true"
        False / "false" / 0     -> "false"
        None / absent / "null"  -> "missing"
        anything else               -> "invalid"

    Returns:
        One of "true", "false", "missing", "invalid".
    """
    if value is None:
        return "missing"

    # Boolean
    if isinstance(value, bool):
        return "true" if value else "false"

    # Integer (0/1 from JSON)
    if isinstance(value, int):
        if value == 1:
            return "true"
        if value == 0:
            return "false"
        return "invalid"

    # String — check sentinels first (case-insensitive), then booleans
    if isinstance(value, str):
        stripped = value.strip()
        lower = stripped.lower()
        # Sentinels: case-insensitive match
        if lower in ("null", "", "__missing__", "__null__"):
            return "missing"
        # Known boolean strings (exact match required)
        if stripped == "true":
            return "true"
        if stripped == "false":
            return "false"
        return "invalid"

    return "invalid"


# -------------------------------------------------------------------
# Main helper
# -------------------------------------------------------------------

def parse_annotation_result(response_data: dict) -> AnnotationResult:
    """
    Extract fields from the correct RAGognize nested path.

    Correct path:
        response_data
          -> details
              -> annotations
                  -> result
                      -> addressed_user_prompt
                      -> all_valid
                      -> cluelessness
                      -> completely_hallucinated

    Args:
        response_data: Raw model-response dict from the dataset.

    Returns:
        AnnotationResult with all fields populated; defaults to safe
        values when the path is incomplete.

    Note:
        This is the ONE place where the correct path is implemented.
        All callers (adapter, audit, validation runners) delegate here.
    """
    details = response_data.get("details", {})
    annotations = details.get("annotations", {})
    result = annotations.get("result", {})

    raw_addressed = result.get("addressed_user_prompt", _MISSING)
    addressed = _normalize_addressed_user_prompt(raw_addressed)

    return AnnotationResult(
        addressed_user_prompt=addressed,
        all_valid=result.get("all_valid", True),
        cluelessness=result.get("cluelessness", False),
        completely_hallucinated=result.get("completely_hallucinated", False),
    )


def parse_addressed_user_prompt(response_data_or_raw) -> AddressedPromptValue:
    """
    Convenience wrapper: return the addressed_user_prompt value from a response.

    Supports two call patterns:
      1. parse_addressed_user_prompt(response_dict)    — full model-response dict
      2. parse_addressed_user_prompt(raw_value)       — the bare addressed_user_prompt value

    Args:
        response_data_or_raw: Either a full model-response dict or a bare value
                             extracted from the correct RAGognize path.

    Returns:
        One of "true", "false", "missing", "invalid".
    """
    # Detect whether this is a raw (non-dict) value or a full dict
    if isinstance(response_data_or_raw, dict):
        return parse_annotation_result(response_data_or_raw).addressed_user_prompt
    else:
        # Raw value (e.g. from a test or a bare extraction)
        return _normalize_addressed_user_prompt(response_data_or_raw)
