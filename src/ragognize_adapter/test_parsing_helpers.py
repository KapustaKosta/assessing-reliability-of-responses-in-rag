"""
Tests for the addressed_user_prompt parsing helper.

Covers:
1. addressed_user_prompt=True
2. addressed_user_prompt=False
3. missing annotations
4. missing result
5. null value
6. string values "true" and "false"
7. malformed/unrecognized values
"""

from ragognize_adapter.parsing_helpers import (
    parse_addressed_user_prompt,
    parse_annotation_result,
)


# Sentinel values for "absent"
_MISSING = "__MISSING__"


class TestParseAddressedUserPrompt:
    """Tests for parse_addressed_user_prompt (handles both raw values and dicts)."""

    def test_true_bool(self):
        assert parse_addressed_user_prompt(True) == "true"

    def test_false_bool(self):
        assert parse_addressed_user_prompt(False) == "false"

    def test_string_true(self):
        assert parse_addressed_user_prompt("true") == "true"

    def test_false_bool_false(self):
        assert parse_addressed_user_prompt("false") == "false"

    def test_int_one(self):
        assert parse_addressed_user_prompt(1) == "true"

    def test_int_zero(self):
        assert parse_addressed_user_prompt(0) == "false"

    def test_none(self):
        assert parse_addressed_user_prompt(None) == "missing"

    def test_missing_sentinel(self):
        assert parse_addressed_user_prompt(_MISSING) == "missing"

    def test_null_string(self):
        assert parse_addressed_user_prompt("null") == "missing"

    def test_empty_string(self):
        assert parse_addressed_user_prompt("") == "missing"

    def test_string_uppercase_true(self):
        # Case-sensitive: "TRUE" is not "true" -> invalid
        assert parse_addressed_user_prompt("TRUE") == "invalid"

    def test_string_uppercase_false(self):
        # Case-sensitive: "FALSE" is not "false" -> invalid
        assert parse_addressed_user_prompt("FALSE") == "invalid"

    def test_string_malformed_random(self):
        assert parse_addressed_user_prompt("unsure") == "invalid"

    def test_int_other(self):
        assert parse_addressed_user_prompt(2) == "invalid"

    def test_float(self):
        assert parse_addressed_user_prompt(1.0) == "invalid"

    def test_dict_full_path(self):
        data = {"details": {"annotations": {"result": {"addressed_user_prompt": True}}}}
        assert parse_addressed_user_prompt(data) == "true"

    def test_dict_missing_path(self):
        data = {"details": {}}
        assert parse_addressed_user_prompt(data) == "missing"


class TestParseAnnotationResult:
    """Tests for parse_annotation_result helper."""

    def test_full_path(self):
        data = {
            "details": {
                "annotations": {
                    "result": {
                        "addressed_user_prompt": True,
                        "all_valid": True,
                    }
                }
            }
        }
        result = parse_annotation_result(data)
        assert result.addressed_user_prompt == "true"
        assert result.all_valid == True
        assert result.cluelessness == False
        assert result.completely_hallucinated == False

    def test_false_value(self):
        data = {
            "details": {
                "annotations": {
                    "result": {
                        "addressed_user_prompt": False,
                    }
                }
            }
        }
        result = parse_annotation_result(data)
        assert result.addressed_user_prompt == "false"

    def test_missing_annotations(self):
        data = {
            "details": {
                "result": {
                    "addressed_user_prompt": True,
                }
            }
        }
        result = parse_annotation_result(data)
        assert result.addressed_user_prompt == "missing"

    def test_missing_result(self):
        data = {
            "details": {
                "annotations": {}
            }
        }
        result = parse_annotation_result(data)
        assert result.addressed_user_prompt == "missing"

    def test_missing_details(self):
        data = {}
        result = parse_annotation_result(data)
        assert result.addressed_user_prompt == "missing"

    def test_null_addressed_user_prompt(self):
        # null value in JSON -> "missing"
        data = {
            "details": {
                "annotations": {
                    "result": {
                        "addressed_user_prompt": None,
                    }
                }
            }
        }
        result = parse_annotation_result(data)
        assert result.addressed_user_prompt == "missing"

    def test_string_true_value(self):
        data = {
            "details": {
                "annotations": {
                    "result": {
                        "addressed_user_prompt": "true",
                    }
                }
            }
        }
        result = parse_annotation_result(data)
        assert result.addressed_user_prompt == "true"

    def test_string_false_value(self):
        data = {
            "details": {
                "annotations": {
                    "result": {
                        "addressed_user_prompt": "false",
                    }
                }
            }
        }
        result = parse_annotation_result(data)
        assert result.addressed_user_prompt == "false"

    def test_malformed_value(self):
        data = {
            "details": {
                "annotations": {
                    "result": {
                        "addressed_user_prompt": "unknown",
                    }
                }
            }
        }
        result = parse_annotation_result(data)
        assert result.addressed_user_prompt == "invalid"

    def test_all_valid_true(self):
        data = {
            "details": {
                "annotations": {
                    "result": {
                        "all_valid": True,
                    }
                }
            }
        }
        result = parse_annotation_result(data)
        assert result.all_valid == True

    def test_cluelessness_true(self):
        data = {
            "details": {
                "annotations": {
                    "result": {
                        "cluelessness": True,
                    }
                }
            }
        }
        result = parse_annotation_result(data)
        assert result.cluelessness == True

    def test_completely_hallucinated_true(self):
        data = {
            "details": {
                "annotations": {
                    "result": {
                        "completely_hallucinated": True,
                    }
                }
            }
        }
        result = parse_annotation_result(data)
        assert result.completely_hallucinated == True
