"""Pytest configuration for token_classifier tests."""

import sys
from pathlib import Path

# Add src to sys.path
SRC_PATH = Path(__file__).parent.parent / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
elif sys.path[0] != str(SRC_PATH):
    sys.path.remove(str(SRC_PATH))
    sys.path.insert(0, str(SRC_PATH))
