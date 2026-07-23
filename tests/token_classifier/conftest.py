"""Pytest configuration for token_classifier tests."""

import sys
from pathlib import Path

# Add src to sys.path using ABSOLUTE path
SRC_PATH = "/home/ma-user/work/assessing-reliability-of-responses-in-rag/src"

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
elif sys.path[0] != SRC_PATH:
    sys.path.remove(SRC_PATH)
    sys.path.insert(0, SRC_PATH)
