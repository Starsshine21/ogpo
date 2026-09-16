"""Pytest path setup for source and command-line script modules."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
for relative in ("src", "scripts"):
    path = str(ROOT / relative)
    if path not in sys.path:
        sys.path.insert(0, path)
