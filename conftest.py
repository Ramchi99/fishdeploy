"""Puts the repository root on sys.path so tests can import deploy.*, hardware.* and analysis.*."""
import sys
from pathlib import Path

_root = str(Path(__file__).resolve().parent)
if _root not in sys.path:
    sys.path.insert(0, _root)
