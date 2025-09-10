# src/tests/conftest.py
import sys
from pathlib import Path

# Add the parent directory of this tests folder (i.e., src/) to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
