"""Shared pytest configuration for the project test suite."""

from pathlib import Path
import sys

import matplotlib

# Allow `from src...` imports when pytest is run directly from a checkout,
# without requiring `pip install -e .` first.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Keep plotting tests headless and CI-friendly.
matplotlib.use("Agg")
