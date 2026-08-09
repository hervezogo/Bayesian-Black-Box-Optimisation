"""Project-wide paths and stable configuration."""

from pathlib import Path


### Global paths

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
INITIAL_DATA_DIR = DATA_DIR / "initial"
FINAL_DATA_DIR = DATA_DIR / "final"
FIGURES_DIR = ROOT_DIR / "figures"

### Global files

QUERIES_FILE = FINAL_DATA_DIR / "queries.csv"
RESULTS_FILE = FINAL_DATA_DIR / "results.csv"

### Global random seeds

RANDOM_SEED = 42


