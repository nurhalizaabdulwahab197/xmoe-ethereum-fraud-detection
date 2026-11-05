"""Run the original study's unchanged synthetic regression checks."""
from pathlib import Path
import sys

FROZEN = Path(__file__).resolve().parents[1] / 'provenance' / 'frozen_pipeline'
sys.path.insert(0, str(FROZEN))

def load_tests(loader, tests, pattern):
    return loader.discover(str(FROZEN), pattern='test_*.py', top_level_dir=str(FROZEN))
