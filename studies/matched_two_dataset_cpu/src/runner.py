"""Public CLI for both datasets; delegate to the byte-preserved study engine."""
from pathlib import Path
import sys

FROZEN = Path(__file__).resolve().parents[1] / 'provenance' / 'frozen_pipeline'
sys.path.insert(0, str(FROZEN))
from run import main

if __name__ == '__main__':
    main()
