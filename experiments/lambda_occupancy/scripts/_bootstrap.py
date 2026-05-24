from __future__ import annotations

import sys
from pathlib import Path


def add_experiment_to_path() -> Path:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    return root

