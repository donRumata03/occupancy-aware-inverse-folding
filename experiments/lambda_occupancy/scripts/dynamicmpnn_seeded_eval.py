from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    overrides = args.overrides
    if overrides and overrides[0] == "--":
        overrides = overrides[1:]

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    experiment_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(experiment_root))
    from src.dynamicmpnn_weighted_pooling import patch_dynamicmpnn_weighted_pooling

    patch_dynamicmpnn_weighted_pooling()

    from dynamicmpnn.evaluate import main as dynamicmpnn_main

    sys.argv = [sys.argv[0], *overrides]
    dynamicmpnn_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
