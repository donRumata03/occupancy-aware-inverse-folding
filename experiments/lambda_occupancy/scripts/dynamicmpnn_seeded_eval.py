from __future__ import annotations

import argparse
import random
import sys

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

    from dynamicmpnn.evaluate import main as dynamicmpnn_main

    sys.argv = [sys.argv[0], *overrides]
    dynamicmpnn_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
