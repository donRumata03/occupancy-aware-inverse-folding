from __future__ import annotations

import argparse
import sys

from _bootstrap import add_experiment_to_path

add_experiment_to_path()

from src.assignment import assign_forward_sample, save_assignment_scores
from src.config import ensure_output_dirs, output_path
from src.config import load_config
from src.pairs import load_pairs
from src.utils import read_csv, save_run_metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    out_dir = ensure_output_dirs(cfg)
    save_run_metadata(cfg, out_dir, sys.argv)
    output = output_path(cfg, "assignment_scores.csv")
    if output.exists() and not args.overwrite:
        print(f"Skipping assignment; existing file found: {output}")
        return 0
    pairs = {pair.pair_id: pair for pair in load_pairs(cfg["conformer_pairs_csv"], create_if_missing=False)}
    rows = []
    for sample_row in read_csv(output_path(cfg, "forward_samples.csv")):
        pair = pairs.get(sample_row["pair_id"])
        if pair is None:
            raise KeyError(f"Unknown pair_id in forward samples: {sample_row['pair_id']}")
        assigned = assign_forward_sample(sample_row, pair, cfg)
        if assigned is not None:
            rows.append(assigned)
    save_assignment_scores(output, rows)
    print(f"Wrote {len(rows)} assignment rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

