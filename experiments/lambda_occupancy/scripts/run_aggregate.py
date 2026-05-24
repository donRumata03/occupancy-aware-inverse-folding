from __future__ import annotations

import argparse
import sys

from _bootstrap import add_experiment_to_path

add_experiment_to_path()

from src.aggregate import (
    aggregate_response_curves,
    aggregate_sequence_occupancies,
    save_response_curves,
    save_sequence_occupancies,
)
from src.config import ensure_output_dirs, output_path
from src.config import load_config
from src.utils import read_csv, save_run_metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    out_dir = ensure_output_dirs(cfg)
    save_run_metadata(cfg, out_dir, sys.argv)
    seq_output = output_path(cfg, "sequence_occupancies.csv")
    curve_output = output_path(cfg, "response_curves.csv")
    if seq_output.exists() and curve_output.exists() and not args.overwrite:
        print(f"Skipping aggregation; existing files found: {seq_output}, {curve_output}")
        return 0
    assignment_rows = read_csv(output_path(cfg, "assignment_scores.csv"))
    sequence_rows = aggregate_sequence_occupancies(assignment_rows, int(cfg["n_forward_samples"]))
    curve_rows = aggregate_response_curves(sequence_rows, cfg)
    save_sequence_occupancies(seq_output, sequence_rows)
    save_response_curves(curve_output, curve_rows)
    print(f"Wrote {len(sequence_rows)} sequence rows and {len(curve_rows)} response rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

