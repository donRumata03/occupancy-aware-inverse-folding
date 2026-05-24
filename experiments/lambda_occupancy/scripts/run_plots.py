from __future__ import annotations

import argparse
import sys

from _bootstrap import add_experiment_to_path

add_experiment_to_path()

from src.config import ensure_output_dirs, output_path
from src.config import load_config
from src.plotting import plot_delta_score_distributions, plot_per_sequence_distributions, plot_response_curves
from src.utils import read_csv, save_run_metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    out_dir = ensure_output_dirs(cfg)
    save_run_metadata(cfg, out_dir, sys.argv)
    figures = out_dir / "figures"
    response_path = figures / "response_curves.png"
    dist_path = figures / "per_sequence_distributions.png"
    delta_path = figures / "delta_score_distributions.png"
    if all(path.exists() for path in [response_path, dist_path, delta_path]) and not args.overwrite:
        print(f"Skipping plots; existing figure files found in {figures}")
        return 0
    plot_response_curves(read_csv(output_path(cfg, "response_curves.csv")), response_path)
    plot_per_sequence_distributions(
        read_csv(output_path(cfg, "sequence_occupancies.csv")),
        [float(x) for x in cfg.get("plots", {}).get("selected_lambdas", [0.5, 1.0, 2.0])],
        dist_path,
    )
    plot_delta_score_distributions(read_csv(output_path(cfg, "assignment_scores.csv")), delta_path)
    print(f"Wrote figures to {figures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

