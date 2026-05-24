from __future__ import annotations

import argparse
import sys

from _bootstrap import add_experiment_to_path

add_experiment_to_path()

from src.bioemu_runner import run_forward_samples, save_forward_samples
from src.config import ensure_output_dirs, output_path
from src.config import load_config
from src.inverse_adapters import load_generated_sequences_csv
from src.utils import save_run_metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    out_dir = ensure_output_dirs(cfg)
    save_run_metadata(cfg, out_dir, sys.argv)
    output = output_path(cfg, "forward_samples.csv")
    if output.exists() and not args.overwrite:
        print(f"Skipping BioEmu; existing file found: {output}")
        return 0
    sequences_path = output_path(cfg, "generated_sequences.csv")
    records = []
    for sequence_record in load_generated_sequences_csv(sequences_path):
        records.extend(run_forward_samples(sequence_record, int(cfg["n_forward_samples"]), cfg))
    save_forward_samples(output, records)
    n_valid = sum(1 for record in records if record.valid_sample)
    print(f"Wrote {len(records)} forward sample rows ({n_valid} valid) to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

