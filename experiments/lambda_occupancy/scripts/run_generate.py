from __future__ import annotations

import argparse
import sys

from _bootstrap import add_experiment_to_path

add_experiment_to_path()

from src.config import ensure_output_dirs, output_path
from src.config import load_config
from src.inverse_adapters import adapter_for, load_generated_sequences_csv, save_generated_sequences
from src.pairs import load_pairs, validate_pairs
from src.utils import save_run_metadata, write_csv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    out_dir = ensure_output_dirs(cfg)
    save_run_metadata(cfg, out_dir, sys.argv)
    generated_path = output_path(cfg, "generated_sequences.csv")
    if generated_path.exists() and not args.overwrite:
        print(f"Skipping generation; existing file found: {generated_path}")
        return 0

    manual_csv = cfg.get("manual_sequences_csv")
    if manual_csv:
        records = load_generated_sequences_csv(manual_csv)
        save_generated_sequences(generated_path, records)
        print(f"Loaded {len(records)} manual sequences into {generated_path}")
        return 0

    pairs = load_pairs(cfg["conformer_pairs_csv"], limit=int(cfg.get("n_pairs", 0)) or None)
    write_csv(out_dir / "pair_metadata.csv", validate_pairs(pairs), [
        "pair_id", "x0_pdb", "x1_pdb", "chain_id", "x0_chain_id", "x1_chain_id",
        "alignment_state0", "alignment_state1", "length_hint", "notes", "preprocessing_warnings"
    ])
    adapter = adapter_for(cfg["inverse_model"])
    cfg["_stage_dir"] = str(out_dir / "raw" / "inverse")
    records = []
    for pair_idx, pair in enumerate(pairs):
        for lambda_value in cfg["lambdas"]:
            seed = int(cfg["random_seed"]) + pair_idx * 10000
            try:
                records.extend(
                    adapter.generate_sequences(
                        pair,
                        float(lambda_value),
                        int(cfg["n_sequences_per_lambda"]),
                        seed,
                        cfg,
                    )
                )
            except NotImplementedError as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                return 2
    save_generated_sequences(generated_path, records)
    print(f"Wrote {len(records)} generated sequences to {generated_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
