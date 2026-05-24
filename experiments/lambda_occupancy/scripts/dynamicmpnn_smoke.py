from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=None, help="Python executable with DynamicMPNN dependencies installed.")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--output-dir", default="outputs/lambda_occupancy/dynamicmpnn_smoke")
    args = parser.parse_args(argv)

    python_executable = args.python or sys.executable
    output_dir = Path(args.output_dir)
    wrapper = Path("experiments/lambda_occupancy/scripts/dynamicmpnn_seeded_eval.py")
    command = [
        str(python_executable),
        str(wrapper),
        "--seed",
        "1",
        "--",
        "eval=1bdt_1qtg",
        "eval.model_ref=external/DynamicMPNN/checkpoints/single_chain_k2.ckpt",
        "eval.num_samples=1",
        "eval.af3_evaluate=false",
        f"eval.device={args.device}",
        "eval.sampling_mode=single",
        "eval.refold_mode=single",
        "model.temperature=0.1",
        f"output_dir={str(output_dir).replace(chr(92), '/')}",
        f"hydra.run.dir={str(output_dir).replace(chr(92), '/')}",
    ]
    proc = subprocess.run(command, text=True)
    if proc.returncode != 0:
        return proc.returncode

    samples_csv = output_dir / "samples" / "samples.csv"
    if not samples_csv.exists():
        print(f"ERROR: expected samples CSV was not created: {samples_csv}", file=sys.stderr)
        return 1
    print(f"DynamicMPNN smoke OK: {samples_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
