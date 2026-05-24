from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=None, help="Python executable with BioEmu installed.")
    parser.add_argument("--output-dir", default="outputs/lambda_occupancy/bioemu_smoke")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--sequence", default="GYDPETGTWG")
    parser.add_argument("--no-convert", action="store_true")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    command = [
        args.python or sys.executable,
        "experiments/lambda_occupancy/scripts/bioemu_sample.py",
        "--sequence",
        args.sequence,
        "--num-samples",
        str(args.num_samples),
        "--output-dir",
        str(output_dir),
        "--seed",
        "1",
        "--filter-samples",
        "false",
    ]
    if not args.no_convert:
        command.append("--convert-xtc-to-pdb")

    proc = subprocess.run(command, text=True)
    if proc.returncode != 0:
        return proc.returncode

    required = [output_dir / "samples.xtc", output_dir / "topology.pdb"]
    missing = [path for path in required if not path.exists()]
    if missing:
        print("ERROR: BioEmu did not create expected outputs: " + ", ".join(str(path) for path in missing), file=sys.stderr)
        return 1
    if not args.no_convert and not list((output_dir / "frames").glob("sample_*.pdb")):
        print(f"ERROR: BioEmu frame export did not create PDB files under {output_dir / 'frames'}", file=sys.stderr)
        return 1
    print(f"BioEmu smoke OK: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
