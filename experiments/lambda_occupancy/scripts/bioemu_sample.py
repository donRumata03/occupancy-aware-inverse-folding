from __future__ import annotations

import argparse
import inspect
import random
from pathlib import Path
from typing import Any

import numpy as np


def _str_to_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def _call_bioemu(args: argparse.Namespace) -> None:
    from bioemu.sample import main as sample

    kwargs: dict[str, Any] = {
        "sequence": args.sequence,
        "num_samples": args.num_samples,
        "output_dir": str(args.output_dir),
        "filter_samples": args.filter_samples,
    }
    optional = {
        "batch_size_100": args.batch_size_100,
        "model_name": args.model_name,
        "msa_host_url": args.msa_host_url,
        "denoiser_config": args.denoiser_config,
    }
    kwargs.update({key: value for key, value in optional.items() if value is not None})

    signature = inspect.signature(sample)
    supported = {name for name, parameter in signature.parameters.items() if parameter.kind in {
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    }}
    sample(**{key: value for key, value in kwargs.items() if key in supported})


def _convert_xtc_to_pdb(output_dir: Path) -> int:
    import mdtraj as md

    xtc_path = output_dir / "samples.xtc"
    topology_path = output_dir / "topology.pdb"
    if not xtc_path.exists() or not topology_path.exists():
        return 0

    frame_dir = output_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    trajectory = md.load_xtc(str(xtc_path), top=str(topology_path))
    for idx in range(trajectory.n_frames):
        trajectory[idx].save_pdb(str(frame_dir / f"sample_{idx:04d}.pdb"))
    return int(trajectory.n_frames)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--filter-samples", type=_str_to_bool, default=False)
    parser.add_argument("--convert-xtc-to-pdb", action="store_true")
    parser.add_argument("--batch-size-100", type=int)
    parser.add_argument("--model-name")
    parser.add_argument("--msa-host-url")
    parser.add_argument("--denoiser-config")
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        parser.error(f"unknown BioEmu wrapper arguments: {' '.join(unknown)}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _call_bioemu(args)
    if args.convert_xtc_to_pdb:
        n_frames = _convert_xtc_to_pdb(args.output_dir)
        print(f"BioEmu frame export: {n_frames} PDB frame(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
