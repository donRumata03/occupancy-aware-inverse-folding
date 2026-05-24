from __future__ import annotations

import math
import os
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import experiment_dir
from .inverse_adapters import GeneratedSequence
from .structure_io import write_single_sequence_a3m
from .utils import write_csv


FORWARD_SAMPLE_FIELDS = [
    "sample_id",
    "sequence_id",
    "pair_id",
    "lambda_value",
    "inverse_model",
    "sample_index",
    "structure_path",
    "forward_model",
    "forward_seed",
    "valid_sample",
    "metadata",
]


@dataclass
class ForwardSample:
    sample_id: str
    sequence_id: str
    pair_id: str
    lambda_value: float
    inverse_model: str
    sample_index: int
    structure_path: str
    forward_model: str
    forward_seed: int | None = None
    valid_sample: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["metadata"] = repr(self.metadata)
        return row


def save_forward_samples(path: str | Path, records: list[ForwardSample]) -> None:
    write_csv(path, [record.to_row() for record in records], FORWARD_SAMPLE_FIELDS)


def _build_bioemu_command(
    sequence_record: GeneratedSequence,
    output_dir: Path,
    requested_samples: int,
    config: dict[str, Any],
    a3m_path: Path | None,
) -> list[str] | str:
    bioemu_cfg = config.get("bioemu", {})
    template = bioemu_cfg.get("command_template")
    if template:
        return template.format(
            sequence=sequence_record.sequence,
            sequence_id=sequence_record.sequence_id,
            a3m_path=a3m_path or "",
            output_dir=output_dir,
            n_samples=requested_samples,
            device=bioemu_cfg.get("device", "cuda"),
            batch_size=bioemu_cfg.get("batch_size", 1),
        )

    executable = bioemu_cfg.get("executable_or_module")
    extra_args = [str(x) for x in bioemu_cfg.get("extra_args", [])]
    if executable:
        if str(executable).startswith("module:"):
            module_name = str(executable).split(":", 1)[1]
            return [
                "python",
                "-m",
                module_name,
                "--sequence",
                str(a3m_path or sequence_record.sequence),
                "--num_samples",
                str(requested_samples),
                "--output_dir",
                str(output_dir),
                "--filter_samples",
                str(bool(bioemu_cfg.get("filter_samples", False))),
                *extra_args,
            ]
        return [
            str(executable),
            "--sequence",
            str(a3m_path or sequence_record.sequence),
            "--num_samples",
            str(requested_samples),
            "--output_dir",
            str(output_dir),
            "--filter_samples",
            str(bool(bioemu_cfg.get("filter_samples", False))),
            *extra_args,
        ]

    python_executable = (
        bioemu_cfg.get("python_executable")
        or os.environ.get("BIOEMU_PYTHON")
        or sys.executable
    )
    wrapper = Path("experiments/lambda_occupancy/scripts/bioemu_sample.py")
    command = [
        str(python_executable),
        str(wrapper),
        "--sequence",
        str(a3m_path or sequence_record.sequence),
        "--num-samples",
        str(requested_samples),
        "--output-dir",
        str(output_dir),
        "--seed",
        str(int(sequence_record.seed) + 1000003),
    ]
    if bioemu_cfg.get("filter_samples") is not None:
        command.extend(["--filter-samples", str(bool(bioemu_cfg.get("filter_samples"))).lower()])
    if bioemu_cfg.get("convert_xtc_to_pdb", True):
        command.append("--convert-xtc-to-pdb")
    if bioemu_cfg.get("batch_size_100") is not None:
        command.extend(["--batch-size-100", str(bioemu_cfg["batch_size_100"])])
    if bioemu_cfg.get("model_name"):
        command.extend(["--model-name", str(bioemu_cfg["model_name"])])
    if bioemu_cfg.get("msa_host_url"):
        command.extend(["--msa-host-url", str(bioemu_cfg["msa_host_url"])])
    if bioemu_cfg.get("denoiser_config"):
        command.extend(["--denoiser-config", str(bioemu_cfg["denoiser_config"])])
    command.extend(extra_args)
    return command


def _find_bioemu_frame_pdbs(output_dir: Path) -> list[Path]:
    frame_dir = output_dir / "frames"
    if frame_dir.exists():
        return sorted(frame_dir.glob("sample_*.pdb"))
    return []


def _find_xtc_or_topology_outputs(output_dir: Path) -> list[Path]:
    paths = []
    for name in ("samples.xtc", "topology.pdb"):
        path = output_dir / name
        if path.exists():
            paths.append(path)
    return paths


def _find_structures(output_dir: Path) -> list[Path]:
    frame_pdbs = _find_bioemu_frame_pdbs(output_dir)
    if frame_pdbs:
        return frame_pdbs

    suffixes = {".pdb", ".cif", ".mmcif"}
    excluded_names = {"topology.pdb"}
    structures = [
        path
        for path in output_dir.rglob("*")
        if path.suffix.lower() in suffixes and path.name not in excluded_names
    ]
    if structures:
        return sorted(structures)
    return _find_xtc_or_topology_outputs(output_dir)


def run_forward_samples(
    sequence_record: GeneratedSequence,
    m_valid: int,
    config: dict[str, Any],
) -> list[ForwardSample]:
    bioemu_cfg = config.get("bioemu", {})
    seq_dir = experiment_dir(config) / "bioemu" / sequence_record.sequence_id
    seq_dir.mkdir(parents=True, exist_ok=True)
    multiplier = float(bioemu_cfg.get("requested_samples_multiplier", 1.0))
    requested_samples = max(m_valid, int(math.ceil(m_valid * multiplier)))
    a3m_path = None
    if bioemu_cfg.get("use_single_sequence_a3m", True):
        a3m_path = write_single_sequence_a3m(sequence_record.sequence_id, sequence_record.sequence, seq_dir)

    forward_seed = int(sequence_record.seed) + 1000003
    try:
        command = _build_bioemu_command(sequence_record, seq_dir, requested_samples, config, a3m_path)
        command_text = command if isinstance(command, str) else " ".join(shlex.quote(part) for part in command)
        proc = subprocess.run(command, shell=isinstance(command, str), text=True, capture_output=True)
        metadata = {
            "command": command_text,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "requested_samples": requested_samples,
            "a3m_path": str(a3m_path) if a3m_path else "",
        }
        if proc.returncode != 0:
            if config.get("strict"):
                raise RuntimeError(f"BioEmu command failed: {command_text}\n{proc.stderr}")
            return [
                ForwardSample(
                    sample_id=f"{sequence_record.sequence_id}_failed",
                    sequence_id=sequence_record.sequence_id,
                    pair_id=sequence_record.pair_id,
                    lambda_value=sequence_record.lambda_value,
                    inverse_model=sequence_record.inverse_model,
                    sample_index=-1,
                    structure_path="",
                    forward_model=config["forward_model"],
                    forward_seed=forward_seed,
                    valid_sample=False,
                    metadata=metadata,
                )
            ]
    except Exception as exc:
        if config.get("strict"):
            raise
        metadata = {
            "error": str(exc),
            "requested_samples": requested_samples,
            "a3m_path": str(a3m_path) if a3m_path else "",
        }
        return [
            ForwardSample(
                sample_id=f"{sequence_record.sequence_id}_failed",
                sequence_id=sequence_record.sequence_id,
                pair_id=sequence_record.pair_id,
                lambda_value=sequence_record.lambda_value,
                inverse_model=sequence_record.inverse_model,
                sample_index=-1,
                structure_path="",
                forward_model=config["forward_model"],
                forward_seed=forward_seed,
                valid_sample=False,
                metadata=metadata,
            )
        ]

    structures = _find_structures(seq_dir)[:m_valid]
    records: list[ForwardSample] = []
    for idx, structure_path in enumerate(structures):
        records.append(
            ForwardSample(
                sample_id=f"{sequence_record.sequence_id}_m{idx:04d}",
                sequence_id=sequence_record.sequence_id,
                pair_id=sequence_record.pair_id,
                lambda_value=sequence_record.lambda_value,
                inverse_model=sequence_record.inverse_model,
                sample_index=idx,
                structure_path=str(structure_path),
                forward_model=config["forward_model"],
                forward_seed=forward_seed,
                valid_sample=True,
                metadata=metadata,
            )
        )
    if len(records) < m_valid:
        records.append(
            ForwardSample(
                sample_id=f"{sequence_record.sequence_id}_incomplete",
                sequence_id=sequence_record.sequence_id,
                pair_id=sequence_record.pair_id,
                lambda_value=sequence_record.lambda_value,
                inverse_model=sequence_record.inverse_model,
                sample_index=-1,
                structure_path="",
                forward_model=config["forward_model"],
                forward_seed=forward_seed,
                valid_sample=False,
                metadata={**metadata, "warning": f"only {len(records)} valid samples found"},
            )
        )
    return records
