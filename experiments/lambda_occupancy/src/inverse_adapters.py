from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .pairs import ConformerPair
from .utils import read_csv, validate_sequence, write_csv


GENERATED_SEQUENCE_FIELDS = [
    "sequence_id",
    "pair_id",
    "lambda_value",
    "inverse_model",
    "seed",
    "sequence",
    "temperature",
    "metadata",
]


@dataclass
class GeneratedSequence:
    sequence_id: str
    pair_id: str
    lambda_value: float
    inverse_model: str
    seed: int
    sequence: str
    temperature: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["metadata"] = repr(self.metadata)
        return row


def lambda_to_state_weights(lambda_value: float) -> tuple[float, float]:
    if lambda_value < 0:
        raise ValueError("lambda_value must be non-negative")
    alpha = lambda_value / (1.0 + lambda_value)
    return 1.0 - alpha, alpha


def load_generated_sequences_csv(path: str | Path) -> list[GeneratedSequence]:
    records: list[GeneratedSequence] = []
    for row in read_csv(path):
        sequence = row["sequence"].strip().upper()
        validate_sequence(sequence)
        records.append(
            GeneratedSequence(
                sequence_id=row["sequence_id"],
                pair_id=row["pair_id"],
                lambda_value=float(row["lambda_value"]),
                inverse_model=row.get("inverse_model", "manual_csv"),
                seed=int(float(row.get("seed") or 0)),
                sequence=sequence,
                temperature=float(row.get("temperature") or 0.0),
                metadata={"source_csv": str(path)},
            )
        )
    return records


def save_generated_sequences(path: str | Path, records: list[GeneratedSequence]) -> None:
    write_csv(path, [record.to_row() for record in records], GENERATED_SEQUENCE_FIELDS)


class DynamicMPNNAdapter:
    def generate_sequences(
        self,
        pair: ConformerPair,
        lambda_value: float,
        n_sequences: int,
        seed: int,
        config: dict[str, Any],
    ) -> list[GeneratedSequence]:
        command_template = config.get("inverse", {}).get("command_template")
        w0, w1 = lambda_to_state_weights(lambda_value)
        output_csv = Path(config["_stage_dir"]) / f"dynamic_{pair.pair_id}_lambda_{lambda_value:g}_seed_{seed}.csv"
        if not command_template:
            return self._generate_with_dynamicmpnn(
                pair=pair,
                lambda_value=lambda_value,
                weights=(w0, w1),
                n_sequences=n_sequences,
                seed=seed,
                output_csv=output_csv,
                config=config,
            )

        command = command_template.format(
            x0_pdb=pair.x0_pdb,
            x1_pdb=pair.x1_pdb,
            pair_id=pair.pair_id,
            lambda_value=lambda_value,
            weight0=w0,
            weight1=w1,
            n_sequences=n_sequences,
            seed=seed,
            temperature=config["sampling_temperature"],
            output_csv=output_csv,
        )
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(command, shell=True, text=True, capture_output=True)
        command_log = {
            "command": command,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "effective_weight0": w0,
            "effective_weight1": w1,
        }
        if proc.returncode != 0:
            raise RuntimeError(f"DynamicMPNN command failed: {command}\n{proc.stderr}")
        records = load_generated_sequences_csv(output_csv)
        for record in records:
            record.metadata.update(command_log)
        return records

    def _generate_with_dynamicmpnn(
        self,
        pair: ConformerPair,
        lambda_value: float,
        weights: tuple[float, float],
        n_sequences: int,
        seed: int,
        output_csv: Path,
        config: dict[str, Any],
    ) -> list[GeneratedSequence]:
        inverse_cfg = config.get("inverse", {})
        dynamic_cfg = inverse_cfg.get("dynamicmpnn", {})

        repo_path = Path(dynamic_cfg.get("repo_path") or "external/DynamicMPNN")
        model_ref = Path(dynamic_cfg.get("model_ref") or repo_path / "checkpoints" / "single_chain_k2.ckpt")
        if not model_ref.exists():
            raise FileNotFoundError(
                f"DynamicMPNN checkpoint not found: {model_ref}. "
                "Initialize submodules or set inverse.dynamicmpnn.model_ref."
            )

        chain_id = dynamic_cfg.get("chain_id") or pair.chain_id
        if not chain_id:
            raise ValueError(f"DynamicMPNN requires a chain_id for pair {pair.pair_id}")

        run_dir = output_csv.with_suffix("")
        samples_csv = run_dir / "samples" / "samples.csv"
        wrapper = Path("experiments/lambda_occupancy/scripts/dynamicmpnn_seeded_eval.py")
        python_executable = (
            dynamic_cfg.get("python_executable")
            or os.environ.get("DYNAMICMPNN_PYTHON")
            or sys.executable
        )

        hydra_overrides = [
            f"eval.model_ref={_hydra_path(model_ref)}",
            f"eval.num_samples={int(n_sequences)}",
            f"eval.device={dynamic_cfg.get('device', 'auto')}",
            f"eval.sampling_mode={dynamic_cfg.get('sampling_mode', 'single')}",
            f"eval.refold_mode={dynamic_cfg.get('refold_mode', 'single')}",
            f"eval.af3_evaluate={str(bool(dynamic_cfg.get('af3_evaluate', False))).lower()}",
            f"eval.targets.state1.pdb_path={_hydra_path(pair.x0_pdb)}",
            f"eval.targets.state1.chain_id={chain_id}",
            f"eval.targets.state2.pdb_path={_hydra_path(pair.x1_pdb)}",
            f"eval.targets.state2.chain_id={chain_id}",
            f"model.temperature={float(config['sampling_temperature'])}",
            f"output_dir={_hydra_path(run_dir)}",
            f"hydra.run.dir={_hydra_path(run_dir)}",
        ]
        if dynamic_cfg.get("alignment_state0"):
            hydra_overrides.append(f"eval.alignment.state1={_hydra_quoted(dynamic_cfg['alignment_state0'])}")
        if dynamic_cfg.get("alignment_state1"):
            hydra_overrides.append(f"eval.alignment.state2={_hydra_quoted(dynamic_cfg['alignment_state1'])}")
        hydra_overrides.extend(str(item) for item in dynamic_cfg.get("extra_overrides", []))
        hydra_overrides.extend(str(item) for item in inverse_cfg.get("extra_args", []))

        command = [
            str(python_executable),
            str(wrapper),
            "--seed",
            str(seed),
            "--",
            *hydra_overrides,
        ]
        env = os.environ.copy()
        repo_src = str((repo_path / "src").resolve())
        env["PYTHONPATH"] = repo_src + os.pathsep + env.get("PYTHONPATH", "")

        output_csv.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(command, text=True, capture_output=True, env=env)
        command_log = {
            "command": command,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "effective_weight0": weights[0],
            "effective_weight1": weights[1],
            "dynamicmpnn_samples_csv": str(samples_csv),
            "lambda_controls_sampling": bool(dynamic_cfg.get("lambda_controls_sampling", False)),
        }
        if proc.returncode != 0:
            raise RuntimeError(f"DynamicMPNN command failed: {' '.join(command)}\n{proc.stderr}")
        if not samples_csv.exists():
            raise FileNotFoundError(f"DynamicMPNN did not write expected samples CSV: {samples_csv}")

        rows = read_csv(samples_csv)
        records: list[GeneratedSequence] = []
        for idx, row in enumerate(rows):
            sequence = row["sequence"].strip().upper()
            validate_sequence(sequence)
            sequence_id = f"{pair.pair_id}_lambda_{lambda_value:g}_seed_{seed}_{row.get('sequence_id') or idx}"
            records.append(
                GeneratedSequence(
                    sequence_id=sequence_id,
                    pair_id=pair.pair_id,
                    lambda_value=lambda_value,
                    inverse_model="dynamic_mpnn",
                    seed=seed,
                    sequence=sequence,
                    temperature=float(config["sampling_temperature"]),
                    metadata={
                        **command_log,
                        "native_sequence_id": row.get("sequence_id", ""),
                        "native_length": row.get("length", ""),
                        "lambda_note": (
                            "The public DynamicMPNN sampler does not expose state-weighted lambda control; "
                            "weights are recorded for experiment bookkeeping."
                        ),
                    },
                )
            )
        save_generated_sequences(output_csv, records)
        return records


class PlaceholderAdapter:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def generate_sequences(self, *_: Any, **__: Any) -> list[GeneratedSequence]:
        raise NotImplementedError(
            f"{self.model_name} was not found in this repository. Add a command template "
            "or implement the adapter after the DynamicMPNN pipeline is validated."
        )


def adapter_for(model_name: str) -> DynamicMPNNAdapter | PlaceholderAdapter:
    normalized = model_name.lower()
    if normalized == "dynamic_mpnn":
        return DynamicMPNNAdapter()
    if normalized in {"proto_mpnn", "protein_mpnn_msd"}:
        return PlaceholderAdapter(model_name)
    raise ValueError(f"Unknown inverse model: {model_name}")


def _hydra_path(path: str | Path) -> str:
    return str(Path(path)).replace("\\", "/")


def _hydra_quoted(value: Any) -> str:
    text = str(value).replace("\\", "\\\\").replace("'", "\\'")
    return f"'{text}'"
