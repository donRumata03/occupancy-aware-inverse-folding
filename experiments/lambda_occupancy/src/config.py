from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "experiment_name": "lambda_occupancy",
    "output_dir": "outputs/lambda_occupancy",
    "conformer_pairs_csv": "experiments/lambda_occupancy/conformer_pairs_template.csv",
    "inverse_model": "dynamic_mpnn",
    "forward_model": "bioemu",
    "lambdas": [0.5, 1.0, 2.0],
    "n_sequences_per_lambda": 3,
    "n_forward_samples": 10,
    "sampling_temperature": 0.1,
    "random_seed": 0,
    "strict": False,
    "manual_sequences_csv": None,
    "inverse": {
        "command_template": None,
        "extra_args": [],
        "dynamicmpnn": {
            "repo_path": "external/DynamicMPNN",
            "python_executable": None,
            "model_ref": "external/DynamicMPNN/checkpoints/single_chain_k2.ckpt",
            "device": "auto",
            "sampling_mode": "single",
            "refold_mode": "single",
            "af3_evaluate": False,
            "chain_id": None,
            "alignment_state0": None,
            "alignment_state1": None,
            "extra_overrides": [],
            "lambda_controls_sampling": False,
        },
    },
    "bioemu": {
        "executable_or_module": None,
        "python_executable": None,
        "device": "cuda",
        "batch_size": 1,
        "batch_size_100": None,
        "model_name": None,
        "filter_samples": False,
        "convert_xtc_to_pdb": True,
        "msa_host_url": None,
        "denoiser_config": None,
        "extra_args": [],
        "use_single_sequence_a3m": True,
        "requested_samples_multiplier": 1.5,
        "command_template": None,
    },
    "assignment": {
        "metric": "tm_score",
        "use_hard_assignment": True,
        "margin_delta": 0.0,
        "allow_unknown": False,
        "alignment_tool": None,
        "allow_python_fallback": False,
    },
    "bootstrap": {
        "n_bootstrap": 10000,
        "ci": 0.95,
    },
    "plots": {
        "selected_lambdas": [0.5, 1.0, 2.0],
    },
}


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        user_config = yaml.safe_load(handle) or {}
    cfg = _deep_update(DEFAULT_CONFIG, user_config)
    cfg["_config_path"] = str(path)
    return cfg


def experiment_dir(config: dict[str, Any]) -> Path:
    return Path(config["output_dir"]) / str(config["experiment_name"])


def ensure_output_dirs(config: dict[str, Any]) -> Path:
    out_dir = experiment_dir(config)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)
    (out_dir / "bioemu").mkdir(parents=True, exist_ok=True)
    return out_dir


def output_path(config: dict[str, Any], filename: str) -> Path:
    return experiment_dir(config) / filename
