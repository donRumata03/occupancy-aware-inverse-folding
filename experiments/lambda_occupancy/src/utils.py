from __future__ import annotations

import csv
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml


AA_ALPHABET = set("ACDEFGHIKLMNPQRSTVWY")


def read_csv(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: str | Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def append_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def copy_config(config: dict[str, Any], out_dir: Path) -> None:
    serializable = {k: v for k, v in config.items() if not k.startswith("_")}
    with (out_dir / "config.resolved.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(serializable, handle, sort_keys=False)


def git_commit() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return proc.stdout.strip()
    except Exception:
        return None


def torch_metadata() -> dict[str, Any]:
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
        return {
            "torch_version": torch.__version__,
            "cuda_available": cuda_available,
            "gpu_name": gpu_name,
        }
    except Exception as exc:
        return {"torch_error": str(exc)}


def run_metadata(config: dict[str, Any], argv: list[str]) -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "working_directory": os.getcwd(),
        "git_commit": git_commit(),
        "command_line": argv,
        "config_path": config.get("_config_path"),
        **torch_metadata(),
    }


def save_run_metadata(config: dict[str, Any], out_dir: Path, argv: list[str]) -> None:
    copy_config(config, out_dir)
    write_json(out_dir / "run_metadata.json", run_metadata(config, argv))


def validate_sequence(sequence: str) -> None:
    bad = sorted(set(sequence.upper()) - AA_ALPHABET)
    if bad:
        raise ValueError(f"Invalid amino-acid letters in sequence: {''.join(bad)}")


def str_to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}

