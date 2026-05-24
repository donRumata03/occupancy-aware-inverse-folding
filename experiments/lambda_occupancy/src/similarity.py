from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


class AlignmentToolMissing(RuntimeError):
    pass


def find_alignment_tool(preferred: str | None = None) -> str:
    candidates = [preferred] if preferred else []
    candidates.extend(["USalign", "US-align", "TMalign", "TM-align"])
    for candidate in candidates:
        if candidate and shutil.which(candidate):
            return candidate
    raise AlignmentToolMissing(
        "No US-align/TM-align executable found on PATH. Install US-align or TM-align, "
        "or set assignment.alignment_tool in the YAML config."
    )


def parse_tm_score(output: str) -> float:
    scores = [float(x) for x in re.findall(r"TM-score\s*=\s*([0-9]*\.?[0-9]+)", output)]
    if not scores:
        raise ValueError("Could not parse TM-score from alignment output")
    return max(scores)


def tm_score(mobile: str | Path, target: str | Path, config: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    assignment_cfg = config.get("assignment", {})
    try:
        tool = find_alignment_tool(assignment_cfg.get("alignment_tool"))
    except AlignmentToolMissing:
        if assignment_cfg.get("allow_python_fallback"):
            return ca_tm_score_fallback(mobile, target)
        raise
    command = [tool, str(mobile), str(target)]
    proc = subprocess.run(command, check=False, capture_output=True, text=True)
    metadata = {
        "command": " ".join(command),
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
    if proc.returncode != 0:
        raise RuntimeError(f"Alignment command failed: {' '.join(command)}\n{proc.stderr}")
    return parse_tm_score(proc.stdout + "\n" + proc.stderr), metadata


def ca_tm_score_fallback(mobile: str | Path, target: str | Path) -> tuple[float, dict[str, Any]]:
    mobile_coords = _read_ca_coords(mobile)
    target_coords = _read_ca_coords(target)
    n = min(len(mobile_coords), len(target_coords))
    if n < 3:
        raise ValueError(f"Need at least 3 CA atoms for fallback alignment: {mobile}, {target}")
    mobile_aligned = np.asarray([coord for _, coord in mobile_coords[:n]], dtype=float)
    target_aligned = np.asarray([coord for _, coord in target_coords[:n]], dtype=float)
    fitted = _kabsch_fit(mobile_aligned, target_aligned)
    distances = np.linalg.norm(fitted - target_aligned, axis=1)
    d0 = _tm_d0(len(target_coords))
    score = float(np.sum(1.0 / (1.0 + (distances / d0) ** 2)) / len(target_coords))
    metadata = {
        "method": "python_ca_tm_score_fallback",
        "mobile": str(mobile),
        "target": str(target),
        "aligned_ca_count": n,
        "target_ca_count": len(target_coords),
        "d0": d0,
        "note": "Approximate CA-order TM-like score used because no TM-align/US-align executable was available.",
    }
    return score, metadata


def _read_ca_coords(path: str | Path) -> list[tuple[tuple[str, str, str], np.ndarray]]:
    coords: list[tuple[tuple[str, str, str], np.ndarray]] = []
    with Path(path).open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip() != "CA":
                continue
            key = (line[21].strip(), line[22:26].strip(), line[26].strip())
            coord = np.asarray(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                dtype=float,
            )
            coords.append((key, coord))
    if not coords:
        raise ValueError(f"No CA atoms found in PDB: {path}")
    return coords


def _kabsch_fit(mobile: np.ndarray, target: np.ndarray) -> np.ndarray:
    mobile_center = mobile.mean(axis=0)
    target_center = target.mean(axis=0)
    mobile_zero = mobile - mobile_center
    target_zero = target - target_center
    covariance = mobile_zero.T @ target_zero
    u_mat, _, vt_mat = np.linalg.svd(covariance)
    sign = np.sign(np.linalg.det(vt_mat.T @ u_mat.T))
    correction = np.diag([1.0, 1.0, sign])
    rotation = vt_mat.T @ correction @ u_mat.T
    return mobile_zero @ rotation.T + target_center


def _tm_d0(target_length: int) -> float:
    if target_length <= 21:
        return 0.5
    return 1.24 * ((target_length - 15) ** (1.0 / 3.0)) - 1.8
