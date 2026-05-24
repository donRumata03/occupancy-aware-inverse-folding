from __future__ import annotations

from pathlib import Path
from typing import Any

from .pairs import ConformerPair
from .similarity import tm_score
from .utils import str_to_bool, write_csv


ASSIGNMENT_FIELDS = [
    "sample_id",
    "sequence_id",
    "pair_id",
    "lambda_value",
    "inverse_model",
    "structure_path",
    "A_m0",
    "A_m1",
    "delta_A",
    "assigned_state",
    "valid_sample",
    "assignment_metric",
    "metadata",
]


def hard_assign(a_m0: float, a_m1: float, margin_delta: float = 0.0, allow_unknown: bool = False) -> str:
    delta = a_m1 - a_m0
    if allow_unknown and abs(delta) <= margin_delta:
        return "unknown"
    return "1" if delta > 0.0 else "0"


def assign_forward_sample(row: dict[str, Any], pair: ConformerPair, config: dict[str, Any]) -> dict[str, Any] | None:
    if not str_to_bool(row.get("valid_sample", False)):
        return None
    structure_path = row.get("structure_path", "")
    if not structure_path or not Path(structure_path).exists():
        if config.get("strict"):
            raise FileNotFoundError(f"Missing forward sample structure: {structure_path}")
        return None

    metric = config.get("assignment", {}).get("metric", "tm_score")
    if metric != "tm_score":
        raise ValueError(f"Unsupported assignment metric: {metric}")
    a_m0, meta0 = tm_score(structure_path, pair.x0_pdb, config, target_chain_id=pair.x0_chain_id or pair.chain_id)
    a_m1, meta1 = tm_score(structure_path, pair.x1_pdb, config, target_chain_id=pair.x1_chain_id or pair.chain_id)
    delta = a_m1 - a_m0
    assigned = hard_assign(
        a_m0,
        a_m1,
        margin_delta=float(config.get("assignment", {}).get("margin_delta", 0.0)),
        allow_unknown=bool(config.get("assignment", {}).get("allow_unknown", False)),
    )
    return {
        "sample_id": row["sample_id"],
        "sequence_id": row["sequence_id"],
        "pair_id": row["pair_id"],
        "lambda_value": row["lambda_value"],
        "inverse_model": row["inverse_model"],
        "structure_path": structure_path,
        "A_m0": a_m0,
        "A_m1": a_m1,
        "delta_A": delta,
        "assigned_state": assigned,
        "valid_sample": True,
        "assignment_metric": metric,
        "metadata": repr({"x0_alignment": meta0, "x1_alignment": meta1}),
    }


def save_assignment_scores(path: str | Path, rows: list[dict[str, Any]]) -> None:
    write_csv(path, rows, ASSIGNMENT_FIELDS)
