from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .bootstrap import bootstrap_mean_ci
from .utils import write_csv


SEQUENCE_OCCUPANCY_FIELDS = [
    "sequence_id",
    "pair_id",
    "lambda_value",
    "inverse_model",
    "n_valid_samples",
    "n_assigned_0",
    "n_assigned_1",
    "n_unknown",
    "pi_hat_0",
    "pi_hat_1",
    "pi_hat_unknown",
    "mean_A0",
    "mean_A1",
    "mean_delta_A",
    "warnings",
]

RESPONSE_CURVE_FIELDS = [
    "pair_id",
    "lambda_value",
    "inverse_model",
    "n_sequences",
    "mu_hat_1",
    "ci_lower",
    "ci_upper",
    "n_bootstrap",
    "ci",
    "warnings",
]


def aggregate_sequence_occupancies(
    assignment_rows: list[dict[str, Any]],
    expected_m: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in assignment_rows:
        grouped[row["sequence_id"]].append(row)

    output: list[dict[str, Any]] = []
    for sequence_id, rows in sorted(grouped.items()):
        n_valid = len(rows)
        n0 = sum(1 for row in rows if str(row["assigned_state"]) == "0")
        n1 = sum(1 for row in rows if str(row["assigned_state"]) == "1")
        nu = sum(1 for row in rows if str(row["assigned_state"]) == "unknown")
        denom = max(n_valid, 1)
        warnings = []
        if n_valid < expected_m:
            warnings.append(f"n_valid_samples {n_valid} < requested {expected_m}")
        a0 = np.asarray([float(row["A_m0"]) for row in rows], dtype=float)
        a1 = np.asarray([float(row["A_m1"]) for row in rows], dtype=float)
        delta = np.asarray([float(row["delta_A"]) for row in rows], dtype=float)
        first = rows[0]
        output.append(
            {
                "sequence_id": sequence_id,
                "pair_id": first["pair_id"],
                "lambda_value": first["lambda_value"],
                "inverse_model": first["inverse_model"],
                "n_valid_samples": n_valid,
                "n_assigned_0": n0,
                "n_assigned_1": n1,
                "n_unknown": nu,
                "pi_hat_0": n0 / denom,
                "pi_hat_1": n1 / denom,
                "pi_hat_unknown": nu / denom,
                "mean_A0": float(a0.mean()) if a0.size else float("nan"),
                "mean_A1": float(a1.mean()) if a1.size else float("nan"),
                "mean_delta_A": float(delta.mean()) if delta.size else float("nan"),
                "warnings": "; ".join(warnings),
            }
        )
    return output


def aggregate_response_curves(
    sequence_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in sequence_rows:
        grouped[(row["pair_id"], str(row["lambda_value"]), row["inverse_model"])].append(float(row["pi_hat_1"]))

    output: list[dict[str, Any]] = []
    n_bootstrap = int(config.get("bootstrap", {}).get("n_bootstrap", 10000))
    ci = float(config.get("bootstrap", {}).get("ci", 0.95))
    seed = int(config.get("random_seed", 0))
    for (pair_id, lambda_value, inverse_model), values in sorted(grouped.items(), key=lambda x: (x[0][0], float(x[0][1]))):
        warnings = []
        finite_values = [v for v in values if np.isfinite(v)]
        if finite_values and all(v in {0.0, 1.0} for v in finite_values):
            warnings.append("all pi_hat_1 values are exactly 0 or 1")
        lower, upper = bootstrap_mean_ci(finite_values, n_bootstrap=n_bootstrap, ci=ci, seed=seed)
        output.append(
            {
                "pair_id": pair_id,
                "lambda_value": lambda_value,
                "inverse_model": inverse_model,
                "n_sequences": len(finite_values),
                "mu_hat_1": float(np.mean(finite_values)) if finite_values else float("nan"),
                "ci_lower": lower,
                "ci_upper": upper,
                "n_bootstrap": n_bootstrap,
                "ci": ci,
                "warnings": "; ".join(warnings),
            }
        )
    return output


def save_sequence_occupancies(path: str | Path, rows: list[dict[str, Any]]) -> None:
    write_csv(path, rows, SEQUENCE_OCCUPANCY_FIELDS)


def save_response_curves(path: str | Path, rows: list[dict[str, Any]]) -> None:
    write_csv(path, rows, RESPONSE_CURVE_FIELDS)

