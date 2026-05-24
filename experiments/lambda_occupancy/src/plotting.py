from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def plot_response_curves(rows: list[dict[str, Any]], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    lambda_values: set[float] = set()
    for row in rows:
        grouped[row["pair_id"]].append(row)
        lambda_value = _as_float(row["lambda_value"])
        if lambda_value > 0:
            lambda_values.add(lambda_value)
    if not grouped:
        ax.text(0.5, 0.5, "No response-curve rows available", ha="center", va="center")
    for pair_id, pair_rows in grouped.items():
        pair_rows = sorted(pair_rows, key=lambda row: _as_float(row["lambda_value"]))
        x = [_as_float(row["lambda_value"]) for row in pair_rows]
        y = [_as_float(row["mu_hat_1"]) for row in pair_rows]
        lower = [_as_float(row["ci_lower"]) for row in pair_rows]
        upper = [_as_float(row["ci_upper"]) for row in pair_rows]
        ax.plot(x, y, marker="o", label=pair_id)
        ax.fill_between(x, lower, upper, alpha=0.15)
    ax.set_xscale("log")
    if lambda_values:
        ticks = sorted(lambda_values)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{value:g}" for value in ticks])
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("lambda")
    ax.set_ylabel("mu_hat_1(lambda)")
    ax.set_title("Lambda occupancy response curves")
    if grouped:
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_per_sequence_distributions(
    rows: list[dict[str, Any]],
    selected_lambdas: list[float],
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected = {float(x) for x in selected_lambdas}
    filtered = [row for row in rows if _as_float(row["lambda_value"]) in selected]
    fig, ax = plt.subplots(figsize=(10, 5))
    if not filtered:
        ax.text(0.5, 0.5, "No sequence occupancy rows available", ha="center", va="center")
    groups: dict[str, list[float]] = defaultdict(list)
    for row in filtered:
        key = f"{row['pair_id']}\nlambda={_as_float(row['lambda_value']):g}"
        groups[key].append(_as_float(row["pi_hat_1"]))
    labels = list(groups)
    values = [groups[label] for label in labels]
    if values:
        ax.boxplot(values, labels=labels, showfliers=False)
        for idx, vals in enumerate(values, start=1):
            ax.scatter([idx] * len(vals), vals, s=18, alpha=0.7)
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("pi_hat_1(S)")
    ax.set_title("Per-sequence occupancy distributions")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_delta_score_distributions(rows: list[dict[str, Any]], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        key = f"{row['pair_id']}\nlambda={_as_float(row['lambda_value']):g}"
        grouped[key].append(_as_float(row["delta_A"]))
    fig, ax = plt.subplots(figsize=(10, 5))
    if not grouped:
        ax.text(0.5, 0.5, "No assignment rows available", ha="center", va="center")
    labels = list(grouped)
    values = [grouped[label] for label in labels]
    if values:
        ax.violinplot(values, showmeans=True, showextrema=True)
        ax.set_xticks(range(1, len(labels) + 1), labels, rotation=30, ha="right")
        ax.axhline(0.0, color="black", linewidth=1)
    ax.set_ylabel("delta_A = A_m1 - A_m0")
    ax.set_title("Delta score distributions")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
