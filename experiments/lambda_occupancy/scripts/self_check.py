from __future__ import annotations

import math
import tempfile
from pathlib import Path

from _bootstrap import add_experiment_to_path

add_experiment_to_path()

from src.aggregate import aggregate_response_curves, aggregate_sequence_occupancies
from src.assignment import hard_assign
from src.bootstrap import bootstrap_mean_ci
from src.inverse_adapters import GeneratedSequence, lambda_to_state_weights, load_generated_sequences_csv, save_generated_sequences
from src.utils import read_csv, write_csv


def test_lambda_weights() -> None:
    assert lambda_to_state_weights(1.0) == (0.5, 0.5)
    w0, w1 = lambda_to_state_weights(3.0)
    assert math.isclose(w0, 0.25)
    assert math.isclose(w1, 0.75)


def test_assignment() -> None:
    assert hard_assign(0.4, 0.6) == "1"
    assert hard_assign(0.6, 0.4) == "0"
    assert hard_assign(0.5, 0.51, margin_delta=0.02, allow_unknown=True) == "unknown"


def test_aggregation() -> None:
    rows = [
        {"sequence_id": "s1", "pair_id": "p1", "lambda_value": "1.0", "inverse_model": "manual", "assigned_state": "1", "A_m0": "0.3", "A_m1": "0.7", "delta_A": "0.4"},
        {"sequence_id": "s1", "pair_id": "p1", "lambda_value": "1.0", "inverse_model": "manual", "assigned_state": "0", "A_m0": "0.8", "A_m1": "0.2", "delta_A": "-0.6"},
        {"sequence_id": "s2", "pair_id": "p1", "lambda_value": "1.0", "inverse_model": "manual", "assigned_state": "1", "A_m0": "0.1", "A_m1": "0.9", "delta_A": "0.8"},
    ]
    seq_rows = aggregate_sequence_occupancies(rows, expected_m=2)
    assert len(seq_rows) == 2
    s1 = next(row for row in seq_rows if row["sequence_id"] == "s1")
    assert math.isclose(s1["pi_hat_1"], 0.5)
    curves = aggregate_response_curves(seq_rows, {"bootstrap": {"n_bootstrap": 100, "ci": 0.95}, "random_seed": 0})
    assert len(curves) == 1
    assert math.isclose(curves[0]["mu_hat_1"], 0.75)


def test_bootstrap() -> None:
    lower, upper = bootstrap_mean_ci([0.0, 0.5, 1.0], n_bootstrap=100, ci=0.9, seed=0)
    assert math.isfinite(lower)
    assert math.isfinite(upper)
    assert lower <= upper


def test_csv_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        seq_path = tmp_path / "generated_sequences.csv"
        records = [
            GeneratedSequence(
                sequence_id="s1",
                pair_id="p1",
                lambda_value=1.0,
                inverse_model="manual",
                seed=1,
                sequence="ACDEFGHIK",
                temperature=0.1,
                metadata={"note": "roundtrip"},
            )
        ]
        save_generated_sequences(seq_path, records)
        loaded = load_generated_sequences_csv(seq_path)
        assert loaded[0].sequence == "ACDEFGHIK"

        assign_path = tmp_path / "assignment_scores.csv"
        write_csv(
            assign_path,
            [
                {
                    "sample_id": "m1",
                    "sequence_id": "s1",
                    "pair_id": "p1",
                    "lambda_value": "1.0",
                    "inverse_model": "manual",
                    "structure_path": "x.pdb",
                    "A_m0": "0.2",
                    "A_m1": "0.8",
                    "delta_A": "0.6",
                    "assigned_state": "1",
                    "valid_sample": "True",
                    "assignment_metric": "tm_score",
                    "metadata": "{}",
                }
            ],
            [
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
            ],
        )
        loaded_assignment = read_csv(assign_path)
        assert loaded_assignment[0]["A_m1"] == "0.8"


def main() -> int:
    tests = [
        test_lambda_weights,
        test_assignment,
        test_aggregation,
        test_bootstrap,
        test_csv_roundtrip,
    ]
    for test in tests:
        test()
        print(f"ok - {test.__name__}")
    print("self_check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

