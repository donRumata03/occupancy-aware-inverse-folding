from __future__ import annotations

import sys
from pathlib import Path

import torch

from _bootstrap import add_experiment_to_path

add_experiment_to_path()

from src.dynamicmpnn_weighted_pooling import patch_dynamicmpnn_weighted_pooling
from src.inverse_adapters import lambda_to_state_weights


def assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    if not torch.allclose(actual, expected, atol=1e-6, rtol=1e-6):
        raise AssertionError(f"\nactual={actual}\nexpected={expected}")


def main() -> int:
    sys.path.insert(0, str(Path("external/DynamicMPNN/src").resolve()))
    patch_dynamicmpnn_weighted_pooling()

    from dynamicmpnn.modules.models.models import pool_conformations, pool_edges_conformations

    node_s = torch.tensor([[[1.0], [3.0]], [[10.0], [20.0]]])
    node_v = torch.stack([node_s, node_s + 100.0], dim=-1).unsqueeze(-2)
    node_features = (node_s, node_v)
    node_mask = torch.tensor([[True, True], [True, False]])

    edge_s = torch.tensor([[[2.0], [6.0]], [[5.0], [9.0]]])
    edge_v = torch.stack([edge_s, edge_s + 50.0], dim=-1).unsqueeze(-2)
    edge_features = (edge_s, edge_v)
    edge_mask = torch.tensor([[True, True], [False, True]])

    mean_nodes = pool_conformations(node_features, node_mask)
    weighted_half_nodes = pool_conformations(node_features, node_mask, [0.5, 0.5])
    assert_close(weighted_half_nodes[0], mean_nodes[0])
    assert_close(weighted_half_nodes[1], mean_nodes[1])

    assert_close(pool_conformations(node_features, node_mask, [1.0, 0.0])[0], torch.tensor([[1.0], [10.0]]))
    assert_close(pool_conformations(node_features, node_mask, [0.0, 1.0])[0], torch.tensor([[3.0], [0.0]]))

    mean_edges = pool_edges_conformations(edge_features, edge_mask)
    weighted_half_edges = pool_edges_conformations(edge_features, edge_mask, [0.5, 0.5])
    assert_close(weighted_half_edges[0], mean_edges[0])
    assert_close(weighted_half_edges[1], mean_edges[1])

    assert_close(pool_edges_conformations(edge_features, edge_mask, [1.0, 0.0])[0], torch.tensor([[2.0], [0.0]]))
    assert_close(pool_edges_conformations(edge_features, edge_mask, [0.0, 1.0])[0], torch.tensor([[6.0], [9.0]]))

    assert lambda_to_state_weights(1.0) == (0.5, 0.5)
    print("weighted_pooling_self_check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
