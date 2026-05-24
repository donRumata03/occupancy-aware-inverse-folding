from __future__ import annotations

from typing import Sequence

import numpy as np


def bootstrap_mean_ci(
    values: Sequence[float],
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    sample_idx = rng.integers(0, arr.size, size=(int(n_bootstrap), arr.size))
    means = arr[sample_idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))

