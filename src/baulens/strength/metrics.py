"""Point metrics and uncertainty with whole-mix resampling."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np
import pandas as pd
from scipy.stats import t  # type: ignore[import-untyped]
from sklearn.metrics import (  # type: ignore[import-untyped]
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)


@dataclass(frozen=True)
class Scores:
    mae: float
    rmse: float
    r2: float | None


@dataclass(frozen=True)
class CVSummary:
    mean: float | None
    std: float | None
    ci: tuple[float, float] | None


def score_predictions(actual: np.ndarray, predicted: np.ndarray) -> Scores:
    return Scores(
        float(mean_absolute_error(actual, predicted)),
        float(root_mean_squared_error(actual, predicted)),
        float(r2_score(actual, predicted)) if len(actual) > 1 else None,
    )


def summarize_cv(values: list[float | None], repeats: list[int]) -> CVSummary:
    scores = pd.Series(values, dtype=float)
    count = int(scores.count())
    if count == 0:
        return CVSummary(None, None, None)
    mean = float(scores.mean())
    fold_sd = float(scores.std(ddof=1)) if count > 1 else None
    repeat_means = scores.groupby(repeats).mean().dropna()
    n = len(repeat_means)
    if n < 2:
        return CVSummary(mean, fold_sd, None)
    margin = float(t.ppf(0.975, n - 1) * repeat_means.std(ddof=1) / np.sqrt(n))
    center = float(repeat_means.mean())
    return CVSummary(mean, fold_sd, (center - margin, center + margin))


def conformal_radius(errors: np.ndarray, groups: pd.Series[int], coverage: float) -> float:
    """Calibrate on one maximum absolute residual per calibration mix."""
    scores = pd.Series(np.abs(errors), index=groups.index).groupby(groups).max().to_numpy()
    rank = ceil((len(scores) + 1) * coverage)
    if rank > len(scores):
        raise ValueError(f"Too few calibration mixes for finite {coverage:.0%} conformal intervals")
    return float(np.partition(scores, rank - 1)[rank - 1])


def bootstrap_metrics(
    actual: np.ndarray, predicted: np.ndarray, groups: pd.Series[int], resamples: int, seed: int
) -> dict[str, tuple[float | None, float | None]]:
    """Use group sufficient statistics so sampled repeats keep every row in each selected mix."""
    centered = actual - actual.mean()
    stats = (
        pd.DataFrame(
            {
                "n": 1,
                "absolute": np.abs(actual - predicted),
                "squared": (actual - predicted) ** 2,
                "target": centered,
                "target_squared": centered**2,
            },
            index=groups.index,
        )
        .groupby(groups)
        .sum()
        .to_numpy()
    )
    generator = np.random.default_rng(seed)
    selections = generator.integers(len(stats), size=(resamples, len(stats)))
    n, absolute, squared, target, target_squared = stats[selections].sum(axis=1).T
    sst = target_squared - target**2 / n
    values = {
        "mae": absolute / n,
        "rmse": np.sqrt(squared / n),
        "r2": 1 - squared[sst > 0] / sst[sst > 0],
    }
    return {metric: _percentiles(samples) for metric, samples in values.items()}


def _percentiles(samples: np.ndarray) -> tuple[float | None, float | None]:
    if not len(samples):
        return None, None
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)
