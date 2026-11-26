"""Threshold selection, classification metrics, bootstrap intervals and calibration."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import average_precision_score, roc_auc_score  # type: ignore[import-untyped]

type FloatArray = NDArray[np.float64]
type IntArray = NDArray[np.int64]

THRESHOLD_GRID = np.round(np.arange(1, 1000) / 1000, 3)
POINT_METRICS = ("precision", "recall", "f1", "accuracy", "roc_auc", "pr_auc")


def _counts(
    labels: IntArray, probabilities: FloatArray, thresholds: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    predicted = probabilities[None, :] >= thresholds[:, None]
    positive = labels.astype(bool)[None, :]
    tp = (predicted & positive).sum(axis=1).astype(np.float64)
    fp = (predicted & ~positive).sum(axis=1).astype(np.float64)
    fn = (~predicted & positive).sum(axis=1).astype(np.float64)
    tn = (~predicted & ~positive).sum(axis=1).astype(np.float64)
    return tp, fp, fn, tn


def choose_threshold(
    labels: IntArray, probabilities: FloatArray, strategy: Literal["f1", "youden"]
) -> float:
    """Best threshold on a 0.001 grid; ties go to the threshold closest to 0.5.

    With near-perfect separation many thresholds share the optimum. Taking the one nearest
    0.5 avoids an arbitrary pick at the edge of the plateau, where a slightly shifted test
    distribution would flip predictions first.
    """
    tp, fp, fn, tn = _counts(labels, probabilities, THRESHOLD_GRID)
    if strategy == "f1":
        score = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp), where=tp + fp + fn > 0)
    else:
        score = tp / np.maximum(tp + fn, 1) + tn / np.maximum(tn + fp, 1) - 1
    best = np.flatnonzero(np.isclose(score, score.max(), rtol=0, atol=1e-12))
    return float(THRESHOLD_GRID[best[np.argmin(np.abs(THRESHOLD_GRID[best] - 0.5))]])


def binary_metrics(
    labels: IntArray, probabilities: FloatArray, threshold: float
) -> dict[str, float]:
    tp, fp, fn, tn = (float(c[0]) for c in _counts(labels, probabilities, np.array([threshold])))
    both_classes = 0 < labels.sum() < len(labels)
    return {
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0,
        "accuracy": (tp + tn) / len(labels),
        "roc_auc": float(roc_auc_score(labels, probabilities)) if both_classes else float("nan"),
        "pr_auc": (
            float(average_precision_score(labels, probabilities)) if both_classes else float("nan")
        ),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def bootstrap_intervals(
    labels: IntArray,
    probabilities: FloatArray,
    threshold: float,
    *,
    resamples: int,
    seed: int,
    level: float = 0.95,
) -> dict[str, tuple[float, float]]:
    """Percentile bootstrap over test images with the threshold held fixed."""
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {name: [] for name in POINT_METRICS}
    for _ in range(resamples):
        draw = rng.integers(0, len(labels), len(labels))
        metrics = binary_metrics(labels[draw], probabilities[draw], threshold)
        for name in POINT_METRICS:
            samples[name].append(metrics[name])
    tail = (1 - level) / 2 * 100
    return {
        name: (
            float(np.nanpercentile(values, tail)),
            float(np.nanpercentile(values, 100 - tail)),
        )
        for name, values in samples.items()
    }


@dataclass(frozen=True)
class Calibration:
    ece: float
    bin_edges: list[float]
    mean_confidence: list[float | None]
    fraction_positive: list[float | None]
    counts: list[int]


def calibration(labels: IntArray, probabilities: FloatArray, bins: int) -> Calibration:
    """Reliability curve of P(crack) on equal-width bins and the expected calibration error."""
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(probabilities, edges[1:-1], right=False), 0, bins - 1)
    confidence: list[float | None] = []
    observed: list[float | None] = []
    counts: list[int] = []
    ece = 0.0
    for b in range(bins):
        mask = which == b
        count = int(mask.sum())
        counts.append(count)
        if count == 0:
            confidence.append(None)
            observed.append(None)
            continue
        mean_p, frac = float(probabilities[mask].mean()), float(labels[mask].mean())
        confidence.append(mean_p)
        observed.append(frac)
        ece += count / len(labels) * abs(mean_p - frac)
    return Calibration(ece, edges.tolist(), confidence, observed, counts)
