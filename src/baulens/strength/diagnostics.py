"""Measure holdout errors and visualise predictions and model attributions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from baulens.strength.artifacts import atomic_file
from baulens.strength.features import RAW_FEATURES, TARGET
from baulens.strength.inference import StrengthPredictor

AGE_BUCKETS = ("1-7", "8-28", "29-99", "100+")
STRENGTH_BUCKETS = ("<20", "20 to <40", "40 to <60", "60+")
RESIDUAL_FIGURE = "residuals.png"
SHAP_FIGURE = "shap-summary.png"
DEPENDENCE_FIGURE = "shap-dependence.png"


@dataclass(frozen=True)
class ErrorSlice:
    label: str
    rows: int
    mixes: int
    mae: float | None
    bias: float | None
    coverage: float | None
    width: float | None


@dataclass(frozen=True)
class ShapSummary:
    feature: str
    magnitude_mpa: float
    rank_correlation: float | None


def error_summary(rows: pd.DataFrame, label: str) -> ErrorSlice:
    count = len(rows)
    return ErrorSlice(
        label,
        count,
        int(rows["mix_id"].nunique()),
        float(rows["residual_mpa"].abs().mean()) if count else None,
        float(rows["residual_mpa"].mean()) if count else None,
        float(rows["covered"].mean()) if count else None,
        float((rows["upper_mpa"] - rows["lower_mpa"]).mean()) if count else None,
    )


def slice_errors(test: pd.DataFrame) -> tuple[list[ErrorSlice], list[ErrorSlice]]:
    ages = pd.cut(test["age_days"], [0, 7, 28, 99, np.inf], labels=list(AGE_BUCKETS))
    strengths = pd.cut(
        test[TARGET], [-np.inf, 20, 40, 60, np.inf], labels=list(STRENGTH_BUCKETS), right=False
    )
    return (
        [error_summary(test[ages == label], label) for label in AGE_BUCKETS],
        [error_summary(test[strengths == label], label) for label in STRENGTH_BUCKETS],
    )


def worst_mixes(test: pd.DataFrame) -> list[dict[str, float | None]]:
    errors = test.assign(absolute_error=test["residual_mpa"].abs())
    worst_row = errors.sort_values("absolute_error", ascending=False, kind="stable")
    worst_row = worst_row.drop_duplicates("mix_id").head(5).copy()
    worst_row["measurements"] = worst_row["mix_id"].map(errors.groupby("mix_id").size())
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        ratios = worst_row["water"] / worst_row["cement"]
    worst_row["water_cement"] = ratios.astype(object).where(np.isfinite(ratios), None)
    columns = [
        "mix_id",
        "measurements",
        "age_days",
        "water_cement",
        TARGET,
        "predicted_mpa",
        "residual_mpa",
    ]
    return cast(list[dict[str, float | None]], worst_row.loc[:, columns].to_dict(orient="records"))


def plot_residuals(test: pd.DataFrame, directory: Path) -> None:
    figure = Figure(figsize=(6.3, 3.5), layout="constrained")
    axis = figure.subplots()
    axis.axhline(0, color="#58677c", linewidth=1, linestyle="--")
    scatter = axis.scatter(
        test["predicted_mpa"],
        test["residual_mpa"],
        c=np.log10(test["age_days"].clip(lower=1)),
        cmap="viridis",
        vmin=0,
        vmax=np.log10(max(365, test["age_days"].max())),
        alpha=0.8,
        s=22,
        linewidths=0,
    )
    axis.set(
        xlabel="Predicted strength (MPa)",
        ylabel="Observed minus predicted (MPa)",
        title="Residuals on held-out exact tuples",
    )
    cb = figure.colorbar(scatter, ax=axis, label="Age (days, log scale)", pad=0.02)
    cb.set_ticks(np.log10([1, 7, 28, 91, 365]).tolist(), labels=["1", "7", "28", "91", "365"])
    _save(figure, directory / RESIDUAL_FIGURE)


def plot_shap(
    predictor: StrengthPredictor, calibration: pd.DataFrame, directory: Path, seed: int
) -> list[ShapSummary]:
    features, values = predictor.explain(calibration.loc[:, list(RAW_FEATURES)])
    names = features.columns.to_list()
    order = np.argsort(-np.abs(values).mean(axis=0), kind="stable")
    summary = [
        ShapSummary(
            str(names[i]),
            float(np.abs(values[:, i]).mean()),
            _rank_correlation(features.iloc[:, i], values[:, i]),
        )
        for i in order
    ]
    figure = Figure(figsize=(7.0, 4.5), layout="constrained")
    axis = figure.subplots()
    rng = np.random.default_rng(seed)
    axis.axvline(0, color="#58677c", linewidth=1)
    top = order[:10][::-1]
    for y, column in enumerate(top):
        feature_rank = features.iloc[:, column].rank(pct=True).to_numpy()
        cloud = axis.scatter(
            values[:, column],
            y + rng.uniform(-0.24, 0.24, len(values)),
            c=feature_rank,
            cmap="coolwarm",
            vmin=0,
            vmax=1,
            s=14,
            alpha=0.78,
            linewidths=0,
        )
    axis.set_yticks(range(len(top)), [_feature_label(names[i]) for i in top])
    axis.set(
        xlabel="SHAP contribution to predicted strength (MPa)", title="SHAP on calibration mixes"
    )
    figure.colorbar(cloud, ax=axis, ticks=[0, 1], pad=0.02).set_ticklabels(
        ["Low value", "High value"]
    )
    _save(figure, directory / SHAP_FIGURE)

    indices = list(
        dict.fromkeys(
            [
                *map(int, order[:3]),
                *(
                    names.index(name)
                    for name in ("age_days", "water_cement", "water_binder")
                    if name in names
                ),
            ]
        )
    )
    _plot_dependence(features, values, indices, directory / DEPENDENCE_FIGURE)
    return summary


def _plot_dependence(
    features: pd.DataFrame, values: np.ndarray, indices: list[int], destination: Path
) -> None:
    columns = 2 if len(indices) == 4 else 3
    rows = (len(indices) + columns - 1) // columns
    figure = Figure(figsize=(2.85 * columns, 2.6 * rows + 0.3), layout="constrained")
    axes = figure.subplots(rows, columns, squeeze=False).ravel()
    for axis, i in zip(axes[: len(indices)], indices, strict=True):
        name = str(features.columns[i])
        axis.scatter(features.iloc[:, i], values[:, i], c="#285e8e", s=16, alpha=0.56, linewidths=0)
        axis.axhline(0, color="#58677c", linewidth=0.8, linestyle="--")
        axis.set(xlabel=_feature_label(name), ylabel="SHAP (MPa)")
        axis.xaxis.set_major_locator(MaxNLocator(4))
        if name == "age_days":
            axis.set_xscale("log")
            axis.set_xticks([1, 7, 28, 91, 365], labels=["1", "7", "28", "91", "365"])
    for axis in axes[len(indices) :]:
        axis.set_visible(False)
    figure.suptitle("SHAP dependence on calibration mixes")
    _save(figure, destination)


def _rank_correlation(feature: pd.Series[float], contributions: np.ndarray) -> float | None:
    if feature.nunique() < 2 or np.ptp(contributions) == 0:
        return None
    return float(feature.rank().corr(pd.Series(contributions, index=feature.index).rank()))


def _feature_label(name: str) -> str:
    labels = {
        "age_days": "Age (days)",
        "log_age": "Log age",
        "water_cement": "Water/cement",
        "water_binder": "Water/binder",
        "aggregate_binder": "Aggregate/binder",
        "coarse_fine_aggregate": "Coarse/fine aggregate",
        "superplasticizer_binder": "Superplasticizer/binder",
        "scm_fraction": "SCM fraction",
    }
    return labels.get(name, name.replace("_", " ").capitalize())


def _save(figure: Figure, destination: Path) -> None:
    with atomic_file(destination) as staged:
        figure.savefig(staged, dpi=118)
