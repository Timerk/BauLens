"""Candidate pipelines and paired outer/inner cross-validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.dummy import DummyRegressor  # type: ignore[import-untyped]
from sklearn.ensemble import (  # type: ignore[import-untyped]
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.inspection import permutation_importance  # type: ignore[import-untyped]
from sklearn.linear_model import LinearRegression, Ridge  # type: ignore[import-untyped]
from sklearn.model_selection import GridSearchCV  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from baulens.config import StrengthConfig
from baulens.strength.features import RAW_FEATURES, MixFeatures
from baulens.strength.metrics import Scores, score_predictions
from baulens.strength.splits import Fold, fold_indices

type Family = Literal["Mean", "Linear", "Ridge", "Random forest", "Histogram GBM", "LightGBM"]
type Parameters = dict[str, int | float]

FAMILIES: tuple[Family, ...] = (
    "Mean",
    "Linear",
    "Ridge",
    "Random forest",
    "Histogram GBM",
    "LightGBM",
)


@dataclass(frozen=True)
class Candidate:
    family: Family
    engineered: bool

    @property
    def features(self) -> str:
        return "Raw + engineered" if self.engineered else "Raw"


@dataclass(frozen=True)
class FoldResult:
    candidate: Candidate
    protocol: Literal["grouped", "row"]
    repeat: int
    fold: int
    scores: Scores
    overlapping_validation_rows: int
    validation_rows: int
    parameters: Parameters


def fit_candidate(
    candidate: Candidate,
    inputs: pd.DataFrame,
    y: pd.Series[float],
    inner: list[Fold],
    config: StrengthConfig,
) -> tuple[Pipeline, Parameters]:
    grid: dict[str, list[int | float]]
    match candidate.family:
        case "Mean":
            estimator, grid = DummyRegressor(strategy="mean"), {}
        case "Linear":
            estimator, grid = LinearRegression(), {}
        case "Ridge":
            estimator = Ridge()
            grid = {"regressor__alpha": [0.1, 1.0, 10.0, 100.0]}
        case "Random forest":
            estimator = RandomForestRegressor(
                n_estimators=80, random_state=config.seed, n_jobs=1, max_features=0.8
            )
            grid = {"regressor__min_samples_leaf": [1, 3, 6]}
        case "Histogram GBM":
            estimator = HistGradientBoostingRegressor(
                max_iter=180, learning_rate=0.08, early_stopping=False, random_state=config.seed
            )
            grid = {"regressor__max_leaf_nodes": [7, 15, 31]}
        case "LightGBM":
            estimator = LGBMRegressor(
                n_estimators=220,
                learning_rate=0.05,
                min_child_samples=8,
                verbosity=-1,
                random_state=config.seed,
                n_jobs=config.threads,
                deterministic=True,
                force_col_wise=True,
            )
            grid = {"regressor__num_leaves": [7, 15, 31]}

    pipeline = Pipeline(
        [
            ("features", MixFeatures(engineered=candidate.engineered)),
            (
                "scale",
                StandardScaler() if candidate.family in {"Linear", "Ridge"} else "passthrough",
            ),
            ("regressor", estimator),
        ]
    )
    if not grid:
        pipeline.fit(inputs, y)
        return pipeline, {}
    search = GridSearchCV(
        pipeline, grid, scoring="neg_mean_absolute_error", cv=inner, error_score="raise"
    )
    search.fit(inputs, y)
    return search.best_estimator_, {
        name.removeprefix("regressor__"): value for name, value in search.best_params_.items()
    }


def cross_validate(
    candidates: list[Candidate],
    inputs: pd.DataFrame,
    y: pd.Series[float],
    groups: pd.Series[int],
    config: StrengthConfig,
    *,
    grouped: bool,
    explain: bool = False,
) -> tuple[list[FoldResult], dict[str, float]]:
    results: list[FoldResult] = []
    importances: list[np.ndarray] = []
    for repeat in range(config.cv_repeats):
        seed = (config.seed + repeat) % 2**32
        outer = fold_indices(groups, config.cv_splits, seed, grouped=grouped)
        for fold, (fit_idx, val_idx) in enumerate(outer, start=1):
            fit_groups = groups.iloc[fit_idx]
            val_groups = groups.iloc[val_idx]
            validation_overlap = int(val_groups.isin(fit_groups).sum())
            inner = inner_folds(fit_groups, config, seed, grouped=grouped)
            for candidate in candidates:
                pipeline, parameters = fit_candidate(
                    candidate,
                    inputs.iloc[fit_idx],
                    y.iloc[fit_idx],
                    inner,
                    config,
                )
                validation_inputs, y_val = inputs.iloc[val_idx], y.iloc[val_idx]
                scores = score_predictions(y_val.to_numpy(), pipeline.predict(validation_inputs))
                results.append(
                    FoldResult(
                        candidate,
                        "grouped" if grouped else "row",
                        repeat + 1,
                        fold,
                        scores,
                        validation_overlap,
                        len(val_idx),
                        parameters,
                    )
                )
                if explain:
                    # Permute raw inputs upstream so each ratio is recomputed consistently.
                    importance = permutation_importance(
                        pipeline,
                        validation_inputs,
                        y_val,
                        scoring="neg_mean_absolute_error",
                        n_repeats=4,
                        random_state=seed,
                        n_jobs=1,
                    )
                    importances.append(importance.importances_mean)
    if not importances:
        return results, {}
    means = np.asarray(importances).mean(axis=0)
    return results, dict(zip(RAW_FEATURES, map(float, means), strict=True))


def inner_folds(
    groups: pd.Series[int], config: StrengthConfig, seed: int, *, grouped: bool = True
) -> list[Fold]:
    available = groups.nunique() if grouped else len(groups)
    if available < 2:
        raise ValueError("Too few distinct fitting samples for the inner CV")
    return fold_indices(groups, min(config.inner_splits, available), seed, grouped=grouped)


def predictions(pipeline: Pipeline, inputs: pd.DataFrame) -> np.ndarray:
    return cast(np.ndarray, pipeline.predict(inputs))
