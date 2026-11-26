"""Physics-informed features computed inside the fitted sklearn pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, TransformerMixin  # type: ignore[import-untyped]
from sklearn.utils.validation import check_is_fitted  # type: ignore[import-untyped]

from baulens.data.concrete import CONCRETE_COLUMNS

RAW_FEATURES = CONCRETE_COLUMNS[:-1]
INGREDIENTS = RAW_FEATURES[:-1]
TARGET = CONCRETE_COLUMNS[-1]
FEATURE_RATIONALES = {
    "water_cement": "Abrams' law relates water per unit of cement to strength through porosity.",
    "water_binder": "Water per unit of binder accounts for slag and fly ash in blended mixes.",
    "log_age": "Log age represents the slowing rate of strength gain as hydration progresses.",
    "scm_fraction": "Slag and fly ash track changes to hydraulic and pozzolanic reactions.",
    "superplasticizer_binder": "Binder-normalised dosage relates to dispersion and water demand.",
    "coarse_fine_aggregate": "Coarse-to-fine balance can affect particle packing and water demand.",
    "aggregate_binder": "Aggregate relative to binder tracks the balance of skeleton and paste.",
    "total_mass": "Summed mass per cubic metre is a proxy for mixture density.",
}


class MixFeatures(TransformerMixin, BaseEstimator):  # type: ignore[misc]
    """Keep raw inputs and optionally add physically motivated relationships.

    Water/cement links capillary porosity to strength by Abrams' law.
    Water/binder accounts for slag and fly ash in blended mixtures.
    Log age approximates diminishing strength gain with ongoing hydration.
    SCM fraction tracks changes to hydraulic and pozzolanic reactions.
    Superplasticizer/binder normalises the dosage for water reduction and dispersion.
    Coarse/fine aggregate describes the balance that affects packing and water demand.
    Aggregate/binder approximates the balance of aggregate skeleton and binder paste.
    Total mass per cubic metre serves as a proxy for mixture density.
    """

    def __init__(self, engineered: bool = True) -> None:
        self.engineered = engineered

    def fit(self, inputs: pd.DataFrame, y: pd.Series[float] | None = None) -> MixFeatures:
        self.feature_names_in_ = np.asarray(inputs.loc[:, list(RAW_FEATURES)].columns, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, inputs: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "feature_names_in_")
        features = inputs.loc[:, list(RAW_FEATURES)].copy()
        if self.engineered:
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                cement = features["cement"]
                scm = features["blast_furnace_slag"] + features["fly_ash"]
                binder = cement + scm
                coarse, fine = features["coarse_aggregate"], features["fine_aggregate"]
                features = features.assign(
                    water_cement=_safe_ratio(features["water"], cement),
                    water_binder=_safe_ratio(features["water"], binder),
                    log_age=np.log(features["age_days"].clip(lower=1)),
                    scm_fraction=_safe_ratio(scm, binder),
                    superplasticizer_binder=_safe_ratio(features["superplasticizer"], binder),
                    coarse_fine_aggregate=_safe_ratio(coarse, fine),
                    aggregate_binder=_safe_ratio(coarse + fine, binder),
                    total_mass=features.loc[:, list(INGREDIENTS)].sum(axis=1),
                )
            names = list(FEATURE_RATIONALES)
            nonfinite = ~np.isfinite(features.loc[:, names].to_numpy()).all(axis=0)
            if nonfinite.any():
                invalid = ", ".join(np.asarray(names)[nonfinite])
                raise ValueError(
                    f"Ingredient amounts produce non-finite engineered features: {invalid}"
                )
        return features

    def get_feature_names_out(self, input_features: ArrayLike | None = None) -> np.ndarray:
        check_is_fitted(self, "feature_names_in_")
        if input_features is not None and not np.array_equal(
            input_features, self.feature_names_in_
        ):
            raise ValueError("Expected canonical raw feature names")
        names = (*RAW_FEATURES, *FEATURE_RATIONALES) if self.engineered else RAW_FEATURES
        return np.asarray(names, dtype=object)


def _safe_ratio(numerator: pd.Series[float], denominator: pd.Series[float]) -> pd.Series[float]:
    """Use a neutral finite value for a zero denominator in malformed input tables."""
    return (numerator / denominator.mask(denominator == 0)).where(denominator != 0, 0.0)
