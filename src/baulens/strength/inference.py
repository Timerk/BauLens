"""Serve predictions and local attributions from a fitted strength artifact."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import cast

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]

from baulens.schemas import ConcreteMix, FeatureContribution, StrengthPrediction
from baulens.strength.artifacts import METADATA_PATH, MODEL_PATH, read_fit_files
from baulens.strength.features import RAW_FEATURES
from baulens.strength.modeling import Candidate, predictions
from baulens.strength.splits import Partitions

COVERAGE = 0.90
INTERVAL_METHOD = "split conformal (maximum absolute residual per mix)"


@dataclass(frozen=True)
class FittedStrength:
    pipeline: Pipeline
    candidate: Candidate
    version: str
    data_sha256: str
    radius_mpa: float
    ranges: dict[str, tuple[float, float]]
    feature_means: np.ndarray
    partitions: Partitions

    @classmethod
    def load(cls, model_dir: Path) -> FittedStrength:
        # The manifest uses paths relative to the artifacts root, even when only model/ is shipped.
        files = read_fit_files(model_dir, (MODEL_PATH, METADATA_PATH))
        return cls.from_bytes(files[MODEL_PATH])

    @classmethod
    def from_bytes(cls, model: bytes) -> FittedStrength:
        artifact = joblib.load(BytesIO(model))
        if not isinstance(artifact, cls):
            raise ValueError("Invalid fitted strength pipeline")
        return artifact


class StrengthPredictor:
    """Load once and reuse for requests with the shared ConcreteMix schema."""

    def __init__(self, artifact: FittedStrength) -> None:
        self._artifact = artifact
        self._tree_explainer = None
        if artifact.candidate.family in {"Random forest", "Histogram GBM", "LightGBM"}:
            import shap  # type: ignore[import-untyped]

            self._tree_explainer = shap.TreeExplainer(
                artifact.pipeline.named_steps["regressor"],
                feature_perturbation="tree_path_dependent",
            )

    @classmethod
    def load(cls, model_dir: Path) -> StrengthPredictor:
        """Load a locally trained, trusted joblib artifact and build its SHAP explainer."""
        return cls(FittedStrength.load(model_dir))

    def predict(self, mix: ConcreteMix) -> StrengthPrediction:
        frame = pd.DataFrame([mix.model_dump()], columns=list(RAW_FEATURES))
        point = float(predictions(self._artifact.pipeline, frame)[0])
        features, contributions = self.explain(frame)
        ranks = np.argsort(-np.abs(contributions[0]), kind="stable")[:5]
        return StrengthPrediction(
            point_mpa=point,
            lower_mpa=point - self._artifact.radius_mpa,
            upper_mpa=point + self._artifact.radius_mpa,
            interval_coverage=COVERAGE,
            interval_method=INTERVAL_METHOD,
            in_training_domain=all(
                low <= getattr(mix, name) <= high
                for name, (low, high) in self._artifact.ranges.items()
            ),
            top_features=[
                FeatureContribution(
                    feature=str(features.columns[i]),
                    value=float(features.iloc[0, i]),
                    contribution_mpa=float(contributions[0, i]),
                )
                for i in ranks
            ],
            model_name=self._artifact.candidate.family,
            model_version=self._artifact.version,
        )

    def explain(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        """Return unscaled feature values and additive contributions in MPa."""
        steps = self._artifact.pipeline.named_steps
        features = cast(pd.DataFrame, steps["features"].transform(frame))
        if self._tree_explainer is not None:
            return features, np.asarray(self._tree_explainer.shap_values(features))
        if self._artifact.candidate.family == "Mean":
            return features, np.zeros_like(features.to_numpy(), dtype=float)
        # Linear interventional SHAP with the fitting feature means as the reference.
        weights = steps["regressor"].coef_ / steps["scale"].scale_
        return features, (features.to_numpy() - self._artifact.feature_means) * np.asarray(weights)
