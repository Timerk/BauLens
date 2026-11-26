"""Load the analysis models once per process and describe what was loaded.

A component whose artifacts are missing or fail verification is kept as an
:class:`Unavailable` record instead of stopping the service: the other endpoints keep
working, the affected ones answer 503, and ``/health`` says why.
"""

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from baulens.api.models import (
    ComponentStatus,
    CrackModelInfo,
    FeatureRange,
    StrengthModelInfo,
)
from baulens.artifacts import ArtifactError
from baulens.config import ApiConfig
from baulens.cracks.inference import CrackClassifier
from baulens.schemas import ConcreteMix, StrengthPrediction
from baulens.strength.artifacts import METADATA_PATH, MODEL_PATH, read_fit_files
from baulens.strength.inference import FittedStrength, StrengthPredictor

logger = logging.getLogger(__name__)

_UNITS = {"age_days": "d"}
_STRENGTH_HINT = "run `make strength` (or mount artifacts/strength/model)"
_CRACKS_HINT = "run `make cracks` (or mount artifacts/cracks/model)"


@dataclass(frozen=True)
class Unavailable:
    detail: str

    def status(self) -> ComponentStatus:
        return ComponentStatus(ready=False, detail=self.detail)


class StrengthService:
    """Strength predictor plus the summary served by ``/models``.

    Predictions are serialised with a lock: they take ~20 ms and the SHAP explainer makes
    no thread-safety promise.
    """

    def __init__(self, predictor: StrengthPredictor, info: StrengthModelInfo) -> None:
        self._predictor = predictor
        self._lock = threading.Lock()
        self.info = info

    @classmethod
    def load(cls, model_dir: Path, metrics_path: Path) -> "StrengthService":
        files = read_fit_files(model_dir, (MODEL_PATH, METADATA_PATH))
        artifact = FittedStrength.from_bytes(files[MODEL_PATH])
        metadata: dict[str, Any] = json.loads(files[METADATA_PATH])
        predictor = StrengthPredictor(artifact)
        info = StrengthModelInfo(
            model_name=artifact.candidate.family,
            model_version=artifact.version,
            interval_coverage=float(metadata.get("interval_coverage", 0.9)),
            interval_method=str(metadata.get("interval_method", "split conformal")),
            interval_half_width_mpa=artifact.radius_mpa,
            training_rows=metadata.get("partitions", {}).get("fit", {}).get("rows"),
            training_ranges=[
                FeatureRange(
                    feature=name, minimum=low, maximum=high, unit=_UNITS.get(name, "kg/m³")
                )
                for name, (low, high) in artifact.ranges.items()
            ],
            test_metrics=_strength_metrics(metrics_path, artifact.version),
        )
        return cls(predictor, info)

    def status(self) -> ComponentStatus:
        return ComponentStatus(
            ready=True,
            detail="loaded",
            model_name=self.info.model_name,
            model_version=self.info.model_version,
        )

    def predict(self, mix: ConcreteMix) -> StrengthPrediction:
        with self._lock:
            return self._predictor.predict(mix)


def _strength_metrics(path: Path, version: str) -> dict[str, float]:
    """Unseen-tuple test scores, only if ``strength evaluate`` ran for this exact version."""
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(report, dict) or report.get("model_version") != version:
        return {}
    scores = report.get("scores", {})
    metrics = {k: float(v) for k, v in scores.items() if isinstance(v, int | float)}
    if isinstance(report.get("joint_coverage"), int | float):
        metrics["interval_coverage"] = float(report["joint_coverage"])
    return metrics


class CrackService:
    """Crack classifier; ONNX Runtime and Grad-CAM are both safe for concurrent requests."""

    def __init__(self, classifier: CrackClassifier, info: CrackModelInfo) -> None:
        self.classifier = classifier
        self.info = info

    @classmethod
    def load(cls, model_dir: Path, config: ApiConfig) -> "CrackService":
        classifier = CrackClassifier.load(
            model_dir, backend=config.cracks_backend, threads=config.inference_threads
        )
        meta = classifier.metadata
        test = meta.metrics.get("test", {})
        info = CrackModelInfo(
            model_name=meta.model_name,
            model_version=meta.model_version,
            architecture=meta.architecture,
            image_size=meta.image_size,
            threshold=classifier.threshold,
            threshold_strategy=meta.threshold_strategy,
            backend=classifier.backend,
            test_metrics={
                k: float(v)
                for k, v in test.items()
                if k in {"precision", "recall", "f1", "accuracy", "roc_auc", "pr_auc"}
            },
        )
        return cls(classifier, info)

    def status(self) -> ComponentStatus:
        return ComponentStatus(
            ready=True,
            detail=f"loaded ({self.info.backend})",
            model_name=self.info.model_name,
            model_version=self.info.model_version,
        )


def load_strength(config: ApiConfig) -> StrengthService | Unavailable:
    model_dir = config.strength_model_dir
    if not (model_dir / "current.json").exists() and not (model_dir / "manifest.json").exists():
        return Unavailable(f"no strength model in {config.artifacts_dir}; {_STRENGTH_HINT}")
    try:
        return StrengthService.load(model_dir, config.strength_metrics_path)
    except (OSError, ValueError) as error:
        logger.warning("strength model not loaded: %s", error)
        return Unavailable(f"strength model failed verification ({error}); {_STRENGTH_HINT}")


def load_cracks(config: ApiConfig) -> CrackService | Unavailable:
    model_dir = config.cracks_model_dir
    if not (model_dir / "current.json").exists():
        return Unavailable(f"no crack model in {config.artifacts_dir}; {_CRACKS_HINT}")
    try:
        return CrackService.load(model_dir, config)
    except (ArtifactError, OSError, ValueError) as error:
        logger.warning("crack model not loaded: %s", error)
        return Unavailable(f"crack model failed verification ({error}); {_CRACKS_HINT}")
