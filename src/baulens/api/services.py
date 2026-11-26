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
        parsed = json.loads(files[METADATA_PATH])
        metadata: dict[str, Any] = parsed if isinstance(parsed, dict) else {}
        predictor = StrengthPredictor(artifact)
        info = StrengthModelInfo(
            model_name=artifact.candidate.family,
            model_version=artifact.version,
            interval_coverage=float(metadata.get("interval_coverage", 0.9)),
            interval_method=str(metadata.get("interval_method", "split conformal")),
            interval_half_width_mpa=artifact.radius_mpa,
            training_rows=_training_rows(metadata),
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


def _training_rows(metadata: dict[str, Any]) -> int | None:
    fit = metadata.get("partitions")
    fit = fit.get("fit") if isinstance(fit, dict) else None
    rows = fit.get("rows") if isinstance(fit, dict) else None
    return rows if isinstance(rows, int) else None


def _numbers(values: object, allowed: set[str] | None = None) -> dict[str, float]:
    """Numeric entries of an optional metrics mapping; anything malformed is dropped."""
    if not isinstance(values, dict):
        return {}
    return {
        str(key): float(value)
        for key, value in values.items()
        if isinstance(value, int | float)
        and not isinstance(value, bool)
        and (allowed is None or key in allowed)
    }


def _strength_metrics(path: Path, version: str) -> dict[str, float]:
    """Unseen-tuple test scores, only if ``strength evaluate`` ran for this exact version.

    The file is optional: unreadable, malformed or mismatched content counts as absent.
    """
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(report, dict) or report.get("model_version") != version:
        return {}
    metrics = _numbers(report.get("scores"), {"mae", "rmse", "r2"})
    coverage = _numbers({"interval_coverage": report.get("joint_coverage")})
    return metrics | coverage


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
        test = meta.metrics.get("test")
        info = CrackModelInfo(
            model_name=meta.model_name,
            model_version=meta.model_version,
            architecture=meta.architecture,
            image_size=meta.image_size,
            threshold=classifier.threshold,
            threshold_strategy=meta.threshold_strategy,
            backend=classifier.backend,
            test_metrics=_numbers(
                test, {"precision", "recall", "f1", "accuracy", "roc_auc", "pr_auc"}
            ),
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
    """Load the strength model; public details never contain paths or exception text."""
    model_dir = config.strength_model_dir
    if not (model_dir / "current.json").exists() and not (model_dir / "manifest.json").exists():
        logger.warning("no strength model at %s", model_dir)
        return Unavailable(f"no strength model published; {_STRENGTH_HINT}")
    try:
        return StrengthService.load(model_dir, config.strength_metrics_path)
    except Exception:
        # Startup must survive any broken artifact; the cause goes to the server log only.
        logger.exception("strength model at %s not loaded", model_dir)
        return Unavailable(f"strength model could not be loaded (see server log); {_STRENGTH_HINT}")


def load_cracks(config: ApiConfig) -> CrackService | Unavailable:
    """Load the crack model; public details never contain paths or exception text."""
    model_dir = config.cracks_model_dir
    if not (model_dir / "current.json").exists():
        logger.warning("no crack model at %s", model_dir)
        return Unavailable(f"no crack model published; {_CRACKS_HINT}")
    try:
        return CrackService.load(model_dir, config)
    except Exception:
        logger.exception("crack model at %s not loaded", model_dir)
        return Unavailable(f"crack model could not be loaded (see server log); {_CRACKS_HINT}")
