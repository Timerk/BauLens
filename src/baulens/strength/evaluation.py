"""Score only the frozen CV winner on the holdout and record the experiment."""

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from filelock import FileLock
from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]

from baulens.config import StrengthConfig
from baulens.paths import project_path
from baulens.strength.artifacts import atomic_file, run_identity, write_json
from baulens.strength.diagnostics import (
    error_summary,
    plot_residuals,
    plot_shap,
    slice_errors,
    worst_mixes,
)
from baulens.strength.features import RAW_FEATURES, TARGET
from baulens.strength.inference import MODEL_FILE, FittedStrength, StrengthPredictor
from baulens.strength.metrics import Scores, bootstrap_metrics, score_predictions
from baulens.strength.modeling import predictions
from baulens.strength.reporting import EvaluationReport, render_results
from baulens.strength.splits import mix_groups, prepare_data


@dataclass(frozen=True)
class EvaluationResult:
    model_name: str
    model_version: str
    scores: Scores
    coverage: float
    results_path: Path


def evaluate(config: StrengthConfig) -> EvaluationResult:
    root = project_path(config.artifacts_dir)
    if not (root / "model" / MODEL_FILE).is_file():
        raise FileNotFoundError(f"Run strength train before evaluation. Expected {root / 'model'}")
    with FileLock(str(root / ".strength.lock")), threadpool_limits(limits=config.threads):
        return _evaluate(config, root)


def _evaluate(config: StrengthConfig, root: Path) -> EvaluationResult:
    model_dir = root / "model"
    artifact = FittedStrength.load(model_dir)
    source = project_path(config.data_path)
    version, checksum = run_identity(config, source)
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    cv = json.loads((root / "cv_summary.json").read_text(encoding="utf-8"))
    if version != artifact.version or checksum != artifact.data_sha256:
        raise ValueError("This configuration or data differs from the fitted artifact")
    if metadata["model_version"] != version or cv["model_version"] != version:
        raise ValueError("The model and training records differ. Run strength train again")

    frame, _ = prepare_data(source)
    groups = mix_groups(frame)
    test = frame.iloc[artifact.partitions.test].assign(mix_id=groups)
    artifact_checksum = sha256((model_dir / MODEL_FILE).read_bytes()).hexdigest()
    test = _holdout_predictions(test, artifact, root / "holdout" / f"{artifact_checksum}.csv")
    actual, point = test[TARGET].to_numpy(), test["predicted_mpa"].to_numpy()
    scores = score_predictions(actual, point)
    confidence = bootstrap_metrics(
        actual, point, test["mix_id"], config.bootstrap_resamples, config.seed
    )
    overall = error_summary(test, "Overall")
    assert overall.coverage is not None
    joint_coverage = float(test.groupby("mix_id")["covered"].all().mean())
    age_errors, strength_errors = slice_errors(test)

    figure_dir = project_path(config.figures_dir)
    plot_residuals(test, figure_dir)
    calibration = frame.iloc[artifact.partitions.calibration]
    shap = plot_shap(StrengthPredictor(artifact), calibration, figure_dir, config.seed)
    report = EvaluationReport(
        scores,
        confidence,
        overall,
        age_errors,
        strength_errors,
        joint_coverage,
        worst_mixes(test),
        shap,
        config.bootstrap_resamples,
    )
    write_json(root / "test_metrics.json", {"model_version": version, **asdict(report)})
    with atomic_file(root / "test_predictions.csv") as staged:
        test.to_csv(staged, index=False)
    results_path = project_path(config.results_path)
    with atomic_file(results_path) as staged:
        staged.write_text(
            render_results(report, metadata, cv, results_path, figure_dir), encoding="utf-8"
        )
    return EvaluationResult(
        artifact.candidate.family, version, scores, overall.coverage, results_path
    )


def _holdout_predictions(test: pd.DataFrame, artifact: FittedStrength, cache: Path) -> pd.DataFrame:
    """Reuse scores for the identical model so regenerating a report never refits or rescores it."""
    if cache.is_file():
        point = np.loadtxt(cache, delimiter=",", skiprows=1, ndmin=1)
    else:
        point = predictions(artifact.pipeline, test.loc[:, list(RAW_FEATURES)])
        with atomic_file(cache) as staged:
            pd.DataFrame({"predicted_mpa": point}).to_csv(staged, index=False)
    lower = point - artifact.radius_mpa
    upper = point + artifact.radius_mpa
    return test.assign(
        predicted_mpa=point,
        lower_mpa=lower,
        upper_mpa=upper,
        residual_mpa=test[TARGET] - point,
        covered=test[TARGET].between(lower, upper),
    )
