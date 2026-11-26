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
from baulens.strength.artifacts import (
    CURRENT_PATH,
    CV_SUMMARY_PATH,
    FIT_FILES,
    METADATA_PATH,
    MODEL_PATH,
    atomic_file,
    read_fit_files,
    run_identity,
    write_json,
)
from baulens.strength.diagnostics import (
    error_summary,
    plot_residuals,
    plot_shap,
    slice_errors,
    worst_mixes,
)
from baulens.strength.features import RAW_FEATURES, TARGET
from baulens.strength.inference import FittedStrength, StrengthPredictor
from baulens.strength.metrics import Scores, bootstrap_metrics, score_predictions
from baulens.strength.modeling import predictions
from baulens.strength.reporting import EvaluationReport, render_results
from baulens.strength.splits import mix_groups, near_fitting_mix, prepare_data


@dataclass(frozen=True)
class EvaluationResult:
    model_name: str
    model_version: str
    scores: Scores
    coverage: float
    results_path: Path


def evaluate(config: StrengthConfig) -> EvaluationResult:
    root = project_path(config.artifacts_dir)
    model_dir = root / "model"
    if not ((model_dir / CURRENT_PATH).is_file() or (model_dir / MODEL_PATH).is_file()):
        raise FileNotFoundError(f"Run strength train before evaluation. Expected {model_dir}")
    with FileLock(str(root / ".strength.lock")), threadpool_limits(limits=config.threads):
        return _evaluate(config, root)


def _evaluate(config: StrengthConfig, root: Path) -> EvaluationResult:
    files = read_fit_files(root / "model", FIT_FILES)
    artifact = FittedStrength.from_bytes(files[MODEL_PATH])
    source = project_path(config.data_path)
    version, checksum = run_identity(config, source)
    metadata = json.loads(files[METADATA_PATH])
    cv = json.loads(files[CV_SUMMARY_PATH])
    if version != artifact.version or checksum != artifact.data_sha256:
        raise ValueError("This configuration or data differs from the fitted artifact")
    fitted_id = sha256(files[MODEL_PATH]).hexdigest()
    if any(
        sidecar.get("model_version") != version
        or sidecar.get("fitted_artifact_sha256") != fitted_id
        for sidecar in (metadata, cv)
    ):
        raise ValueError("The model and training records differ. Run strength train again")

    frame, _ = prepare_data(source)
    groups = mix_groups(frame)
    fit = frame.iloc[artifact.partitions.fit]
    test = frame.iloc[artifact.partitions.test].assign(
        mix_id=groups, strength_stratum=artifact.partitions.test_strata
    )
    test = test.assign(
        near_fitting_mix=near_fitting_mix(fit, test), fitted_artifact_sha256=fitted_id
    )
    test = _holdout_predictions(test, artifact, root / "holdout" / f"{fitted_id}.csv")
    actual, point = test[TARGET].to_numpy(), test["predicted_mpa"].to_numpy()
    scores = score_predictions(actual, point)
    confidence = bootstrap_metrics(
        actual,
        point,
        test["mix_id"],
        test["strength_stratum"],
        config.bootstrap_resamples,
        config.seed,
    )
    overall = error_summary(test, "Overall")
    assert overall.coverage is not None
    covered_mixes = int(test.groupby("mix_id")["covered"].all().sum())
    stratum_counts = test.groupby("strength_stratum")["mix_id"].nunique().to_list()
    near_mix_errors = [
        error_summary(test[test["near_fitting_mix"] == nearby], label)
        for nearby, label in (
            (True, "With a nearby fitting tuple"),
            (False, "Without a nearby fitting tuple"),
        )
    ]
    age_errors, strength_errors = slice_errors(test)

    figure_dir = project_path(config.figures_dir)
    plot_residuals(test, figure_dir)
    calibration = frame.iloc[artifact.partitions.calibration]
    shap = plot_shap(StrengthPredictor(artifact), calibration, figure_dir, config.seed)
    report = EvaluationReport(
        scores=scores,
        confidence=confidence,
        overall=overall,
        age_errors=age_errors,
        strength_errors=strength_errors,
        covered_rows=int(test["covered"].sum()),
        covered_mixes=covered_mixes,
        near_mix_errors=near_mix_errors,
        stratum_counts=stratum_counts,
        worst=worst_mixes(test),
        shap=shap,
        resamples=config.bootstrap_resamples,
    )
    write_json(
        root / "test_metrics.json",
        {
            "model_version": version,
            "fitted_artifact_sha256": fitted_id,
            "joint_coverage": report.joint_coverage,
            **asdict(report),
        },
    )
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
