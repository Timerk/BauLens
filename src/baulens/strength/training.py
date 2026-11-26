"""Compare candidates, refit the CV winner, and calibrate without scoring the test set."""

from dataclasses import asdict, dataclass
from importlib.metadata import version as package_version
from pathlib import Path
from time import perf_counter
from typing import cast

import joblib  # type: ignore[import-untyped]
import pandas as pd
from filelock import FileLock
from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]

from baulens.config import StrengthConfig
from baulens.paths import project_path
from baulens.seed import seed_everything
from baulens.strength.artifacts import atomic_file, run_identity, write_json
from baulens.strength.features import RAW_FEATURES, TARGET
from baulens.strength.inference import COVERAGE, INTERVAL_METHOD, MODEL_FILE, FittedStrength
from baulens.strength.metrics import CVSummary, conformal_radius, summarize_cv
from baulens.strength.modeling import (
    FAMILIES,
    Candidate,
    FoldResult,
    cross_validate,
    fit_candidate,
    inner_folds,
    predictions,
)
from baulens.strength.splits import mix_groups, prepare_data, split_partitions


@dataclass(frozen=True)
class TrainResult:
    model_dir: Path
    model_name: str
    model_version: str
    grouped_cv_mae: float


def train(config: StrengthConfig) -> TrainResult:
    root = project_path(config.artifacts_dir)
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / ".strength.lock")), threadpool_limits(limits=config.threads):
        return _train(config, root)


def _train(config: StrengthConfig, root: Path) -> TrainResult:
    started = perf_counter()
    seed_everything(config.seed)
    source = project_path(config.data_path)
    version, checksum = run_identity(config, source)
    frame, duplicates = prepare_data(source)
    groups = mix_groups(frame)
    partitions = split_partitions(frame, groups, config)
    fit = frame.iloc[partitions.fit]
    cal = frame.iloc[partitions.calibration]
    fit_groups = groups.iloc[partitions.fit]
    cal_groups = groups.iloc[partitions.calibration]
    inputs, y = fit.loc[:, list(RAW_FEATURES)], fit[TARGET]

    candidates = [
        Candidate(family, engineered) for family in FAMILIES for engineered in (False, True)
    ]
    grouped, _ = cross_validate(candidates, inputs, y, fit_groups, config, grouped=True)
    summaries = {
        candidate: cv_summary([x for x in grouped if x.candidate == candidate])
        for candidate in candidates
    }
    # MAE is defined on every nonempty validation fold.
    mae_by_candidate = {
        candidate: cast(float, summary["mae"].mean) for candidate, summary in summaries.items()
    }
    winner = min(candidates, key=mae_by_candidate.__getitem__)
    naive, _ = cross_validate([winner], inputs, y, fit_groups, config, grouped=False)
    _, permutation = cross_validate(
        [winner], inputs, y, fit_groups, config, grouped=True, explain=True
    )

    final, params = fit_candidate(
        winner, inputs, y, inner_folds(fit_groups, config, config.seed), config
    )
    cal_point = predictions(final, cal.loc[:, list(RAW_FEATURES)])
    radius = conformal_radius(cal[TARGET].to_numpy() - cal_point, cal_groups, COVERAGE)
    artifact = FittedStrength(
        pipeline=final,
        candidate=winner,
        version=version,
        data_sha256=checksum,
        radius_mpa=radius,
        ranges={
            name: (float(inputs[name].min()), float(inputs[name].max())) for name in RAW_FEATURES
        },
        feature_means=final.named_steps["features"].transform(inputs).mean().to_numpy(),
        partitions=partitions,
    )
    model_dir = root / "model"
    metadata = {
        "model_name": winner.family,
        "model_version": version,
        "features": winner.features,
        "parameters": params,
        "data_sha256": checksum,
        "config": config.model_dump(mode="json"),
        "rows_before_deduplication": len(frame) + duplicates,
        "exact_duplicates_removed": duplicates,
        "strength_bins": partitions.strength_bins,
        "partitions": {
            label: {"rows": len(index), "mixes": int(groups.iloc[index].nunique())}
            for label, index in (
                ("fit", partitions.fit),
                ("calibration", partitions.calibration),
                ("test", partitions.test),
            )
        },
        "interval_coverage": COVERAGE,
        "interval_method": INTERVAL_METHOD,
        "radius_mpa": radius,
        "training_ranges": artifact.ranges,
        "package_versions": {
            name: package_version(name)
            for name in ("scikit-learn", "lightgbm", "pandas", "numpy", "shap", "joblib")
        },
        "training_seconds": perf_counter() - started,
    }
    evidence = {
        "model_version": version,
        "grouped": [
            {
                "model": c.family,
                "features": c.features,
                "metrics": {metric: asdict(summary) for metric, summary in summaries[c].items()},
            }
            for c in candidates
        ],
        "row_cv": {metric: asdict(summary) for metric, summary in cv_summary(naive).items()},
        "row_validation_overlap_fraction": (
            sum(x.overlapping_validation_rows for x in naive)
            / sum(x.validation_rows for x in naive)
        ),
        "permutation_mae_mpa": permutation,
    }
    write_json(root / "cv_summary.json", evidence)
    with atomic_file(root / "cv_folds.csv") as staged:
        pd.DataFrame([_fold_record(x) for x in grouped + naive]).to_csv(staged, index=False)
    write_json(model_dir / "metadata.json", metadata)
    with atomic_file(model_dir / MODEL_FILE) as staged:
        joblib.dump(artifact, staged, compress=3)
    return TrainResult(model_dir, winner.family, version, mae_by_candidate[winner])


def cv_summary(folds: list[FoldResult]) -> dict[str, CVSummary]:
    repeats = [fold.repeat for fold in folds]
    return {
        name: summarize_cv([getattr(fold.scores, name) for fold in folds], repeats)
        for name in ("mae", "rmse", "r2")
    }


def _fold_record(fold: FoldResult) -> dict[str, object]:
    return {
        "model": fold.candidate.family,
        "features": fold.candidate.features,
        "protocol": fold.protocol,
        "repeat": fold.repeat,
        "fold": fold.fold,
        "overlapping_validation_rows": fold.overlapping_validation_rows,
        "validation_rows": fold.validation_rows,
        **asdict(fold.scores),
        "parameters": repr(fold.parameters),
    }
