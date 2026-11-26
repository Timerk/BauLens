"""Render the frozen run's tables and link the generated figures."""

from dataclasses import dataclass
from os.path import relpath
from pathlib import Path
from typing import TypedDict

from baulens.strength.diagnostics import (
    DEPENDENCE_FIGURE,
    RESIDUAL_FIGURE,
    SHAP_FIGURE,
    ErrorSlice,
    ShapSummary,
)
from baulens.strength.features import FEATURE_RATIONALES, TARGET
from baulens.strength.metrics import Scores

METRICS = ("mae", "rmse", "r2")


class _Counts(TypedDict):
    rows: int
    mixes: int


class ModelMetadata(TypedDict):
    model_name: str
    model_version: str
    features: str
    rows_before_deduplication: int
    exact_duplicates_removed: int
    strength_bins: int
    partitions: dict[str, _Counts]
    radius_mpa: float
    training_seconds: float


class _CVMetric[MeanT](TypedDict):
    mean: MeanT
    std: float | None
    ci: list[float] | None


class _CVMetrics(TypedDict):
    mae: _CVMetric[float]
    rmse: _CVMetric[float]
    r2: _CVMetric[float | None]


class _CandidateSummary(TypedDict):
    model: str
    features: str
    metrics: _CVMetrics


class CVEvidence(TypedDict):
    grouped: list[_CandidateSummary]
    row_cv: _CVMetrics
    row_validation_overlap_fraction: float
    permutation_mae_mpa: dict[str, float]


@dataclass(frozen=True)
class EvaluationReport:
    scores: Scores
    confidence: dict[str, tuple[float | None, float | None]]
    overall: ErrorSlice
    age_errors: list[ErrorSlice]
    strength_errors: list[ErrorSlice]
    joint_coverage: float
    worst: list[dict[str, float]]
    shap: list[ShapSummary]
    resamples: int


def render_results(
    report: EvaluationReport,
    metadata: ModelMetadata,
    cv: CVEvidence,
    results_path: Path,
    figures_dir: Path,
) -> str:
    figures = Path(relpath(figures_dir, results_path.parent)).as_posix()
    model = f"{metadata['model_name']} ({metadata['features'].lower()})"
    title = (
        "# Concrete compressive strength\n\n"
        f"**{model}** (`{metadata['model_version']}`) achieved **{report.scores.mae:.2f} MPa MAE** "
        f"on {report.overall.mixes} unseen mix designs. The 90% prediction intervals covered "
        f"{_fmt(report.overall.coverage, '.1%')} of measurements and "
        f"{report.joint_coverage:.1%} of entire mixes.\n\n"
        "## Data and protocol\n\n"
        f"This run read {metadata['rows_before_deduplication']} measurements from the "
        "[UCI concrete strength data](https://doi.org/10.24432/C5PK67). It removed "
        f"{metadata['exact_duplicates_removed']} exact duplicate rows and kept distinct outcomes "
        "at the same mix and age. A mix is the exact tuple of its seven ingredients in kg/m³. "
        f"The test split used {metadata['strength_bins']} strata of mean strength per mix. "
        "Calibration mixes were randomly drawn from the remainder.\n\n"
        "| Partition | Rows | Mixes |\n| --- | ---: | ---: |\n"
        + "\n".join(
            f"| {name.title()} | {counts['rows']} | {counts['mixes']} |"
            for name, counts in metadata["partitions"].items()
        )
        + "\n\n"
        "All candidates used the same repeated shuffled GroupKFold outer splits. Tuned "
        "hyperparameters and final refit settings came from grouped inner CV, scored by MAE. "
        "Selection used the lowest mean grouped outer MAE. The fitting partition includes the "
        "outer validation folds and supplies the final refit. The refit then predicts the "
        "separate calibration partition for interval construction. See the "
        "[evaluation ADR](../adr/0003-strength-evaluation-protocol.md).\n"
    )
    features = (
        "## Engineered features\n\n"
        "Binder is cement plus blast-furnace slag and fly ash. SCM is slag plus fly ash.\n\n"
        "| Feature | Rationale |\n| --- | --- |\n"
        + "\n".join(f"| `{name}` | {rationale} |" for name, rationale in FEATURE_RATIONALES.items())
        + "\n"
    )
    reproduction = (
        "## Limitations and reproduction\n\n"
        "This small historical benchmark is based on formulations published by Yeh in 1998. "
        "The workbook omits lab and batch identifiers, cement type, and curing conditions. "
        "Grouping detects exact formulations. Rounded variants or related production batches may "
        "still span partitions. A new lab, material source, or curing regime may change accuracy "
        "and coverage. Input ranges detect marginal extrapolation. In-range ingredients can still "
        "form an unfamiliar mix. The [model card](../model_cards/strength.md) describes "
        "intended use.\n\n"
        "```sh\n"
        "uv run --locked baulens strength train --config configs/strength.yaml\n"
        "uv run --locked baulens strength evaluate --config configs/strength.yaml\n"
        "```\n\n"
        f"Training took {metadata['training_seconds']:.1f} s on a CPU under shared load. "
        "The local `artifacts/strength/` directory contains every fold score, test prediction, "
        "and a machine-readable copy of the metrics. Evaluation reuses cached predictions for "
        "the identical fitted model.\n"
    )
    return "\n".join(
        (
            title,
            features,
            _cv_tables(cv, metadata),
            _test_tables(report, metadata, figures),
            _explainability(report, cv, figures),
            reproduction,
        )
    )


def _cv_tables(cv: CVEvidence, metadata: ModelMetadata) -> str:
    rows = cv["grouped"]
    winner = next(
        row
        for row in rows
        if row["model"] == metadata["model_name"] and row["features"] == metadata["features"]
    )
    row_mae = cv["row_cv"]["mae"]["mean"]
    grouped_mae = winner["metrics"]["mae"]["mean"]
    return "\n".join(
        [
            "## Repeated cross-validation and feature ablation",
            "",
            "Cells show fold mean ± sample SD and a 95% Student t interval in brackets. "
            "Intervals use repeat means as the observations because folds share training rows. "
            "Repeats share the same underlying mixes. The full run uses three repeats. "
            "The intervals describe split sensitivity and offer limited evidence about differences "
            "between candidates. The single-repeat smoke run omits them.",
            "",
            "| Family | Inputs | MAE (MPa) | RMSE (MPa) | R² |",
            "| --- | --- | ---: | ---: | ---: |",
            *(
                f"| {row['model']} | {row['features']} | " + _cv_cells(row["metrics"]) + " |"
                for row in rows
            ),
            "",
            "Engineered minus raw MAE (MPa, lower is better) for each family.",
            "",
            "| Mean | Linear | Ridge | Random forest | Histogram GBM | LightGBM |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
            "| "
            + " | ".join(
                f"{engineered['metrics']['mae']['mean'] - raw['metrics']['mae']['mean']:+.2f}"
                for raw, engineered in zip(rows[::2], rows[1::2], strict=True)
            )
            + " |",
            "",
            "### What a shuffled row split would report",
            "",
            "The comparison uses the selected family and feature set, the same development rows, "
            "fold counts, seeds and tuning grid, and row splits at both CV levels.",
            "",
            "| Protocol | MAE (MPa) | RMSE (MPa) | R² |",
            "| --- | ---: | ---: | ---: |",
            f"| Grouped | {_cv_cells(winner['metrics'])} |",
            f"| Random row | {_cv_cells(cv['row_cv'])} |",
            "",
            f"In random-row CV, {cv['row_validation_overlap_fraction']:.1%} of validation rows "
            "shared their mix design with a fitting row. "
            f"Grouped minus random-row MAE was {grouped_mae - row_mae:+.2f} MPa. "
            "The grouped score estimates error for an unseen formulation.",
            "",
        ]
    )


def _cv_cells(metrics: _CVMetrics) -> str:
    cells = []
    for metric in (metrics["mae"], metrics["rmse"], metrics["r2"]):
        interval = metric["ci"]
        ci = f" [{interval[0]:.2f}, {interval[1]:.2f}]" if interval else ""
        cells.append(f"{_fmt(metric['mean'], '.2f')} ± {_fmt(metric['std'], '.2f')}{ci}")
    return " | ".join(cells)


def _test_tables(report: EvaluationReport, metadata: ModelMetadata, figures: str) -> str:
    hardest_age = max(report.age_errors, key=lambda bucket: bucket.mae or 0.0)
    hardest_strength = max(report.strength_errors, key=lambda bucket: bucket.mae or 0.0)
    return "\n".join(
        [
            "## Held-out mix designs",
            "",
            f"95% percentile intervals use {report.resamples:,} bootstrap replicates. Each samples "
            "complete test mixes with replacement and weights all measurements equally. The "
            "bootstrap describes holdout uncertainty for the frozen fit.",
            "",
            "| Metric | Estimate | 95% group bootstrap CI |",
            "| --- | ---: | ---: |",
            *(
                f"| {label} | {_fmt(getattr(report.scores, name), '.3f')} | "
                f"[{_fmt(report.confidence[name][0], '.3f')}, "
                f"{_fmt(report.confidence[name][1], '.3f')}] |"
                for name, label in zip(METRICS, ("MAE (MPa)", "RMSE (MPa)", "R²"), strict=True)
            ),
            "",
            f"![Residuals versus predicted strength on unseen mixes]({figures}/{RESIDUAL_FIGURE})",
            "",
            "### Error and coverage by age",
            "",
            "Split conformal uses one maximum absolute residual per calibration mix. The "
            f"finite-sample order statistic gave a radius of {metadata['radius_mpa']:.2f} MPa. "
            "The same radius applies "
            "to each prediction. With exchangeable mixes, this construction targets joint coverage "
            "across the measured ages of a new mix. Strength stratification and source shift can "
            "affect exchangeability. Empirical bucket coverage can vary in small samples. "
            f"Every measurement was covered in {report.joint_coverage:.1%} of test mixes. "
            "Positive bias means the model underpredicted the observed strength.",
            "",
            "| Age (days) | Rows | Mixes | MAE (MPa) | Bias (MPa) | Coverage | Mean width (MPa) |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            *(
                _slice_row(bucket, intervals=True)
                for bucket in [report.overall, *report.age_errors]
            ),
            "",
            f"The {hardest_age.label} day bucket had the highest MAE "
            f"({_fmt(hardest_age.mae, '.2f')} MPa) across {hardest_age.mixes} mixes.",
            "",
            "### Error by observed strength",
            "",
            "| Strength (MPa) | Rows | Mixes | MAE (MPa) | Bias (MPa) |",
            "| --- | ---: | ---: | ---: | ---: |",
            *(_slice_row(bucket, intervals=False) for bucket in report.strength_errors),
            "",
            f"Error was highest in the {hardest_strength.label} MPa range "
            f"({_fmt(hardest_strength.mae, '.2f')} MPa MAE). Its mean observed-minus-predicted "
            f"residual was {_fmt(hardest_strength.bias, '+.2f')} MPa across "
            f"{hardest_strength.mixes} mixes.",
            "",
            "### Worst-predicted mixes",
            "",
            "Ranked by largest absolute residual at any recorded age. The table shows the worst "
            "measurement per mix. Mix IDs identify the exact tuple in `test_predictions.csv`.",
            "",
            "| Mix ID | Rows | Age (days) | Water/cement | Observed (MPa) | "
            "Predicted (MPa) | Residual (MPa) |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            *(
                f"| {row['mix_id']:.0f} | {row['measurements']:.0f} | {row['age_days']:.0f} | "
                f"{row['water_cement']:.2f} | {row[TARGET]:.2f} | {row['predicted_mpa']:.2f} | "
                f"{row['residual_mpa']:+.2f} |"
                for row in report.worst
            ),
            "",
        ]
    )


def _explainability(report: EvaluationReport, cv: CVEvidence, figures: str) -> str:
    top = report.shap[0]
    shap_by_name = {row.feature: row.rank_correlation for row in report.shap}
    trends = []
    for feature in ("age_days", "log_age", "water_cement", "water_binder"):
        rho = shap_by_name.get(feature)
        if rho is not None:
            expectation = 1 if feature in {"age_days", "log_age"} else -1
            relationship = "agrees" if rho * expectation > 0 else "disagrees"
            trends.append(f"`{feature}` {relationship} with the expected direction ({rho:+.2f})")
    return "\n".join(
        [
            "## Explainability",
            "",
            "SHAP values use the refitted model on calibration mixes. The summary ranks features "
            "by mean absolute contribution. Dependence plots show fitted associations in MPa.",
            "",
            f"![SHAP contribution summary]({figures}/{SHAP_FIGURE})",
            "",
            f"![Dependence of SHAP contributions on feature values]({figures}/{DEPENDENCE_FIGURE})",
            "",
            f"The strongest attribution on calibration mixes was `{top.feature}` "
            f"({top.magnitude_mpa:.2f} MPa mean absolute SHAP).",
            "",
            "Hydration generally raises strength with age. Higher water ratios generally reduce "
            "strength at similar composition and curing. "
            + (". ".join(trends) + ". " if trends else "")
            + "These are rank correlations between feature values and their own attributions. "
            "The model can share attribution among correlated raw and engineered inputs. The plots "
            "describe associations within this dataset and cannot establish causal effects.",
            "",
            "Permutation importance measures the mean increase in MAE on the grouped validation "
            "folds for the chosen pipeline. Each shuffle permutes a raw column before recomputing "
            "derived ratios. Shuffles can create uncommon ingredient combinations.",
            "",
            "| Raw input | Increase in MAE (MPa) |",
            "| --- | ---: |",
            *(
                f"| `{name}` | {change:+.2f} |"
                for name, change in sorted(
                    cv["permutation_mae_mpa"].items(), key=lambda row: row[1], reverse=True
                )
            ),
            "",
        ]
    )


def _slice_row(bucket: ErrorSlice, *, intervals: bool) -> str:
    cells = [
        bucket.label,
        str(bucket.rows),
        str(bucket.mixes),
        _fmt(bucket.mae, ".2f"),
        _fmt(bucket.bias, "+.2f"),
    ]
    if intervals:
        cells.extend((_fmt(bucket.coverage, ".1%"), _fmt(bucket.width, ".2f")))
    return "| " + " | ".join(cells) + " |"


def _fmt(value: float | None, spec: str) -> str:
    return format(value, spec) if value is not None else "—"
