"""Per-model metrics of a comparison run, the Markdown table and the summary figure."""

import statistics
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict

from baulens.report.monitor import LoadSummary
from baulens.report.schemas import VIOLATION_KINDS, ReportResult, ViolationKind

TEMPLATE_ROW = "template (no LLM)"


class ModelMetrics(BaseModel):
    """Metric definitions are documented in ``docs/results/report.md``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str
    cases: int
    schema_valid_first: float
    schema_valid: float
    faithful_initial: float
    faithful_final: float
    rejected: int
    invalid: int
    errors: int
    violations_per_report: dict[ViolationKind, float]
    completeness_mean: float | None
    completeness_accepted_mean: float | None
    latency_median_s: float | None
    latency_p95_s: float | None
    calls_per_case: float
    output_tokens_mean: float | None
    tokens_per_s: float | None
    load_time_s: float | None = None
    loaded_size_bytes: int | None = None
    load: LoadSummary | None = None


def _rate(values: Sequence[bool]) -> float:
    return sum(values) / len(values) if values else 0.0


def compute_metrics(
    model: str,
    results: Sequence[ReportResult],
    *,
    load: LoadSummary | None = None,
    load_time_s: float | None = None,
    loaded_size_bytes: int | None = None,
) -> ModelMetrics:
    first_valid = [
        bool(result.attempts) and result.attempts[0].schema_error is None for result in results
    ]
    if results and all(result.source == "template" for result in results):
        first_valid = [True] * len(results)
    initial = [result.initial_faithfulness for result in results]
    drafted = [faith for faith in initial if faith is not None]
    violations = {
        kind: (
            statistics.fmean(faith.violation_counts()[kind] for faith in drafted)
            if drafted
            else 0.0
        )
        for kind in VIOLATION_KINDS
    }
    scores = [result.completeness.score for result in results if result.completeness is not None]
    accepted_scores = [
        result.completeness.score
        for result in results
        if result.completeness is not None and result.status in {"accepted", "template"}
    ]
    latencies = [result.latency_s for result in results]
    responses = [attempt.response for result in results for attempt in result.attempts]
    eval_counts = [response.eval_count for response in responses if response.eval_count]
    durations = [
        (response.eval_count, response.eval_duration_ns)
        for response in responses
        if response.eval_count and response.eval_duration_ns
    ]
    total_tokens = sum(count for count, _ in durations)
    total_seconds = sum(duration for _, duration in durations) / 1e9
    return ModelMetrics(
        model=model,
        cases=len(results),
        schema_valid_first=_rate(first_valid),
        schema_valid=_rate([result.candidate is not None for result in results]),
        faithful_initial=_rate([faith is not None and faith.faithful for faith in initial]),
        faithful_final=_rate([result.status in {"accepted", "template"} for result in results]),
        rejected=sum(result.status == "rejected" for result in results),
        invalid=sum(result.status == "invalid" for result in results),
        errors=sum(result.status == "error" for result in results),
        violations_per_report=violations,
        completeness_mean=statistics.fmean(scores) if scores else None,
        completeness_accepted_mean=statistics.fmean(accepted_scores) if accepted_scores else None,
        latency_median_s=float(np.median(latencies)) if latencies else None,
        latency_p95_s=float(np.percentile(latencies, 95)) if latencies else None,
        calls_per_case=len(responses) / len(results) if results else 0.0,
        output_tokens_mean=statistics.fmean(eval_counts) if eval_counts else None,
        tokens_per_s=total_tokens / total_seconds if total_seconds else None,
        load_time_s=load_time_s,
        loaded_size_bytes=loaded_size_bytes,
        load=load,
    )


def _percent(value: float) -> str:
    return f"{100 * value:.0f} %"


def _number(value: float | None, digits: int = 1, suffix: str = "") -> str:
    return "-" if value is None else f"{value:.{digits}f}{suffix}"


def _gib(value: int | None) -> str:
    return "-" if value is None else f"{value / 2**30:.2f}"


def metrics_table(rows: Sequence[ModelMetrics]) -> str:
    """Two Markdown tables: quality and resources."""
    quality = [
        "| Model | Cases | Schema-valid (1st try / final) | Faithful before retry | "
        "Faithful after retry | Rejected / invalid / error | Violations per draft "
        "(invented / misattributed / rounding) | Completeness (all drafts / delivered) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        v = row.violations_per_report
        quality.append(
            f"| {row.model} | {row.cases} | {_percent(row.schema_valid_first)} / "
            f"{_percent(row.schema_valid)} | {_percent(row.faithful_initial)} | "
            f"{_percent(row.faithful_final)} | {row.rejected} / {row.invalid} / {row.errors} | "
            f"{v['invented']:.2f} / {v['misattributed']:.2f} / {v['bad_rounding']:.2f} | "
            f"{_number(row.completeness_mean, 2)} / {_number(row.completeness_accepted_mean, 2)} |"
        )
    resources = [
        "| Model | Latency per case, median / p95 (s) | Calls per case | Output tokens per call "
        "| Tokens/s | Load time (s) | Ollama model size (GiB) | Peak runner RSS (GiB) "
        "| System CPU mean / max |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        load = row.load
        cpu = (
            f"{_number(load.cpu_mean_percent, 0, ' %')} / {_number(load.cpu_max_percent, 0, ' %')}"
            if load is not None
            else "-"
        )
        rss = _gib(load.ollama_rss_peak_bytes) if load is not None else "-"
        digits = 3 if row.model == TEMPLATE_ROW else 1
        resources.append(
            f"| {row.model} | {_number(row.latency_median_s, digits)} / "
            f"{_number(row.latency_p95_s, digits)} | {row.calls_per_case:.2f} | "
            f"{_number(row.output_tokens_mean, 0)} | {_number(row.tokens_per_s, 1)} | "
            f"{_number(row.load_time_s, 1)} | {_gib(row.loaded_size_bytes)} | {rss} | {cpu} |"
        )
    return "\n".join(quality) + "\n\n" + "\n".join(resources) + "\n"


# Reference categorical palette (dataviz skill), fixed order; neutral gray for the baseline.
_SERIES = ("#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac")
_BASELINE = "#8c8c8c"
_TEXT = "#191919"
_MUTED = "#666666"
_GRID = "#e6e6e6"


def save_figure(rows: Sequence[ModelMetrics], path: Path, *, subtitle: str) -> Path:
    """Rates per model (left) and latency (right); two panels, never a shared dual axis."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    llm_rows = [row for row in rows if row.model != TEMPLATE_ROW]
    template = next((row for row in rows if row.model == TEMPLATE_ROW), None)
    metrics: list[tuple[str, Callable[[ModelMetrics], float]]] = [
        ("Schema-valid", lambda row: row.schema_valid),
        ("Faithful,\nbefore retry", lambda row: row.faithful_initial),
        ("Faithful,\nafter retry", lambda row: row.faithful_final),
        ("Completeness\n(mean)", lambda row: row.completeness_mean or 0.0),
    ]
    figure, (left, right) = plt.subplots(
        1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [2.4, 1]}
    )
    width = 0.8 / max(len(llm_rows), 1)
    positions = np.arange(len(metrics))
    for index, row in enumerate(llm_rows):
        values = [100 * getter(row) for _, getter in metrics]
        offsets = positions - 0.4 + width * (index + 0.5)
        color = _SERIES[index % len(_SERIES)]
        left.bar(offsets, values, width * 0.9, color=color, label=row.model, zorder=2)
        for x, value in zip(offsets, values, strict=True):
            left.text(x, value + 1.5, f"{value:.0f}", ha="center", fontsize=8, color=_TEXT)
    if template is not None:
        left.axhline(
            100,
            color=_BASELINE,
            linewidth=1,
            linestyle="--",
            zorder=1,
            label="template baseline (100 %)",
        )
    left.set_xticks(positions, [label for label, _ in metrics], fontsize=9, color=_TEXT)
    left.set_ylim(0, 112)
    left.set_ylabel("% of cases (completeness: mean score x 100)", fontsize=9, color=_MUTED)
    left.set_title("Report quality per model", fontsize=11, color=_TEXT, loc="left")
    left.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0, -0.14), ncol=4)

    names = [row.model for row in llm_rows]
    medians = [row.latency_median_s or 0.0 for row in llm_rows]
    p95 = [row.latency_p95_s or 0.0 for row in llm_rows]
    ys = np.arange(len(llm_rows))
    for index, (y, median, high) in enumerate(zip(ys, medians, p95, strict=True)):
        color = _SERIES[index % len(_SERIES)]
        right.plot([median, high], [y, y], color=color, linewidth=2, zorder=2)
        right.plot(median, y, "o", color=color, markersize=8, zorder=3)
        right.plot(high, y, "|", color=color, markersize=12, markeredgewidth=2, zorder=3)
        label = f"median {median:.0f} s, p95 {high:.0f} s"
        right.text(high * 1.03, y, label, fontsize=8, color=_TEXT, va="center")
    right.set_yticks(ys, names, fontsize=9, color=_TEXT)
    right.set_ylim(-0.6, len(llm_rows) - 0.3)
    right.set_xlim(0, max(p95, default=1) * 1.75)
    right.invert_yaxis()
    right.set_xlabel("seconds per case (all calls)", fontsize=9, color=_MUTED)
    right.set_title("Latency per case (CPU)", fontsize=11, color=_TEXT, loc="left")

    for axis in (left, right):
        axis.grid(axis="y" if axis is left else "x", color=_GRID, linewidth=0.8, zorder=0)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(_GRID)
        axis.tick_params(colors=_MUTED, labelsize=8)
    figure.text(0.01, 0.015, subtitle, fontsize=8, color=_MUTED, ha="left")
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, facecolor="white")
    plt.close(figure)
    return path
