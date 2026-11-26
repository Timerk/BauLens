"""Run every evaluation case through every configured model and summarise the results.

One command reproduces the committed numbers: ``baulens report compare``. Raw results go to
``artifacts/report/runs/<timestamp>/``; the replay files, the figure and the generated block
in ``docs/results/report.md`` are rewritten in place.

Each model's results and metrics are written as soon as that model finishes, so a crash of
the model service loses at most the model in progress; ``--resume <run dir>`` reuses every
model whose results are complete and error-free.
"""

import platform
import re
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil  # type: ignore[import-untyped]  # no stubs in the locked dev group
from pydantic import BaseModel, ConfigDict, Field

from baulens.config import ReportConfig
from baulens.paths import project_path
from baulens.report.clients import GenerationOptions, LLMClient, LLMError, OllamaClient
from baulens.report.completeness import check_completeness
from baulens.report.facts import build_facts
from baulens.report.generate import generate_report, generate_template
from baulens.report.metrics import (
    TEMPLATE_ROW,
    ModelMetrics,
    compute_metrics,
    metrics_table,
    save_figure,
)
from baulens.report.monitor import LoadMonitor
from baulens.report.prompting import ChatMessage, load_prompt
from baulens.report.replay import RecordingClient, replay_path, write_records
from baulens.report.schemas import LLMResponse, ReportResult, Source
from baulens.schemas import InspectionInput

DOC_BEGIN = "<!-- compare:begin -->"
DOC_END = "<!-- compare:end -->"
FIGURE_NAME = "model_comparison.png"
SHARED_LOAD_PERCENT = 25.0
MAX_CONSECUTIVE_ERRORS = 3


class RunInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    started_at: str
    finished_at: str
    prompt_version: str
    cases: int
    source: Source
    ollama_version: str | None
    machine: str
    logical_cpus: int | None
    memory_gib: float
    cpu_before_percent: float
    resumed_models: list[str] = Field(default_factory=list)
    options: dict[str, float | int | str]


class ComparisonRun(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    info: RunInfo
    metrics: list[ModelMetrics]
    run_dir: str


Progress = Callable[[str], None]


class _UnavailableClient:
    """Stands in for a model whose service failed repeatedly; every call raises."""

    def __init__(self, model: str, reason: str) -> None:
        self._model = model
        self._reason = reason

    @property
    def model(self) -> str:
        return self._model

    @property
    def source(self) -> Source:
        return "live"

    def generate(
        self,
        messages: Sequence[ChatMessage],
        schema: dict[str, Any],
        options: GenerationOptions,
    ) -> LLMResponse:
        raise LLMError(self._reason)


def _file_stem(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", model)


def results_path(run_dir: Path, model: str) -> Path:
    return run_dir / f"results_{_file_stem(model)}.jsonl"


def metrics_path(run_dir: Path, model: str) -> Path:
    return run_dir / f"metrics_{_file_stem(model)}.json"


def _write_results(path: Path, results: Sequence[ReportResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for result in results:
            handle.write(result.model_dump_json() + "\n")


def load_results(path: Path) -> list[ReportResult]:
    with path.open(encoding="utf-8") as handle:
        return [ReportResult.model_validate_json(line) for line in handle if line.strip()]


def run_model(
    model: str,
    cases: Sequence[InspectionInput],
    client: LLMClient,
    config: ReportConfig,
    *,
    run_dir: Path,
    record_replays: bool,
    progress: Progress,
) -> ModelMetrics:
    """Run all cases with one model, then write its results, metrics and replays."""
    active: LLMClient = client
    load_time_s: float | None = None
    if isinstance(client, OllamaClient):
        try:
            load_time_s = client.load()
        except LLMError as error:
            progress(f"{model}: could not load the model: {error}")
            active = _UnavailableClient(model, f"model could not be loaded: {error}")
    recorder = RecordingClient(active)
    results: list[ReportResult] = []
    sizes: list[int] = []
    consecutive_errors = 0
    out_path = results_path(run_dir, model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with LoadMonitor() as monitor, out_path.open("w", encoding="utf-8", newline="\n") as out:
        for index, case in enumerate(cases, start=1):
            result = generate_report(case, recorder, config)
            results.append(result)
            out.write(result.model_dump_json() + "\n")
            out.flush()
            consecutive_errors = consecutive_errors + 1 if result.status == "error" else 0
            if consecutive_errors == MAX_CONSECUTIVE_ERRORS and not isinstance(
                active, _UnavailableClient
            ):
                progress(f"{model}: {consecutive_errors} consecutive errors, skipping the rest")
                active = _UnavailableClient(model, "skipped after repeated service errors")
                recorder = RecordingClient(active)
            if isinstance(client, OllamaClient) and (size := client.loaded_size_bytes()):
                sizes.append(size)
            violations = (
                len(result.initial_faithfulness.violations) if result.initial_faithfulness else "-"
            )
            progress(
                f"{model} {index}/{len(cases)} {case.case_id}: {result.status} "
                f"(initial violations {violations}, {result.latency_s:.0f} s)"
            )
    if isinstance(client, OllamaClient):
        try:
            client.unload()
        except LLMError as error:
            progress(f"{model}: unload failed: {error}")
    metrics = compute_metrics(
        model,
        results,
        load=monitor.summary(),
        load_time_s=load_time_s,
        loaded_size_bytes=max(sizes) if sizes else None,
    )
    metrics_path(run_dir, model).write_text(metrics.model_dump_json(indent=2), encoding="utf-8")
    if record_replays and not any(result.status == "error" for result in results):
        write_records(replay_path(project_path(config.replay_dir), model), recorder.records)
    elif record_replays:
        progress(f"{model}: errors occurred, replay file left unchanged")
    return metrics


def rescore(
    results: Sequence[ReportResult], cases: Sequence[InspectionInput]
) -> list[ReportResult]:
    """Recompute completeness of stored drafts with the current rules (no model calls)."""
    by_id = {case.case_id: case for case in cases}
    rescored = []
    for result in results:
        if result.candidate is not None and result.faithfulness is not None:
            sheet = build_facts(by_id[result.case_id])
            completeness = check_completeness(result.candidate, result.faithfulness, sheet)
            result = result.model_copy(update={"completeness": completeness})
        rescored.append(result)
    return rescored


def resumable_metrics(
    run_dir: Path, model: str, cases: Sequence[InspectionInput]
) -> ModelMetrics | None:
    """Re-scored metrics of a finished, error-free model in an earlier run directory."""
    path = results_path(run_dir, model)
    if not path.exists():
        return None
    results = load_results(path)
    if len(results) != len(cases) or any(result.status == "error" for result in results):
        return None
    results = rescore(results, cases)
    _write_results(path, results)
    saved_path = metrics_path(run_dir, model)
    saved = (
        ModelMetrics.model_validate_json(saved_path.read_text(encoding="utf-8"))
        if saved_path.exists()
        else None
    )
    # Without a saved metrics file the results survived a crash but the resource samples
    # did not.
    metrics = compute_metrics(
        model,
        results,
        load=saved.load if saved else None,
        load_time_s=saved.load_time_s if saved else None,
        loaded_size_bytes=saved.loaded_size_bytes if saved else None,
    )
    saved_path.write_text(metrics.model_dump_json(indent=2), encoding="utf-8")
    return metrics


def run_template(
    cases: Sequence[InspectionInput], config: ReportConfig, *, run_dir: Path
) -> ModelMetrics:
    results = []
    for case in cases:
        started = time.perf_counter()
        result = generate_template(case, config=config)
        results.append(result.model_copy(update={"latency_s": time.perf_counter() - started}))
    _write_results(run_dir / "results_template.jsonl", results)
    return compute_metrics(TEMPLATE_ROW, results)


def update_results_doc(path: Path, block: str) -> bool:
    """Replace the generated block between the markers; return False if they are missing."""
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(re.escape(DOC_BEGIN) + r".*?" + re.escape(DOC_END), re.DOTALL)
    if not pattern.search(text):
        return False
    replacement = f"{DOC_BEGIN}\n{block.strip()}\n{DOC_END}"
    path.write_text(pattern.sub(lambda _: replacement, text), encoding="utf-8", newline="\n")
    return True


def summary_block(run: ComparisonRun) -> str:
    info = run.info
    notes = [
        f"_Generated by `baulens report compare` on {info.finished_at[:10]}: {info.cases} cases, "
        f"prompt `{info.prompt_version}`, Ollama {info.ollama_version or 'n/a'}, "
        f"{info.machine}, {info.logical_cpus} logical CPUs, {info.memory_gib:.0f} GiB RAM, "
        f"system CPU load {info.cpu_before_percent:.0f} % before the run"
    ]
    if info.cpu_before_percent >= SHARED_LOAD_PERCENT:
        notes.append("measured under shared load, so latencies are pessimistic")
    if info.source == "replay":
        notes.append(
            "re-scored from recorded replies: latencies are those of the recorded run and "
            "resource columns are not measured"
        )
    if info.resumed_models:
        notes.append(
            "resumed after a crash of the model service; resource columns of "
            + ", ".join(info.resumed_models)
            + " were not recorded"
        )
    return "; ".join(notes) + "._\n\n" + metrics_table(run.metrics)


def run_comparison(
    config: ReportConfig,
    cases: Sequence[InspectionInput],
    *,
    models: Sequence[str] | None = None,
    client_factory: Callable[[str], LLMClient],
    ollama_version: str | None = None,
    record_replays: bool = True,
    source: Source = "live",
    resume_dir: Path | None = None,
    progress: Progress = lambda _: None,
) -> ComparisonRun:
    started = datetime.now(UTC)
    run_dir = resume_dir or (
        project_path(config.artifacts_dir) / "runs" / started.strftime("%Y%m%dT%H%M%SZ")
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    prompt = load_prompt(config.prompt_template)
    cpu_before = float(psutil.cpu_percent(interval=2))
    rows: list[ModelMetrics] = []
    resumed: list[str] = []
    for model in models or config.models:
        previous = resumable_metrics(run_dir, model, cases) if resume_dir else None
        if previous is not None:
            progress(f"{model}: reusing complete results from {run_dir}")
            rows.append(previous)
            if previous.load is None:
                resumed.append(model)
            continue
        progress(f"running {model} on {len(cases)} cases")
        client = client_factory(model)
        try:
            rows.append(
                run_model(
                    model,
                    cases,
                    client,
                    config,
                    run_dir=run_dir,
                    record_replays=record_replays,
                    progress=progress,
                )
            )
        finally:
            if isinstance(client, OllamaClient):
                client.close()
    rows.append(run_template(cases, config, run_dir=run_dir))
    info = RunInfo(
        started_at=started.isoformat(timespec="seconds"),
        finished_at=datetime.now(UTC).isoformat(timespec="seconds"),
        prompt_version=prompt.prompt_version,
        cases=len(cases),
        source=source,
        ollama_version=ollama_version,
        machine=f"{platform.system()} {platform.release()}, {platform.processor() or 'CPU n/a'}",
        logical_cpus=psutil.cpu_count(),
        memory_gib=psutil.virtual_memory().total / 2**30,
        cpu_before_percent=cpu_before,
        resumed_models=resumed,
        options={
            "temperature": config.temperature,
            "seed": config.seed,
            "max_tokens": config.max_tokens,
            "num_ctx": config.num_ctx,
            "max_schema_retries": config.max_schema_retries,
            "max_correction_retries": config.max_correction_retries,
        },
    )
    run = ComparisonRun(info=info, metrics=rows, run_dir=str(run_dir))
    (run_dir / "summary.json").write_text(run.model_dump_json(indent=2), encoding="utf-8")
    block = summary_block(run)
    (run_dir / "summary.md").write_text(block, encoding="utf-8")
    if config.figures_dir is not None:
        subtitle = (
            f"{len(cases)} synthetic cases · prompt {info.prompt_version} · temperature "
            f"{config.temperature}, seed {config.seed} · CPU only"
            + (", measured under shared load" if cpu_before >= SHARED_LOAD_PERCENT else "")
        )
        save_figure(rows, project_path(config.figures_dir) / FIGURE_NAME, subtitle=subtitle)
    doc = project_path(config.results_doc) if config.results_doc is not None else None
    if doc is not None and not update_results_doc(doc, block):
        progress(f"markers not found in {config.results_doc}; table only in {run_dir}")
    return run
