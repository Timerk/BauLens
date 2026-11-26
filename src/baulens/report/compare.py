"""Run every evaluation case through every model and generation mode and summarise the results.

One command reproduces the committed numbers: ``baulens report compare``. For each model it
runs every mode in ``config.compare_modes`` (free-text and slots) with identical decoding
options, then unloads the model (``keep_alive`` 0) before loading the next one. Raw results
go to ``artifacts/report/runs/<timestamp>/``; the replay files, the figure and the generated
block in ``docs/results/report.md`` are rewritten in place.

Each (model, mode) run writes its results, metrics and a manifest as soon as it finishes.
``--resume <run dir>`` reuses a run only if its manifest matches the current model, mode,
prompt version, checker version, decoding options and the exact input digests of all cases;
anything else is re-run.
"""

import platform
import re
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil  # type: ignore[import-untyped]  # no stubs in the locked dev group
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from baulens.config import ReportConfig
from baulens.paths import project_path
from baulens.report.clients import GenerationOptions, LLMClient, LLMError, OllamaClient
from baulens.report.completeness import check_completeness
from baulens.report.facts import build_facts
from baulens.report.faithfulness import CHECKER_VERSION
from baulens.report.generate import (
    generate_report,
    generate_template,
    generation_options,
    input_sha256,
    prompt_for,
)
from baulens.report.metrics import (
    TEMPLATE_ROW,
    ModelMetrics,
    compute_metrics,
    metrics_table,
    save_figure,
)
from baulens.report.monitor import LoadMonitor
from baulens.report.prompting import ChatMessage
from baulens.report.replay import RecordingClient, ReplayRecord, replay_path, write_records
from baulens.report.schemas import LLMResponse, Mode, ReportResult, Source
from baulens.schemas import InspectionInput

DOC_BEGIN = "<!-- compare:begin -->"
DOC_END = "<!-- compare:end -->"
FIGURE_NAME = "model_comparison.png"
SHARED_LOAD_PERCENT = 25.0
MAX_CONSECUTIVE_ERRORS = 3


class RunManifest(BaseModel):
    """What a (model, mode) result file was produced from; checked before any reuse."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str
    mode: Mode
    prompt_version: str
    checker_version: str
    options: dict[str, float | int | str]
    inputs: dict[str, str]


class RunInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    started_at: str
    finished_at: str
    prompt_versions: dict[str, str]
    checker_version: str
    cases: int
    source: Source
    ollama_version: str | None
    machine: str
    logical_cpus: int | None
    memory_gib: float
    cpu_before_percent: float
    reused: list[str] = Field(default_factory=list)
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


def _stem(model: str, mode: Mode) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", f"{model}__{mode}")


def results_path(run_dir: Path, model: str, mode: Mode) -> Path:
    return run_dir / f"results_{_stem(model, mode)}.jsonl"


def metrics_path(run_dir: Path, model: str, mode: Mode) -> Path:
    return run_dir / f"metrics_{_stem(model, mode)}.json"


def manifest_path(run_dir: Path, model: str, mode: Mode) -> Path:
    return run_dir / f"manifest_{_stem(model, mode)}.json"


def _options(config: ReportConfig) -> dict[str, float | int | str]:
    return {
        **generation_options(config).model_dump(),
        "max_schema_retries": config.max_schema_retries,
        "max_correction_retries": config.max_correction_retries,
    }


def manifest_for(
    model: str, mode: Mode, cases: Sequence[InspectionInput], config: ReportConfig
) -> RunManifest:
    return RunManifest(
        model=model,
        mode=mode,
        prompt_version=prompt_for(config, mode).prompt_version,
        checker_version=CHECKER_VERSION,
        options=_options(config),
        inputs={case.case_id: input_sha256(case) for case in cases},
    )


def _write_results(path: Path, results: Sequence[ReportResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for result in results:
            handle.write(result.model_dump_json() + "\n")


def load_results(path: Path) -> list[ReportResult]:
    with path.open(encoding="utf-8") as handle:
        return [ReportResult.model_validate_json(line) for line in handle if line.strip()]


def run_mode(
    model: str,
    mode: Mode,
    cases: Sequence[InspectionInput],
    client: LLMClient,
    config: ReportConfig,
    *,
    run_dir: Path,
    load_time_s: float | None,
    progress: Progress,
) -> tuple[ModelMetrics, list[ReplayRecord], bool]:
    """Run all cases with one model in one mode; return metrics, replay records, clean flag."""
    active: LLMClient = client
    recorder = RecordingClient(active)
    results: list[ReportResult] = []
    sizes: list[int] = []
    consecutive_errors = 0
    out_path = results_path(run_dir, model, mode)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with LoadMonitor() as monitor, out_path.open("w", encoding="utf-8", newline="\n") as out:
        for index, case in enumerate(cases, start=1):
            result = generate_report(case, recorder, config, mode=mode)
            results.append(result)
            out.write(result.model_dump_json() + "\n")
            out.flush()
            consecutive_errors = consecutive_errors + 1 if result.status == "error" else 0
            if consecutive_errors == MAX_CONSECUTIVE_ERRORS:
                progress(f"{model}/{mode}: {consecutive_errors} consecutive errors, skipping")
                active = _UnavailableClient(model, "skipped after repeated service errors")
                recorder = RecordingClient(active)
            if isinstance(client, OllamaClient) and (size := client.loaded_size_bytes()):
                sizes.append(size)
            first = result.initial_faithfulness
            progress(
                f"{model}/{mode} {index}/{len(cases)} {case.case_id}: {result.status} "
                f"(first-draft violations {len(first.violations) if first else '-'}, "
                f"{result.latency_s:.0f} s)"
            )
    metrics = compute_metrics(
        model,
        results,
        mode=mode,
        load=monitor.summary(),
        load_time_s=load_time_s,
        loaded_size_bytes=max(sizes) if sizes else None,
    )
    clean = not any(result.status in {"error", "replay_miss"} for result in results)
    metrics_path(run_dir, model, mode).write_text(
        metrics.model_dump_json(indent=2), encoding="utf-8"
    )
    if clean:
        manifest = manifest_for(model, mode, cases, config)
        manifest_path(run_dir, model, mode).write_text(
            manifest.model_dump_json(indent=2), encoding="utf-8"
        )
    return metrics, recorder.records, clean


def rescore(
    results: Sequence[ReportResult], cases: Sequence[InspectionInput]
) -> list[ReportResult]:
    """Recompute completeness of stored drafts with the current rules (no model calls)."""
    by_id = {case.case_id: case for case in cases}
    rescored = []
    for result in results:
        case = by_id[result.case_id]
        if result.candidate is not None and result.faithfulness is not None:
            completeness = check_completeness(
                result.candidate, result.faithfulness, build_facts(case)
            )
            result = result.model_copy(update={"completeness": completeness})
        rescored.append(result)
    return rescored


def reusable(
    run_dir: Path, model: str, mode: Mode, cases: Sequence[InspectionInput], config: ReportConfig
) -> tuple[ModelMetrics | None, str]:
    """Metrics of an earlier (model, mode) run if it is provably the same run, else a reason."""
    path, saved = results_path(run_dir, model, mode), manifest_path(run_dir, model, mode)
    if not path.exists() or not saved.exists():
        return None, "no complete earlier run"
    try:
        manifest = RunManifest.model_validate_json(saved.read_text(encoding="utf-8"))
        results = load_results(path)
    except (ValidationError, ValueError) as error:
        return None, f"unreadable earlier run ({error.__class__.__name__})"
    expected = manifest_for(model, mode, cases, config)
    for field in ("model", "mode", "prompt_version", "checker_version", "options", "inputs"):
        if getattr(manifest, field) != getattr(expected, field):
            return None, f"{field} differs from the earlier run"
    if {result.case_id: result.input_sha256 for result in results} != expected.inputs:
        return None, "stored results do not match the current inputs"
    results = rescore(results, cases)
    _write_results(path, results)
    previous = metrics_path(run_dir, model, mode)
    saved_metrics = (
        ModelMetrics.model_validate_json(previous.read_text(encoding="utf-8"))
        if previous.exists()
        else None
    )
    metrics = compute_metrics(
        model,
        results,
        mode=mode,
        load=saved_metrics.load if saved_metrics else None,
        load_time_s=saved_metrics.load_time_s if saved_metrics else None,
        loaded_size_bytes=saved_metrics.loaded_size_bytes if saved_metrics else None,
    )
    previous.write_text(metrics.model_dump_json(indent=2), encoding="utf-8")
    return metrics, "reused"


def run_template(
    cases: Sequence[InspectionInput], config: ReportConfig, *, run_dir: Path
) -> ModelMetrics:
    results = []
    for case in cases:
        started = time.perf_counter()
        result = generate_template(case, config=config)
        results.append(result.model_copy(update={"latency_s": time.perf_counter() - started}))
    _write_results(run_dir / "results_template.jsonl", results)
    return compute_metrics(TEMPLATE_ROW, results, baseline=True)


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
    prompts = ", ".join(f"{mode} `{version}`" for mode, version in info.prompt_versions.items())
    notes = [
        f"_Generated by `baulens report compare` on {info.finished_at[:10]}: {info.cases} cases, "
        f"prompts {prompts}, checker v{info.checker_version}, Ollama "
        f"{info.ollama_version or 'n/a'}, {info.machine}, {info.logical_cpus} logical CPUs, "
        f"{info.memory_gib:.0f} GiB RAM, system CPU load {info.cpu_before_percent:.0f} % "
        "before this invocation"
    ]
    if info.cpu_before_percent >= SHARED_LOAD_PERCENT:
        notes.append("measured under shared load, so latencies are pessimistic")
    if info.source == "replay":
        notes.append(
            "re-scored from recorded replies: latencies are those of the recorded run and "
            "resource columns are not measured"
        )
    if info.reused:
        notes.append(
            "results of "
            + ", ".join(info.reused)
            + " come from earlier invocations into the same run directory and were reused "
            "after manifest validation (inputs, prompts, checker, options); resource columns "
            "were measured during those invocations"
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
    cpu_before = float(psutil.cpu_percent(interval=2))
    rows: list[ModelMetrics] = []
    reused: list[str] = []
    for model in models or config.models:
        client: LLMClient | None = None
        load_time_s: float | None = None
        records: list[ReplayRecord] = []
        all_clean = True
        for mode in config.compare_modes:
            if resume_dir is not None:
                previous, reason = reusable(run_dir, model, mode, cases, config)
                if previous is not None:
                    progress(f"{model}/{mode}: reusing results from {run_dir}")
                    rows.append(previous)
                    reused.append(f"{model} / {mode}")
                    all_clean = False  # keep the existing replay file for this model
                    continue
                progress(f"{model}/{mode}: not reusable ({reason}), running")
            if client is None:
                client = client_factory(model)
                if isinstance(client, OllamaClient):
                    try:
                        load_time_s = client.load()
                    except LLMError as error:
                        progress(f"{model}: could not load the model: {error}")
                        client.close()
                        client = _UnavailableClient(model, f"model could not be loaded: {error}")
            progress(f"running {model}/{mode} on {len(cases)} cases")
            metrics, mode_records, clean = run_mode(
                model,
                mode,
                cases,
                client,
                config,
                run_dir=run_dir,
                load_time_s=load_time_s,
                progress=progress,
            )
            rows.append(metrics)
            records += mode_records
            all_clean = all_clean and clean
        if isinstance(client, OllamaClient):
            try:
                client.unload()
            except LLMError as error:
                progress(f"{model}: unload failed: {error}")
            client.close()
        if record_replays and records and all_clean:
            write_records(replay_path(project_path(config.replay_dir), model), records)
        elif record_replays and records:
            progress(f"{model}: errors, misses or partial reuse, replay file left unchanged")
    rows.append(run_template(cases, config, run_dir=run_dir))
    info = RunInfo(
        started_at=started.isoformat(timespec="seconds"),
        finished_at=datetime.now(UTC).isoformat(timespec="seconds"),
        prompt_versions={
            mode: prompt_for(config, mode).prompt_version for mode in config.compare_modes
        },
        checker_version=CHECKER_VERSION,
        cases=len(cases),
        source=source,
        ollama_version=ollama_version,
        machine=f"{platform.system()} {platform.release()}, {platform.processor() or 'CPU n/a'}",
        logical_cpus=psutil.cpu_count(),
        memory_gib=psutil.virtual_memory().total / 2**30,
        cpu_before_percent=cpu_before,
        reused=reused,
        options=_options(config),
    )
    run = ComparisonRun(info=info, metrics=rows, run_dir=str(run_dir))
    (run_dir / "summary.json").write_text(run.model_dump_json(indent=2), encoding="utf-8")
    block = summary_block(run)
    (run_dir / "summary.md").write_text(block, encoding="utf-8")
    if config.figures_dir is not None:
        subtitle = (
            f"{len(cases)} synthetic cases · checker detections, not proofs · temperature "
            f"{config.temperature}, seed {config.seed} · CPU only"
            + (", measured under shared load" if cpu_before >= SHARED_LOAD_PERCENT else "")
        )
        save_figure(rows, project_path(config.figures_dir) / FIGURE_NAME, subtitle=subtitle)
    doc = project_path(config.results_doc) if config.results_doc is not None else None
    if doc is not None and not update_results_doc(doc, block):
        progress(f"markers not found in {config.results_doc}; table only in {run_dir}")
    return run
