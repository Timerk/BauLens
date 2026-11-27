"""Command-line entry point for the BauLens pipelines."""

import json
import logging
from dataclasses import asdict
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console

from baulens.artifacts import ArtifactError
from baulens.config import (
    CracksConfig,
    GenerationMode,
    ReportConfig,
    Settings,
    StrengthConfig,
    load_config,
)
from baulens.data.download import DatasetError, DatasetName, download_dataset
from baulens.paths import project_path
from baulens.report.cases import CASES_SEED, generate_cases, load_cases, save_cases
from baulens.report.clients import LLMClient, OllamaClient
from baulens.report.compare import run_comparison
from baulens.report.facts import build_facts
from baulens.report.faithfulness import check_text
from baulens.report.generate import generate_report, generate_template
from baulens.report.metrics import metrics_table
from baulens.report.replay import ReplayClient
from baulens.report.schemas import NumberCheck, ReportResult
from baulens.schemas import InspectionInput

app = typer.Typer(help="Run the BauLens inspection pipelines.", no_args_is_help=True)
data_app = typer.Typer(help="Fetch verified copies of the public datasets.", no_args_is_help=True)
strength_app = typer.Typer(help="Predict concrete compressive strength.", no_args_is_help=True)
cracks_app = typer.Typer(help="Classify cracks in concrete images.", no_args_is_help=True)
report_app = typer.Typer(help="Draft German inspection reports.", no_args_is_help=True)
app.add_typer(data_app, name="data")
app.add_typer(strength_app, name="strength")
app.add_typer(cracks_app, name="cracks")
app.add_typer(report_app, name="report")


class DatasetSelection(StrEnum):
    CONCRETE = "concrete"
    CRACKS = "cracks"
    ALL = "all"


@data_app.command("download")
def data_download(
    dataset: Annotated[DatasetSelection, typer.Argument(help="Dataset to fetch.")],
) -> None:
    """Download, verify and extract a dataset into data/raw/."""
    names: tuple[DatasetName, ...]
    match dataset:
        case DatasetSelection.CONCRETE:
            names = ("concrete",)
        case DatasetSelection.CRACKS:
            names = ("cracks",)
        case DatasetSelection.ALL:
            names = ("concrete", "cracks")
    console = Console()
    for name in names:
        try:
            path = download_dataset(name, console=console)
        except DatasetError as error:
            typer.echo(f"Dataset error: {error}", err=True)
            raise typer.Exit(code=1) from error
        typer.echo(f"{name}: ready at {path}")


@strength_app.command("train")
def strength_train(
    config: Annotated[Path, typer.Option(help="YAML training configuration.")] = Path(
        "configs/strength.yaml"
    ),
) -> None:
    """Train concrete compressive strength models."""
    from baulens.strength import train

    try:
        result = train(load_config(config, StrengthConfig))
    except (OSError, ValueError) as error:
        typer.echo(f"Strength training error: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(
        f"{result.model_name} ({result.model_version}), CV MAE {result.grouped_cv_mae:.2f} MPa"
    )
    typer.echo(f"Model saved to {result.model_dir}")


@strength_app.command("evaluate")
def strength_evaluate(
    config: Annotated[Path, typer.Option(help="YAML training configuration.")] = Path(
        "configs/strength.yaml"
    ),
) -> None:
    """Evaluate the frozen winner on unseen concrete mix designs."""
    from baulens.strength import evaluate

    try:
        result = evaluate(load_config(config, StrengthConfig))
    except (OSError, ValueError) as error:
        typer.echo(f"Strength evaluation error: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(
        f"{result.model_name} ({result.model_version}), test MAE {result.scores.mae:.2f} MPa"
    )
    typer.echo(f"Interval coverage {result.coverage:.1%}. Results saved to {result.results_path}")


@strength_app.command("predict")
def strength_predict(
    mix_json: Annotated[Path, typer.Option(help="JSON file containing a ConcreteMix.")],
    config: Annotated[Path, typer.Option(help="YAML training configuration.")] = Path(
        "configs/strength.yaml"
    ),
) -> None:
    """Predict strength and return the prediction as JSON."""
    from baulens.schemas import ConcreteMix
    from baulens.strength.inference import StrengthPredictor

    try:
        settings = load_config(config, StrengthConfig)
        mix = ConcreteMix.model_validate_json(mix_json.read_text(encoding="utf-8"))
        predictor = StrengthPredictor.load(project_path(settings.artifacts_dir) / "model")
        result = predictor.predict(mix)
    except (OSError, ValueError) as error:
        typer.echo(f"Strength prediction error: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(result.model_dump_json(indent=2))


CracksConfigOption = Annotated[
    Path, typer.Option("--config", help="YAML settings validated by CracksConfig.")
]
DEFAULT_CRACKS_CONFIG = Path("configs/cracks.yaml")


def _cracks_config(path: Path) -> CracksConfig:
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    return load_config(path, CracksConfig)


def _echo_json(payload: object) -> None:
    typer.echo(json.dumps(payload, indent=2, default=str))


@cracks_app.command("split")
def cracks_split(config: CracksConfigOption = DEFAULT_CRACKS_CONFIG) -> None:
    """Index the images, check near-duplicates and write the group-aware split manifest."""
    from baulens.cracks.splits import build_splits, split_files

    settings = _cracks_config(config)
    build_splits(settings)
    _echo_json(json.loads(split_files(settings).leakage.read_text(encoding="utf-8")))


@cracks_app.command("train")
def cracks_train(config: CracksConfigOption = DEFAULT_CRACKS_CONFIG) -> None:
    """Train a concrete crack classifier."""
    from baulens.cracks.train import train

    _echo_json(asdict(train(_cracks_config(config))))


@cracks_app.command("evaluate")
def cracks_evaluate(config: CracksConfigOption = DEFAULT_CRACKS_CONFIG) -> None:
    """Evaluate the concrete crack classifier."""
    from baulens.cracks.evaluate import evaluate
    from baulens.cracks.metadata import TrainingDataMismatchError

    try:
        result = asdict(evaluate(_cracks_config(config)))
    except (TrainingDataMismatchError, ArtifactError) as error:
        typer.echo(f"Evaluation refused: {error}", err=True)
        raise typer.Exit(code=1) from error
    _echo_json({key: value for key, value in result.items() if key != "robustness"})


@cracks_app.command("export")
def cracks_export(config: CracksConfigOption = DEFAULT_CRACKS_CONFIG) -> None:
    """Export the concrete crack classifier as ONNX."""
    from baulens.cracks.export import ExportError, export
    from baulens.cracks.metadata import TrainingDataMismatchError

    try:
        result = export(_cracks_config(config))
    except (ExportError, TrainingDataMismatchError, ArtifactError) as error:
        typer.echo(f"Export error: {error}", err=True)
        raise typer.Exit(code=1) from error
    _echo_json(asdict(result))


@cracks_app.command("migrate")
def cracks_migrate(config: CracksConfigOption = DEFAULT_CRACKS_CONFIG) -> None:
    """Publish a pre-generation checkpoint after verifying its provenance (one-off)."""
    from baulens.cracks.metadata import LegacyMigrationError, TrainingDataMismatchError
    from baulens.cracks.migrate import migrate_legacy

    try:
        record = migrate_legacy(_cracks_config(config))
    except (LegacyMigrationError, TrainingDataMismatchError) as error:
        typer.echo(f"Migration refused: {error}", err=True)
        raise typer.Exit(code=1) from error
    _echo_json(record)


@cracks_app.command("benchmark")
def cracks_benchmark(config: CracksConfigOption = DEFAULT_CRACKS_CONFIG) -> None:
    """Benchmark CPU inference latency of PyTorch and ONNX Runtime."""
    from baulens.cracks.benchmark import benchmark
    from baulens.cracks.metadata import TrainingDataMismatchError

    try:
        result = benchmark(_cracks_config(config))
    except (TrainingDataMismatchError, ArtifactError) as error:
        typer.echo(f"Benchmark refused: {error}", err=True)
        raise typer.Exit(code=1) from error
    _echo_json(asdict(result))


class GenerationChoice(StrEnum):
    SLOTS = "slots"
    FREE = "free"


class CompareSource(StrEnum):
    LIVE = "live"
    REPLAY = "replay"


class ReportMode(StrEnum):
    LIVE = "live"
    REPLAY = "replay"
    TEMPLATE = "template"


ReportConfigOption = Annotated[
    Path, typer.Option("--config", help="Report settings (YAML).", dir_okay=False)
]
CaseOption = Annotated[
    Path, typer.Option("--input", help="InspectionInput as JSON.", exists=True, dir_okay=False)
]


def _read_case(path: Path) -> InspectionInput:
    try:
        return InspectionInput.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as error:
        typer.echo(f"Invalid input {path}: {error}", err=True)
        raise typer.Exit(code=2) from error


def _say(console: Console, text: str) -> None:
    console.print(text, markup=False, highlight=False, soft_wrap=True)


def _print_checks(console: Console, checks: list[NumberCheck]) -> None:
    for check in checks:
        _say(console, f"{check.status:>13}  {check.section}: {check.text!r} - {check.reason}")


@report_app.command("generate")
def report_generate(
    input_path: CaseOption,
    model: Annotated[
        str | None, typer.Option(help="Model tag; default: default_model in config.")
    ] = None,
    mode: Annotated[
        ReportMode | None, typer.Option(help="live, replay or template; default BAULENS_LLM_MODE.")
    ] = None,
    generation: Annotated[
        GenerationChoice | None,
        typer.Option(help="slots (numbers inserted by code) or free; default from config."),
    ] = None,
    config_path: ReportConfigOption = Path("configs/report.yaml"),
    as_json: Annotated[bool, typer.Option("--json", help="Print the full ReportResult.")] = False,
) -> None:
    """Generate a German inspection report draft behind the faithfulness gate."""
    config = load_config(config_path, ReportConfig)
    settings = Settings()
    case = _read_case(input_path)
    selected = mode or ReportMode(settings.llm_mode)
    model_name = model or config.default_model
    chosen: GenerationMode = generation.value if generation else config.generation_mode
    result: ReportResult
    if selected is ReportMode.TEMPLATE:
        result = generate_template(case, config=config)
    elif selected is ReportMode.REPLAY:
        replay = ReplayClient.from_directory(project_path(config.replay_dir), model_name)
        result = generate_report(case, replay, config, mode=chosen)
    else:
        client = OllamaClient(
            str(settings.ollama_host),
            model_name,
            timeout_s=config.request_timeout_s,
            keep_alive=config.keep_alive,
        )
        try:
            result = generate_report(case, client, config, mode=chosen)
        finally:
            client.close()
    console = Console()
    if as_json:
        typer.echo(result.model_dump_json(indent=2))
        return
    _say(console, result.delivered.to_markdown())
    _say(
        console,
        f"status: {result.status} | source: {result.source} | model: {result.model or '-'} | "
        f"prompt: {result.prompt_version} | latency: {result.latency_s:.1f} s",
    )
    if result.note:
        _say(console, f"note: {result.note}")
    if result.faithfulness is not None and result.faithfulness.violations:
        _say(console, "violations in the last model draft:")
        _print_checks(console, result.faithfulness.violations)
    if result.completeness is not None:
        missing = ", ".join(result.completeness.missing) or "none"
        _say(console, f"completeness: {result.completeness.score:.2f} (missing: {missing})")


@report_app.command("check")
def report_check(
    input_path: CaseOption,
    text_path: Annotated[
        Path, typer.Option("--text", help="Report text to check.", exists=True, dir_okay=False)
    ],
    show_all: Annotated[bool, typer.Option("--all", help="Also list accepted numbers.")] = False,
) -> None:
    """Run the numeric faithfulness checker on a text; exit 1 if any number is unfaithful."""
    case = _read_case(input_path)
    result = check_text(text_path.read_text(encoding="utf-8"), build_facts(case))
    console = Console()
    _print_checks(console, result.checks if show_all else result.violations)
    counts = ", ".join(f"{kind} {count}" for kind, count in result.violation_counts().items())
    _say(
        console,
        f"{len(result.checks)} numbers checked, {len(result.violations)} violations ({counts})",
    )
    if not result.faithful:
        raise typer.Exit(code=1)


@report_app.command("compare")
def report_compare(
    config_path: ReportConfigOption = Path("configs/report.yaml"),
    models: Annotated[
        list[str] | None, typer.Option("--model", help="Restrict to these models (repeatable).")
    ] = None,
    limit: Annotated[int | None, typer.Option(min=1, help="Use only the first N cases.")] = None,
    record_replays: Annotated[
        bool, typer.Option("--replays/--no-replays", help="Rewrite data/report_replays/.")
    ] = True,
    update_docs: Annotated[
        bool, typer.Option("--docs/--no-docs", help="Rewrite the table and figure in docs/.")
    ] = True,
    source: Annotated[
        CompareSource,
        typer.Option(help="live: query Ollama; replay: re-score the recorded replies offline."),
    ] = CompareSource.LIVE,
    resume: Annotated[
        Path | None,
        typer.Option(help="Earlier run directory; reuse its complete models.", file_okay=False),
    ] = None,
) -> None:
    """Compare the configured local models on all evaluation cases."""
    config = load_config(config_path, ReportConfig)
    if not update_docs:
        config = config.model_copy(update={"results_doc": None, "figures_dir": None})
    host = str(Settings().ollama_host)
    cases = load_cases(project_path(config.cases_path))[:limit]
    console = Console()
    replay_dir = project_path(config.replay_dir)

    def live(model: str) -> LLMClient:
        return OllamaClient(
            host, model, timeout_s=config.request_timeout_s, keep_alive=config.keep_alive
        )

    def replayed(model: str) -> LLMClient:
        return ReplayClient.from_directory(replay_dir, model)

    version: str | None = None
    if source is CompareSource.LIVE:
        probe = OllamaClient(host, config.models[0])
        version = probe.version()
        probe.close()
        if version is None:
            typer.echo(f"Ollama is not reachable at {host}", err=True)
            raise typer.Exit(code=1)
    run = run_comparison(
        config,
        cases,
        models=models,
        client_factory=live if source is CompareSource.LIVE else replayed,
        ollama_version=version,
        record_replays=record_replays and source is CompareSource.LIVE,
        source=source.value,
        resume_dir=project_path(resume) if resume is not None else None,
        progress=lambda line: _say(console, line),
    )
    _say(console, metrics_table(run.metrics))
    _say(console, f"results: {run.run_dir}")


@report_app.command("make-cases")
def report_make_cases(
    output: Annotated[Path, typer.Option(help="Destination JSONL.")] = Path(
        "data/report_cases/cases.jsonl"
    ),
    seed: Annotated[int, typer.Option(help="Generator seed.")] = CASES_SEED,
) -> None:
    """Regenerate the synthetic evaluation cases deterministically."""
    cases = generate_cases(seed)
    save_cases(project_path(output), cases)
    typer.echo(f"wrote {len(cases)} cases to {output}")
