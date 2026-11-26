"""Command-line entry point for the BauLens pipelines."""

import json
import logging
from dataclasses import asdict
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Never

import typer
from rich.console import Console

from baulens.config import CracksConfig, StrengthConfig, load_config
from baulens.data.download import DatasetError, DatasetName, download_dataset
from baulens.paths import project_path

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


def _pending(milestone: int) -> Never:
    typer.echo(f"not implemented yet (milestone {milestone})", err=True)
    raise typer.Exit(code=1)


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

    result = asdict(evaluate(_cracks_config(config)))
    _echo_json({key: value for key, value in result.items() if key != "robustness"})


@cracks_app.command("export")
def cracks_export(config: CracksConfigOption = DEFAULT_CRACKS_CONFIG) -> None:
    """Export the concrete crack classifier as ONNX."""
    from baulens.cracks.export import ExportError, export

    try:
        result = export(_cracks_config(config))
    except ExportError as error:
        typer.echo(f"Export error: {error}", err=True)
        raise typer.Exit(code=1) from error
    _echo_json(asdict(result))


@cracks_app.command("benchmark")
def cracks_benchmark(config: CracksConfigOption = DEFAULT_CRACKS_CONFIG) -> None:
    """Benchmark CPU inference latency of PyTorch and ONNX Runtime."""
    from baulens.cracks.benchmark import benchmark

    _echo_json(asdict(benchmark(_cracks_config(config))))


@report_app.command("generate")
def report_generate() -> None:
    """Generate an inspection report draft."""
    _pending(4)


@report_app.command("compare")
def report_compare() -> None:
    """Compare local models for inspection report drafting."""
    _pending(4)
