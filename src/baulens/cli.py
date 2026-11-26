"""Command-line entry point for the BauLens pipelines."""

from enum import StrEnum
from typing import Annotated, Never

import typer
from rich.console import Console

from baulens.data.download import DatasetError, DatasetName, download_dataset

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
def strength_train() -> None:
    """Train concrete compressive strength models."""
    _pending(2)


@strength_app.command("evaluate")
def strength_evaluate() -> None:
    """Evaluate the concrete compressive strength models."""
    _pending(2)


@cracks_app.command("train")
def cracks_train() -> None:
    """Train a concrete crack classifier."""
    _pending(3)


@cracks_app.command("evaluate")
def cracks_evaluate() -> None:
    """Evaluate the concrete crack classifier."""
    _pending(3)


@cracks_app.command("export")
def cracks_export() -> None:
    """Export the concrete crack classifier as ONNX."""
    _pending(3)


@report_app.command("generate")
def report_generate() -> None:
    """Generate an inspection report draft."""
    _pending(4)


@report_app.command("compare")
def report_compare() -> None:
    """Compare local models for inspection report drafting."""
    _pending(4)
