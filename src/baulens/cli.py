"""Command-line entry point for the BauLens pipelines."""

from typing import Never

import typer

app = typer.Typer(help="Run the BauLens inspection pipelines.", no_args_is_help=True)
strength_app = typer.Typer(help="Predict concrete compressive strength.", no_args_is_help=True)
cracks_app = typer.Typer(help="Classify cracks in concrete images.", no_args_is_help=True)
report_app = typer.Typer(help="Draft German inspection reports.", no_args_is_help=True)
app.add_typer(strength_app, name="strength")
app.add_typer(cracks_app, name="cracks")
app.add_typer(report_app, name="report")


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
