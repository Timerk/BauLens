from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from baulens.cli import app
from baulens.data.download import DatasetError, DatasetName

runner = CliRunner()


def test_cli_help_lists_pipeline_groups() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for group in ("data", "strength", "cracks", "report"):
        assert group in result.stdout


def test_data_download_help_lists_dataset_options() -> None:
    result = runner.invoke(app, ["data", "download", "--help"])
    assert result.exit_code == 0
    for choice in ("concrete", "cracks", "all"):
        assert choice in result.stdout


def test_data_download_all_dispatches_both_datasets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[DatasetName] = []

    def download(name: DatasetName, *, console: Console) -> Path:
        calls.append(name)
        return tmp_path / name

    monkeypatch.setattr("baulens.cli.download_dataset", download)
    result = runner.invoke(app, ["data", "download", "all"])

    assert result.exit_code == 0
    assert calls == ["concrete", "cracks"]
    assert "concrete: ready" in result.stdout
    assert "cracks: ready" in result.stdout


def test_data_download_reports_a_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def download(name: DatasetName, *, console: Console) -> Path:
        raise DatasetError(f"SHA-256 mismatch for {name}")

    monkeypatch.setattr("baulens.cli.download_dataset", download)
    result = runner.invoke(app, ["data", "download", "cracks"])

    assert result.exit_code == 1
    assert "Dataset error: SHA-256 mismatch for cracks" in result.stderr


@pytest.mark.parametrize(
    ("group", "command", "milestone"),
    [
        ("report", "generate", 4),
        ("report", "compare", 4),
    ],
)
def test_pipeline_stub_reports_its_milestone(group: str, command: str, milestone: int) -> None:
    result = runner.invoke(app, [group, command])
    assert result.exit_code == 1
    assert f"not implemented yet (milestone {milestone})" in result.stderr
