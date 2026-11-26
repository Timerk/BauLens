"""Paths shared by the CLI and pipeline modules."""

from pathlib import Path

_CHECKOUT = Path(__file__).resolve().parents[2]
_CWD = Path.cwd()
PROJECT_ROOT = next(
    (path for path in (_CHECKOUT, _CWD, *_CWD.parents) if (path / "configs").is_dir()),
    _CWD,
)
CONFIG_DIR = PROJECT_ROOT / "configs"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"


def project_path(path: Path) -> Path:
    """Resolve a config path relative to the checkout, preserving absolute paths."""
    return PROJECT_ROOT / path
