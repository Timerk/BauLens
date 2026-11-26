"""Model directory layout and the ``metadata.json`` stored next to the weights."""

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from baulens.config import CracksConfig
from baulens.cracks.model import Architecture
from baulens.paths import PROJECT_ROOT, project_path

MODEL_FILE = "model.pt"
ONNX_FILE = "model.onnx"
METADATA_FILE = "metadata.json"


@dataclass(frozen=True)
class RunPaths:
    root: Path
    model_dir: Path
    figures: Path

    @property
    def model(self) -> Path:
        return self.model_dir / MODEL_FILE

    @property
    def onnx(self) -> Path:
        return self.model_dir / ONNX_FILE

    @property
    def metadata(self) -> Path:
        return self.model_dir / METADATA_FILE


def run_paths(config: CracksConfig) -> RunPaths:
    root = project_path(config.artifacts_dir)
    return RunPaths(root=root, model_dir=root / "model", figures=root / "figures")


class ModelMetadata(BaseModel):
    """Everything needed to rebuild preprocessing and the decision rule at inference time."""

    model_config = ConfigDict(extra="forbid")

    model_name: str
    model_version: str
    architecture: Architecture
    image_size: int = Field(gt=0)
    normalization_mean: tuple[float, float, float]
    normalization_std: tuple[float, float, float]
    pretrained_weights: str | None
    threshold: float | None = Field(default=None, gt=0, lt=1)
    threshold_strategy: str | None = None
    data_checksum: str
    split_checksum: str
    git_sha: str
    git_dirty: bool
    torch_version: str
    torchvision_version: str
    created_at: str
    training: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    onnx: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def read(cls, model_dir: Path) -> "ModelMetadata":
        return cls.model_validate_json((model_dir / METADATA_FILE).read_text(encoding="utf-8"))

    def write(self, model_dir: Path) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / METADATA_FILE).write_text(self.model_dump_json(indent=2), encoding="utf-8")


def git_revision() -> tuple[str, bool]:
    """Current commit and whether tracked files differ from it; ``unknown`` outside git."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown", False
    return sha, bool(status)
