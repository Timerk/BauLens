"""Model directory layout and the ``metadata.json`` stored next to the weights."""

import hashlib
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from baulens.config import CracksConfig
from baulens.cracks.model import Architecture
from baulens.cracks.splits import data_checksum, split_checksum, split_settings
from baulens.paths import PROJECT_ROOT, project_path

MODEL_FILE = "model.pt"
ONNX_FILE = "model.onnx"
METADATA_FILE = "metadata.json"
logger = logging.getLogger(__name__)


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
    split_settings: dict[str, Any] | None = None
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


def file_sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


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


class TrainingDataMismatchError(RuntimeError):
    """The current data, split or model settings differ from those the checkpoint was trained on."""


def verify_training_data(
    metadata: ModelMetadata, manifest: pd.DataFrame, config: CracksConfig
) -> None:
    """Refuse to score, export or benchmark a checkpoint on a split it was not trained with.

    Compares image contents (data checksum), the train/val/test assignment (split checksum),
    the split settings and the model input settings. Any difference means test images may
    have been training images, so the only fix is retraining.
    """
    problems = []
    if metadata.data_checksum != data_checksum(manifest):
        problems.append("image contents differ (data checksum)")
    if metadata.split_checksum != split_checksum(manifest):
        problems.append("train/val/test assignment differs (split checksum)")
    if metadata.split_settings is not None and metadata.split_settings != split_settings(config):
        problems.append(
            f"split settings differ: trained with {metadata.split_settings}, "
            f"config has {split_settings(config)}"
        )
    if metadata.architecture != config.architecture:
        problems.append(f"architecture {metadata.architecture} != config {config.architecture}")
    if metadata.image_size != config.image_size:
        problems.append(f"image_size {metadata.image_size} != config {config.image_size}")
    if problems:
        raise TrainingDataMismatchError(
            "checkpoint does not match the current data/split: "
            + "; ".join(problems)
            + ". Retrain with this configuration."
        )


def record_split_settings(metadata: ModelMetadata, config: CracksConfig, model_dir: Path) -> None:
    """One-off migration for checkpoints saved before split settings were recorded.

    Only called after :func:`verify_training_data` passed, i.e. the checksums prove that the
    current manifest is the one the checkpoint was trained on, so its settings can be stamped.
    """
    if metadata.split_settings is None:
        metadata.split_settings = split_settings(config)
        metadata.write(model_dir)
        logger.info("recorded split settings in %s (checksums matched)", model_dir)
