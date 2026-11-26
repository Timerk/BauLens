"""Model directory layout and the ``metadata.json`` stored next to the weights."""

import hashlib
import logging
import subprocess
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from baulens.artifacts import (
    ArtifactError,
    current_generations,
    publish_generation,
    read_generation,
    staging_directory,
)
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
    def checkpoint_in_progress(self) -> Path:
        """Best-so-far weights of a running training job; never read by evaluate or the API."""
        return self.root / "checkpoint_in_progress.pt"


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
    weights_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
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
    def parse(cls, content: bytes) -> "ModelMetadata":
        return cls.model_validate_json(content)

    def dump(self) -> bytes:
        return (self.model_dump_json(indent=2) + "\n").encode()


@dataclass(frozen=True)
class PublishedModel:
    """Verified bytes of one published generation (see :mod:`baulens.artifacts`)."""

    directory: Path
    metadata: ModelMetadata
    weights: bytes
    onnx: bytes | None


def load_published(model_dir: Path, *, require_onnx: bool = False) -> PublishedModel:
    """Read weights, metadata and (if present) the ONNX graph from the current generation.

    Besides the generation manifest, the weights must match ``weights_sha256`` and the graph
    must match ``onnx.sha256`` recorded in the metadata, so the sidecar describes exactly the
    bytes that are loaded.
    """
    names = [MODEL_FILE, METADATA_FILE, *([ONNX_FILE] if require_onnx else [])]
    directory, files = read_generation(model_dir, names, optional=[ONNX_FILE])
    metadata = ModelMetadata.parse(files[METADATA_FILE])
    if metadata.weights_sha256 != sha256(files[MODEL_FILE]).hexdigest():
        raise ArtifactError(f"{MODEL_FILE} in {directory} does not match metadata weights_sha256")
    graph = files.get(ONNX_FILE)
    if graph is not None and metadata.onnx.get("sha256") != sha256(graph).hexdigest():
        raise ArtifactError(f"{ONNX_FILE} in {directory} does not match metadata onnx.sha256")
    return PublishedModel(directory, metadata, files[MODEL_FILE], graph)


def publish_model(
    model_dir: Path, metadata: ModelMetadata, weights: bytes, onnx_graph: bytes | None = None
) -> Path:
    """Publish weights, metadata and optionally a graph together as one new generation."""
    metadata.weights_sha256 = sha256(weights).hexdigest()
    if onnx_graph is not None and metadata.onnx.get("sha256") != sha256(onnx_graph).hexdigest():
        raise ArtifactError("metadata onnx.sha256 does not describe the graph being published")
    with staging_directory(model_dir) as staging:
        (staging / MODEL_FILE).write_bytes(weights)
        if onnx_graph is not None:
            (staging / ONNX_FILE).write_bytes(onnx_graph)
        (staging / METADATA_FILE).write_bytes(metadata.dump())
        return publish_generation(staging, model_dir)


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


class LegacyMigrationError(RuntimeError):
    """A pre-generation checkpoint could not be verified and was not migrated."""


def legacy_files(model_dir: Path) -> tuple[Path, Path] | None:
    """``model.pt`` and ``metadata.json`` of a checkpoint saved before generations existed."""
    weights, metadata = model_dir / MODEL_FILE, model_dir / METADATA_FILE
    if current_generations(model_dir) or not (weights.exists() and metadata.exists()):
        return None
    return weights, metadata
