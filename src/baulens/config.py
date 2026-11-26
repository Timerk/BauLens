"""Typed configuration for pipeline runs."""

from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, PositiveInt, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from baulens.paths import PROJECT_ROOT, project_path


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int = Field(ge=0, le=2**32 - 1)
    artifacts_dir: Path


class StrengthConfig(PipelineConfig):
    data_path: Path
    test_fraction: float = Field(gt=0, lt=1)
    cv_splits: int = Field(ge=2)
    cv_repeats: PositiveInt
    calibration_fraction: float = Field(default=0.25, gt=0, lt=1)
    inner_splits: int = Field(default=3, ge=2)
    bootstrap_resamples: int = Field(default=2000, ge=2000)
    threads: PositiveInt = 2
    results_path: Path = Path("docs/results/strength.md")
    figures_dir: Path = Path("docs/figures/strength")


class CracksConfig(PipelineConfig):
    """Crack classification run: data split, transfer learning, evaluation and export.

    ``epochs`` is the maximum number of fine-tuning epochs after ``head_epochs`` epochs with
    a frozen backbone; early stopping on validation loss usually ends training sooner.
    """

    data_dir: Path
    architecture: Literal["resnet18", "efficientnet_b0"]
    image_size: PositiveInt
    batch_size: PositiveInt
    epochs: PositiveInt
    pretrained: bool

    # Data split and leakage handling
    val_fraction: float = Field(default=0.15, gt=0, lt=0.5)
    test_fraction: float = Field(default=0.15, gt=0, lt=0.5)
    near_duplicate_max_distance: int = Field(default=10, ge=0, le=32)
    near_duplicate_min_correlation: float = Field(default=0.85, ge=-1, le=1)
    train_fraction: float = Field(default=1.0, gt=0, le=1)

    # Compute
    device: Literal["auto", "cpu", "cuda"] = "auto"
    num_threads: PositiveInt | None = None
    num_workers: int = Field(default=0, ge=0)
    eval_batch_size: PositiveInt = 128

    # Optimisation
    head_epochs: int = Field(default=2, ge=0)
    finetune_blocks: int = Field(default=1, ge=0, le=4)
    lr_head: float = Field(default=1e-3, gt=0)
    lr_finetune: float = Field(default=2e-4, gt=0)
    weight_decay: float = Field(default=1e-4, ge=0)
    patience: PositiveInt = 2

    # Augmentation
    crop_scale_min: float = Field(default=0.6, gt=0, le=1)
    color_jitter: float = Field(default=0.2, ge=0, lt=1)

    # Evaluation
    threshold_strategy: Literal["f1", "youden"] = "f1"
    bootstrap_resamples: PositiveInt = 2000
    calibration_bins: int = Field(default=15, ge=2)
    robustness_max_images: PositiveInt = 2000
    gallery_size: PositiveInt = 8
    docs_figures_dir: Path | None = None

    # Export and benchmark
    onnx_opset: int = Field(default=18, ge=17)
    parity_images: PositiveInt = 256
    parity_tolerance: float = Field(default=1e-4, gt=0)
    benchmark_warmup: int = Field(default=5, ge=0)
    benchmark_iterations: PositiveInt = 50

    @model_validator(mode="after")
    def _splits_leave_training_data(self) -> Self:
        if self.val_fraction + self.test_fraction >= 0.8:
            raise ValueError("val_fraction + test_fraction must leave at least 20 % for training")
        return self


class ReportConfig(PipelineConfig):
    models: list[str] = Field(min_length=1)
    temperature: float = Field(ge=0, le=2)
    max_tokens: PositiveInt


class ApiConfig(BaseModel):
    """HTTP service settings.

    ``artifacts_dir`` follows the pipeline layout: the service loads ``strength/model`` and
    ``cracks/model`` below it and reports a missing model as an unavailable component.
    """

    model_config = ConfigDict(extra="forbid")

    host: str
    port: int = Field(ge=1, le=65535)
    artifacts_dir: Path
    cracks_backend: Literal["onnx", "torch"] = "onnx"
    inference_threads: PositiveInt | None = 2
    max_upload_bytes: PositiveInt = 10 * 1024 * 1024
    max_image_pixels: PositiveInt = 40_000_000
    gradcam_max_side: int = Field(default=512, ge=64, le=2048)
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:8501"])
    report_config: Path = Path("configs/report.yaml")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    @property
    def strength_model_dir(self) -> Path:
        return project_path(self.artifacts_dir) / "strength" / "model"

    @property
    def strength_metrics_path(self) -> Path:
        """Test metrics written by ``strength evaluate``; shown only if the version matches."""
        return project_path(self.artifacts_dir) / "strength" / "test_metrics.json"

    @property
    def cracks_model_dir(self) -> Path:
        return project_path(self.artifacts_dir) / "cracks" / "model"


class Settings(BaseSettings):
    """Local service settings with environment variables overriding .env."""

    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    ollama_host: HttpUrl = HttpUrl("http://localhost:11434")
    llm_mode: Literal["live", "replay"] = Field(default="live", validation_alias="BAULENS_LLM_MODE")


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys instead of keeping the last one."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
        seen: set[Any] = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found duplicate key {key!r}",
                    key_node.start_mark,
                )
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def load_config[ConfigT: BaseModel](path: Path, model: type[ConfigT]) -> ConfigT:
    """Read a checkout-relative or absolute YAML file and validate it against a model.

    Duplicate keys raise ``yaml.YAMLError`` with the line of the repeated key.
    """
    with project_path(path).open(encoding="utf-8") as config_file:
        return model.model_validate(yaml.load(config_file, Loader=_UniqueKeyLoader))
