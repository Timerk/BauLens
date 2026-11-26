"""Typed configuration for pipeline runs."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, PositiveInt
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


class CracksConfig(PipelineConfig):
    data_dir: Path
    architecture: Literal["resnet18", "efficientnet_b0"]
    image_size: PositiveInt
    batch_size: PositiveInt
    epochs: PositiveInt
    pretrained: bool


class ReportConfig(PipelineConfig):
    models: list[str] = Field(min_length=1)
    temperature: float = Field(ge=0, le=2)
    max_tokens: PositiveInt


class ApiConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    host: str
    port: int = Field(ge=1, le=65535)
    artifacts_dir: Path


class Settings(BaseSettings):
    """Local service settings with environment variables overriding .env."""

    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    ollama_host: HttpUrl = HttpUrl("http://localhost:11434")
    llm_mode: Literal["live", "replay"] = Field(default="live", validation_alias="BAULENS_LLM_MODE")


def load_config[ConfigT: BaseModel](path: Path, model: type[ConfigT]) -> ConfigT:
    """Read a checkout-relative or absolute YAML file and validate it against a model."""
    with project_path(path).open(encoding="utf-8") as config_file:
        return model.model_validate(yaml.safe_load(config_file))
