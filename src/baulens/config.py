"""Typed configuration for pipeline runs."""

from pathlib import Path
from typing import Any, Literal

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
    calibration_fraction: float = Field(default=0.25, gt=0, lt=1)
    inner_splits: int = Field(default=3, ge=2)
    bootstrap_resamples: int = Field(default=2000, ge=2000)
    threads: PositiveInt = 2
    results_path: Path = Path("docs/results/strength.md")
    figures_dir: Path = Path("docs/figures/strength")


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
