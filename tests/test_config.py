from pathlib import Path

import pytest
import yaml
from pydantic import BaseModel, ValidationError

from baulens.config import (
    ApiConfig,
    CracksConfig,
    PipelineConfig,
    ReportConfig,
    Settings,
    StrengthConfig,
    load_config,
)

CONFIG_MODELS = [
    ("strength", StrengthConfig),
    ("cracks", CracksConfig),
    ("report", ReportConfig),
    ("api", ApiConfig),
]


@pytest.mark.parametrize(("module", "model"), CONFIG_MODELS)
def test_load_config(module: str, model: type[BaseModel]) -> None:
    assert isinstance(load_config(Path("configs") / f"{module}.yaml", model), model)


@pytest.mark.smoke
@pytest.mark.parametrize(("module", "model"), CONFIG_MODELS)
def test_load_smoke_config(module: str, model: type[BaseModel]) -> None:
    assert isinstance(load_config(Path("configs/smoke") / f"{module}.yaml", model), model)


def test_load_config_rejects_unknown_and_invalid_values(tmp_path: Path) -> None:
    config = tmp_path / "api.yaml"
    config.write_text(
        "host: localhost\nport: 70000\nartifacts_dir: artifacts\nextra_key: true\n",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError) as exc:
        load_config(config, ApiConfig)
    assert {error["loc"] for error in exc.value.errors()} == {("port",), ("extra_key",)}


def test_settings_load_ollama_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_HOST", "http://ollama:11434")
    monkeypatch.setenv("BAULENS_LLM_MODE", "replay")
    settings = Settings()
    assert settings.ollama_host.host == "ollama"
    assert settings.llm_mode == "replay"


def test_load_config_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.yaml"
    path.write_text("seed: 1\nartifacts_dir: out\nseed: 2\n", encoding="utf-8")
    with pytest.raises(yaml.YAMLError, match="duplicate key 'seed'"):
        load_config(path, PipelineConfig)
