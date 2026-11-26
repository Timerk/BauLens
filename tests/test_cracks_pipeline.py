"""End-to-end smoke run of the crack pipeline on synthetic images (CPU, no network)."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image
from typer.testing import CliRunner

from baulens.cli import app
from baulens.config import CracksConfig, load_config
from baulens.cracks.inference import CrackClassifier, overlay_heatmap
from baulens.data.synthetic import make_synthetic_cracks
from baulens.schemas import CrackPrediction

runner = CliRunner()


@dataclass(frozen=True)
class SmokeRun:
    config: CracksConfig
    config_path: Path
    data_dir: Path
    outputs: dict[str, dict[str, object]]

    @property
    def root(self) -> Path:
        return self.config.artifacts_dir

    @property
    def model_dir(self) -> Path:
        return self.root / "model"


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory: pytest.TempPathFactory) -> SmokeRun:
    base = tmp_path_factory.mktemp("cracks")
    data_dir = make_synthetic_cracks(base / "data", per_class=16, image_size=64, seed=42)
    settings = yaml.safe_load(Path("configs/smoke/cracks.yaml").read_text(encoding="utf-8"))
    settings |= {"data_dir": str(data_dir), "artifacts_dir": str(base / "artifacts")}
    config_path = base / "cracks.yaml"
    config_path.write_text(yaml.safe_dump(settings), encoding="utf-8")
    outputs: dict[str, dict[str, object]] = {}
    for command in ("split", "train", "evaluate", "export", "benchmark"):
        result = runner.invoke(app, ["cracks", command, "--config", str(config_path)])
        assert result.exit_code == 0, (command, result.output, result.exception)
        outputs[command] = json.loads(result.stdout[result.stdout.index("{") :])
    return SmokeRun(load_config(config_path, CracksConfig), config_path, data_dir, outputs)


@pytest.mark.smoke
def test_split_is_grouped_and_complete(smoke_run: SmokeRun) -> None:
    leakage = smoke_run.outputs["split"]
    assert leakage["images"] == 32
    manifest = (smoke_run.root / "splits" / "manifest.csv").read_text(encoding="utf-8")
    assert manifest.count("\n") == 33
    for row in leakage["sensitivity"]:  # type: ignore[attr-defined]
        assert row["cross_split_pairs_grouped"] == 0 or row["max_distance"] > 4


@pytest.mark.smoke
def test_training_writes_model_history_and_metadata(smoke_run: SmokeRun) -> None:
    assert (smoke_run.model_dir / "model.pt").exists()
    assert (smoke_run.root / "history.csv").exists()
    assert (smoke_run.root / "figures" / "learning_curves.png").exists()
    metadata = json.loads((smoke_run.model_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["architecture"] == "resnet18"
    assert metadata["pretrained_weights"] is None
    assert len(metadata["data_checksum"]) == 64
    assert 0 < metadata["threshold"] < 1


@pytest.mark.smoke
def test_evaluation_reports_metrics_with_intervals(smoke_run: SmokeRun) -> None:
    evaluation = json.loads((smoke_run.root / "evaluation.json").read_text(encoding="utf-8"))
    assert set(evaluation["test_ci95"]) >= {"f1", "roc_auc", "precision", "recall"}
    assert 0 <= evaluation["test"]["accuracy"] <= 1
    assert len(evaluation["robustness"]) == 13
    for name in ("confusion_matrix_test", "reliability_test", "gradcam_examples", "error_gallery"):
        assert (smoke_run.root / "figures" / f"{name}.png").exists()


@pytest.mark.smoke
def test_onnx_export_matches_pytorch(smoke_run: SmokeRun) -> None:
    export = smoke_run.outputs["export"]
    assert export["labels_identical"] is True
    assert float(export["max_abs_diff_probability"]) <= 1e-4  # type: ignore[arg-type]


@pytest.mark.smoke
def test_benchmark_covers_both_backends(smoke_run: SmokeRun) -> None:
    runs = smoke_run.outputs["benchmark"]["runs"]
    assert {(run["backend"], run["batch_size"]) for run in runs} == {  # type: ignore[attr-defined]
        ("torch", 1),
        ("torch", 32),
        ("onnx", 1),
        ("onnx", 32),
    }


@pytest.mark.smoke
def test_inference_api_backends_agree(smoke_run: SmokeRun) -> None:
    image = Image.open(next((smoke_run.data_dir / "Positive").glob("*.jpg"))).convert("RGB")
    onnx_model = CrackClassifier.load(smoke_run.model_dir, "onnx")
    torch_model = CrackClassifier.load(smoke_run.model_dir, "torch")
    prediction = onnx_model.predict(image)
    assert isinstance(prediction, CrackPrediction)
    assert prediction.threshold == onnx_model.threshold
    assert prediction.model_version == onnx_model.model_version
    assert abs(prediction.crack_probability - torch_model.predict(image).crack_probability) < 1e-4
    heatmap = onnx_model.gradcam(image)
    assert heatmap.shape == (64, 64)
    assert heatmap.min() >= 0 and heatmap.max() <= 1
    overlay = overlay_heatmap(image, heatmap)
    assert overlay.size == image.size
    assert np.asarray(overlay).dtype == np.uint8
