"""End-to-end smoke run of the crack pipeline on synthetic images (CPU, no network)."""

import hashlib
import io
import json
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from filelock import FileLock, Timeout
from PIL import Image
from typer.testing import CliRunner

from baulens import artifacts
from baulens.artifacts import ArtifactError
from baulens.cli import app
from baulens.config import CracksConfig, load_config
from baulens.cracks import export as export_module
from baulens.cracks.inference import CrackClassifier, overlay_heatmap
from baulens.cracks.metadata import load_published
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
    confirmed = leakage["confirmed"]
    assert confirmed["cross_split_pairs_grouped"] == 0  # type: ignore[index]
    assert set(leakage["split_counts"]) == {"train", "val", "test"}  # type: ignore[arg-type]


@pytest.mark.smoke
def test_training_writes_model_history_and_metadata(smoke_run: SmokeRun) -> None:
    published = load_published(smoke_run.model_dir, require_onnx=True)
    assert (smoke_run.root / "history.csv").exists()
    assert (smoke_run.root / "figures" / "learning_curves.png").exists()
    assert not (smoke_run.root / "checkpoint_in_progress.pt").exists()
    metadata = published.metadata
    assert metadata.architecture == "resnet18"
    assert metadata.pretrained_weights is None
    assert metadata.weights_sha256 == hashlib.sha256(published.weights).hexdigest()
    assert metadata.split_settings is not None and metadata.split_settings["seed"] == 42
    assert metadata.threshold is not None and 0 < metadata.threshold < 1
    # train, evaluate and export each published a generation; two are kept.
    pointer = json.loads((smoke_run.model_dir / "current.json").read_text(encoding="utf-8"))
    assert len(pointer["generations"]) == 2
    assert len(list((smoke_run.model_dir / "generations").iterdir())) == 2


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


def _copy_run(smoke_run: SmokeRun, destination: Path, **overrides: object) -> Path:
    """Copy the trained run to ``destination`` and write a config pointing at the copy."""
    shutil.copytree(smoke_run.root, destination)
    settings = yaml.safe_load(smoke_run.config_path.read_text(encoding="utf-8"))
    settings |= {"artifacts_dir": str(destination)} | overrides
    path = destination.parent / f"{destination.name}.yaml"
    path.write_text(yaml.safe_dump(settings), encoding="utf-8")
    return path


@pytest.mark.smoke
def test_evaluate_export_and_benchmark_refuse_a_different_split(
    smoke_run: SmokeRun, tmp_path: Path
) -> None:
    config_path = _copy_run(smoke_run, tmp_path / "reseeded", seed=43)
    for command, message in (
        ("evaluate", "Evaluation refused"),
        ("export", "Export error"),
        ("benchmark", "Benchmark refused"),
    ):
        result = runner.invoke(app, ["cracks", command, "--config", str(config_path)])
        assert result.exit_code == 1, command
        assert message in result.stderr
        assert "split checksum" in result.stderr


def _snapshot(model_dir: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(model_dir)): path.read_bytes()
        for path in sorted(model_dir.rglob("*"))
        if path.is_file()
    }


@pytest.mark.smoke
def test_failed_parity_keeps_the_published_model(
    smoke_run: SmokeRun, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _copy_run(smoke_run, tmp_path / "failing")
    config = load_config(config_path, CracksConfig)
    model_dir = config.artifacts_dir / "model"
    before = _snapshot(model_dir)
    monkeypatch.setattr(export_module, "parity", lambda *args: (1.0, 1.0, False))
    with pytest.raises(export_module.ExportError, match="left unchanged"):
        export_module.export(config)
    assert _snapshot(model_dir) == before
    assert CrackClassifier.load(model_dir, "onnx").threshold > 0


@pytest.mark.smoke
def test_failed_pointer_write_keeps_the_previous_generation(
    smoke_run: SmokeRun, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _copy_run(smoke_run, tmp_path / "interrupted")
    config = load_config(config_path, CracksConfig)
    model_dir = config.artifacts_dir / "model"
    before = _snapshot(model_dir)
    real_write = artifacts.write_json_atomic

    def failing_write(destination: Path, content: object) -> None:
        if destination.name == artifacts.CURRENT_FILE:
            raise OSError("simulated crash while switching the pointer")
        real_write(destination, content)

    monkeypatch.setattr(artifacts, "write_json_atomic", failing_write)
    with pytest.raises(OSError, match="simulated crash"):
        export_module.export(config)
    assert _snapshot(model_dir) == before
    assert CrackClassifier.load(model_dir, "onnx").predict(
        Image.new("RGB", (64, 64), (128, 128, 128))
    )


@pytest.mark.smoke
def test_tampered_generation_files_are_refused(smoke_run: SmokeRun, tmp_path: Path) -> None:
    config_path = _copy_run(smoke_run, tmp_path / "tampered")
    model_dir = load_config(config_path, CracksConfig).artifacts_dir / "model"
    generation = load_published(model_dir).directory
    graph = generation / "model.onnx"
    original = graph.read_bytes()
    graph.write_bytes(original + b"\0")
    with pytest.raises(ArtifactError, match="manifest"):
        CrackClassifier.load(model_dir, "onnx")
    # A swapped graph with a consistently updated manifest still disagrees with the metadata.
    manifest_path = generation / artifacts.MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["model.onnx"] = hashlib.sha256(graph.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ArtifactError, match=r"onnx.sha256"):
        CrackClassifier.load(model_dir, "onnx")
    # Same for swapped weights.
    graph.write_bytes(original)
    weights = generation / "model.pt"
    weights.write_bytes(weights.read_bytes() + b"\0")
    manifest["model.onnx"] = hashlib.sha256(original).hexdigest()
    manifest["model.pt"] = hashlib.sha256(weights.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ArtifactError, match="weights_sha256"):
        CrackClassifier.load(model_dir, "torch")


def _legacy_run(smoke_run: SmokeRun, destination: Path, weights: bytes | None = None) -> Path:
    """A run directory in the pre-generation layout, built from the published smoke model."""
    config_path = _copy_run(smoke_run, destination)
    model_dir = load_config(config_path, CracksConfig).artifacts_dir / "model"
    published = load_published(model_dir)
    shutil.rmtree(model_dir)
    model_dir.mkdir()
    metadata = published.metadata.model_copy(update={"weights_sha256": None})
    (model_dir / "model.pt").write_bytes(published.weights if weights is None else weights)
    (model_dir / "metadata.json").write_bytes(metadata.dump())
    return config_path


@pytest.mark.smoke
def test_legacy_checkpoint_migrates_only_with_reproduced_validation_loss(
    smoke_run: SmokeRun, tmp_path: Path
) -> None:
    config_path = _legacy_run(smoke_run, tmp_path / "legacy")
    result = runner.invoke(app, ["cracks", "migrate", "--config", str(config_path)])
    assert result.exit_code == 0, result.output
    model_dir = load_config(config_path, CracksConfig).artifacts_dir / "model"
    assert CrackClassifier.load(model_dir, "torch").threshold > 0
    assert not (model_dir / "model.pt").exists()

    # Weights that are valid but not the trained ones (changed head bias) are refused.
    state = torch.load(
        io.BytesIO(load_published(model_dir).weights), map_location="cpu", weights_only=True
    )
    state["fc.bias"] = state["fc.bias"] + 20.0
    buffer = io.BytesIO()
    torch.save(state, buffer)
    swapped = _legacy_run(smoke_run, tmp_path / "swapped", weights=buffer.getvalue())
    result = runner.invoke(app, ["cracks", "migrate", "--config", str(swapped)])
    assert result.exit_code == 1
    assert "validation loss" in result.stderr
    swapped_dir = load_config(swapped, CracksConfig).artifacts_dir / "model"
    assert not (swapped_dir / "current.json").exists()


@pytest.mark.smoke
def test_gradcam_is_safe_under_concurrent_requests(smoke_run: SmokeRun) -> None:
    classifier = CrackClassifier.load(smoke_run.model_dir, "onnx")
    images = [
        Image.open(path).convert("RGB")
        for path in (
            next((smoke_run.data_dir / "Positive").glob("*.jpg")),
            sorted((smoke_run.data_dir / "Positive").glob("*.jpg"))[-1],
        )
    ]
    expected = [classifier.gradcam(image) for image in images]
    assert not np.array_equal(expected[0], expected[1])
    barrier = threading.Barrier(2)
    failures: list[str] = []

    def worker(position: int) -> None:
        barrier.wait()
        for _ in range(25):
            heatmap = classifier.gradcam(images[position])
            if not np.allclose(heatmap, expected[position], atol=1e-5):
                failures.append(f"image {position} received another request's map")
            classifier.predict(images[1 - position])

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert failures == []


def test_resumable_training_matches_uninterrupted_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from baulens.cracks import train as train_module
    from baulens.cracks.splits import split_files

    config = load_config(Path("configs/smoke/cracks.yaml"), CracksConfig).model_copy(
        update={
            "data_dir": make_synthetic_cracks(
                tmp_path / "data", per_class=12, image_size=64, seed=42
            ),
            "artifacts_dir": tmp_path / "resumed",
            "architecture": "efficientnet_b0",
            "epochs": 2,
            "resumable": True,
        }
    )
    save = train_module._save_checkpoint
    final_states: dict[Path, train_module.Checkpoint] = {}
    interrupted = False

    def capture_checkpoint(path: Path, state: train_module.Checkpoint) -> None:
        nonlocal interrupted
        save(path, state)
        epoch = len(state["history"])
        if epoch == 2 and path.parent == config.artifacts_dir and not interrupted:
            interrupted = True
            raise InterruptedError("simulated interruption")
        if epoch == 3:
            final_states[path.parent] = torch.load(path, map_location="cpu", weights_only=True)

    monkeypatch.setattr(train_module, "_save_checkpoint", capture_checkpoint)
    with pytest.raises(InterruptedError):
        train_module.train(config)
    checkpoint = config.artifacts_dir / "training_checkpoint.pt"
    original = checkpoint.read_bytes()
    state: train_module.Checkpoint = torch.load(io.BytesIO(original), weights_only=True)

    with FileLock(str(config.artifacts_dir / ".cracks-training.lock")), pytest.raises(Timeout):
        train_module.train(config)
    with pytest.raises(ValueError, match=r"checkpoint does not match.*config"):
        train_module.train(config.model_copy(update={"batch_size": 4}))

    torch.save(state | {"format_version": -1}, checkpoint)
    with pytest.raises(ValueError, match="Unsupported training checkpoint format"):
        train_module.train(config)
    identity = state["identity"]
    for field, value in (
        ("source_fingerprint", "different source"),
        ("accelerator", identity["accelerator"] | {"device_name": "different accelerator"}),
    ):
        torch.save(state | {"identity": identity | {field: value}}, checkpoint)
        with pytest.raises(ValueError, match=f"checkpoint does not match.*{field}"):
            train_module.train(config)
    checkpoint.write_bytes(original)

    manifest = split_files(config).manifest
    original_manifest = manifest.read_bytes()
    frame = pd.read_csv(manifest, dtype={"phash": str})
    for changed_frame in (frame.iloc[::-1], frame.assign(label=1 - frame["label"])):
        changed_frame.to_csv(manifest, index=False)
        with pytest.raises(
            ValueError, match=r"checkpoint does not match.*ordered_samples_checksum"
        ):
            train_module.train(config)
    manifest.write_bytes(original_manifest)

    resumed = train_module.train(config)
    control_config = config.model_copy(update={"artifacts_dir": tmp_path / "uninterrupted"})
    uninterrupted = train_module.train(control_config)
    assert not checkpoint.exists()
    assert resumed.epochs_run == uninterrupted.epochs_run == 3
    assert resumed.best_epoch == uninterrupted.best_epoch
    assert resumed.best_val_loss == uninterrupted.best_val_loss
    resumed_state = final_states[config.artifacts_dir]
    control_state = final_states[control_config.artifacts_dir]
    assert resumed_state["model"].keys() == control_state["model"].keys()
    assert all(
        torch.equal(tensor, control_state["model"][name])
        for name, tensor in resumed_state["model"].items()
    )
    history_fields = ("phase", "epoch", "learning_rate", "train_loss", "val_loss")
    assert [tuple(row[key] for key in history_fields) for row in resumed_state["history"]] == [
        tuple(row[key] for key in history_fields) for row in control_state["history"]
    ]

    resumed_bundle = load_published(Path(resumed.model_dir))
    control_bundle = load_published(Path(uninterrupted.model_dir))
    assert resumed_bundle.metadata.training["accelerator"] == identity["accelerator"]
    resumed_weights = torch.load(io.BytesIO(resumed_bundle.weights), weights_only=True)
    control_weights = torch.load(io.BytesIO(control_bundle.weights), weights_only=True)
    assert resumed_weights.keys() == control_weights.keys()
    assert all(
        torch.equal(tensor, control_weights[name]) for name, tensor in resumed_weights.items()
    )
