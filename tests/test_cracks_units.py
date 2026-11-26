"""Fast unit tests for the crack module's metrics, splitting, Grad-CAM and corruptions."""

import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from baulens.config import CracksConfig
from baulens.cracks.data import preprocess
from baulens.cracks.gradcam import GradCAM, overlay_heatmap
from baulens.cracks.metrics import (
    binary_metrics,
    bootstrap_intervals,
    calibration,
    choose_threshold,
    threshold_plateau,
)
from baulens.cracks.model import build_model, configure_trainable, gradcam_layer, resolve_device
from baulens.cracks.robustness import default_corruptions
from baulens.cracks.splits import (
    candidate_pairs,
    confirmed,
    data_checksum,
    duplicate_groups,
    ensure_splits,
    group_stratified_split,
    index_images,
)


def test_threshold_prefers_value_nearest_half_on_a_plateau() -> None:
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.8, 0.9])
    assert choose_threshold(labels, probabilities, "f1") == 0.5
    assert choose_threshold(labels, probabilities, "youden") == 0.5


def test_threshold_moves_to_the_separating_value() -> None:
    labels = np.array([0, 0, 0, 1, 1])
    probabilities = np.array([0.6, 0.65, 0.7, 0.8, 0.95])
    threshold = choose_threshold(labels, probabilities, "f1")
    assert 0.7 < threshold <= 0.8


def test_binary_metrics_counts() -> None:
    labels = np.array([0, 0, 1, 1, 1])
    probabilities = np.array([0.1, 0.7, 0.2, 0.8, 0.9])
    metrics = binary_metrics(labels, probabilities, 0.5)
    assert (metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"]) == (2, 1, 1, 1)
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == pytest.approx(2 / 3)
    assert metrics["accuracy"] == pytest.approx(3 / 5)


def test_bootstrap_interval_contains_point_estimate() -> None:
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, 400)
    probabilities = np.clip(labels * 0.6 + rng.normal(0.2, 0.2, 400), 0, 1)
    point = binary_metrics(labels, probabilities, 0.5)
    intervals = bootstrap_intervals(labels, probabilities, 0.5, resamples=200, seed=1)
    for name in ("f1", "accuracy", "roc_auc"):
        low, high = intervals[name]
        assert low <= point[name] <= high


def test_calibration_of_perfectly_calibrated_bins() -> None:
    labels = np.array([0] * 9 + [1] + [1] * 9 + [0])
    probabilities = np.array([0.1] * 10 + [0.9] * 10)
    result = calibration(labels, probabilities, bins=10)
    assert result.ece == pytest.approx(0.0, abs=1e-12)
    assert sum(result.counts) == 20


def test_duplicates_are_grouped_and_kept_in_one_split(tiny_crack_images: Path) -> None:
    source = tiny_crack_images / "Positive" / "00000.jpg"
    shutil.copy(source, tiny_crack_images / "Positive" / "00100.jpg")
    with Image.open(source) as image:
        image.save(tiny_crack_images / "Positive" / "00101.jpg", quality=70)
    index, thumbs = index_images(tiny_crack_images)
    pairs = candidate_pairs(index, thumbs, max_distance=10)
    mask = confirmed(pairs, max_distance=10, min_correlation=0.9)
    groups = duplicate_groups(len(index), pairs.subset(mask))
    paths = index["path"].tolist()
    members = [paths.index(f"Positive/{name}.jpg") for name in ("00000", "00100", "00101")]
    assert len({groups[i] for i in members}) == 1
    assert pairs.identical.sum() >= 1
    splits = group_stratified_split(
        index["label"].to_numpy(), groups, val_fraction=0.2, test_fraction=0.2, seed=3
    )
    assert len({splits[i] for i in members}) == 1
    assert set(splits) == {"train", "val", "test"}


def test_group_split_is_stratified_and_reproducible() -> None:
    labels = np.array([0] * 500 + [1] * 500)
    groups = np.arange(1000)
    first = group_stratified_split(labels, groups, val_fraction=0.15, test_fraction=0.15, seed=7)
    second = group_stratified_split(labels, groups, val_fraction=0.15, test_fraction=0.15, seed=7)
    assert (first == second).all()
    for label in (0, 1):
        share = np.mean(first[labels == label] == "test")
        assert share == pytest.approx(0.15, abs=0.01)


def test_gradcam_returns_normalised_maps() -> None:
    torch.manual_seed(0)
    model = build_model("resnet18", pretrained=False).eval()
    images = torch.randn(2, 3, 64, 64)
    with GradCAM(model, gradcam_layer(model, "resnet18")) as cam:
        maps = cam(images)
    assert maps.shape == (2, 64, 64)
    assert maps.min() >= 0 and maps.max() <= 1
    with torch.no_grad():  # the hook must not break ordinary inference afterwards
        model(images)


def test_gradcam_layer_receives_gradient_with_frozen_backbone() -> None:
    model = build_model("resnet18", pretrained=False).eval()
    configure_trainable(model, "resnet18", finetune_blocks=0)
    with GradCAM(model, gradcam_layer(model, "resnet18")) as cam:
        maps = cam(torch.randn(1, 3, 64, 64))
    assert np.isfinite(maps).all()


def test_efficientnet_head_has_one_logit() -> None:
    model = build_model("efficientnet_b0", pretrained=False).eval()
    with torch.no_grad():
        assert model(torch.zeros(1, 3, 64, 64)).shape == (1, 1)
    trainable = configure_trainable(model, "efficientnet_b0", finetune_blocks=1)
    assert len(trainable) == 2
    assert isinstance(gradcam_layer(model, "efficientnet_b0"), nn.Module)


def test_overlay_and_preprocess_shapes() -> None:
    image = Image.new("RGB", (227, 227), (120, 120, 120))
    array = preprocess(image, 128)
    assert array.shape == (3, 128, 128) and array.dtype == np.float32
    overlay = overlay_heatmap(image, np.linspace(0, 1, 16).reshape(4, 4))
    assert overlay.size == (227, 227)


def test_corruptions_preserve_size_and_are_deterministic() -> None:
    rng = np.random.default_rng(0)
    image = Image.fromarray(rng.integers(0, 255, (227, 227, 3), dtype=np.uint8))
    for corruption in default_corruptions(seed=5):
        first, second = corruption(image, 3), corruption(image, 3)
        assert first.size == image.size and first.mode == "RGB"
        assert np.array_equal(np.asarray(first), np.asarray(second))


def test_resolve_device_cpu() -> None:
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type in {"cpu", "cuda"}


def test_split_cache_detects_replaced_contents(tiny_crack_images: Path, tmp_path: Path) -> None:
    config = CracksConfig(
        seed=1,
        data_dir=tiny_crack_images,
        artifacts_dir=tmp_path / "run",
        architecture="resnet18",
        image_size=64,
        batch_size=4,
        epochs=1,
        pretrained=False,
    )
    first = ensure_splits(config)
    held_out = first[first["split"] == "test"].iloc[0]
    train = first[(first["split"] == "train") & (first["label"] == held_out["label"])].iloc[0]
    (tiny_crack_images / held_out["path"]).write_bytes(
        (tiny_crack_images / train["path"]).read_bytes()
    )
    second = ensure_splits(config).set_index("path")
    assert second.loc[held_out["path"], "sha256"] == second.loc[train["path"], "sha256"]
    assert second.loc[held_out["path"], "group"] == second.loc[train["path"], "group"]
    assert second.loc[held_out["path"], "split"] == second.loc[train["path"], "split"]
    assert data_checksum(second.reset_index()) != data_checksum(first)


def test_group_bootstrap_resamples_whole_groups() -> None:
    labels = np.array([1, 1, 1, 1, 0, 0])
    probabilities = np.array([0.9, 0.9, 0.9, 0.2, 0.1, 0.1])
    groups = np.array([0, 0, 0, 1, 2, 3])  # three copies of one confident positive
    grouped = bootstrap_intervals(labels, probabilities, 0.5, resamples=400, seed=0, groups=groups)
    single = bootstrap_intervals(labels, probabilities, 0.5, resamples=400, seed=0)
    assert grouped["recall"][0] <= single["recall"][0]
    for low, high in grouped.values():
        assert low <= high


def test_threshold_plateau_and_lower_edge_tie_break() -> None:
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.7, 0.85, 0.95])
    low, high = threshold_plateau(labels, probabilities, "f1")
    assert (low, high) == (0.701, 0.85)
    assert choose_threshold(labels, probabilities, "f1") == low
