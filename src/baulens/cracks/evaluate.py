"""Validation-chosen threshold, test metrics with bootstrap CIs, calibration, robustness probe,
Grad-CAM galleries and the error gallery."""

import json
import logging
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from PIL import Image
from torch import nn

from baulens.config import CracksConfig
from baulens.cracks import figures
from baulens.cracks.data import (
    CrackImageDataset,
    EvalTransform,
    Perturbation,
    make_loader,
    stratified_head,
)
from baulens.cracks.gradcam import GradCAM, overlay_heatmap
from baulens.cracks.metadata import ModelMetadata, run_paths
from baulens.cracks.metrics import (
    binary_metrics,
    bootstrap_intervals,
    calibration,
    choose_threshold,
)
from baulens.cracks.model import build_model, gradcam_layer
from baulens.cracks.robustness import default_corruptions
from baulens.cracks.splits import ensure_splits
from baulens.cracks.train import configure_runtime
from baulens.paths import project_path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvaluationResult:
    threshold: float
    threshold_strategy: str
    val: dict[str, float]
    test: dict[str, float]
    test_ci95: dict[str, tuple[float, float]]
    test_ece: float
    robustness: list[dict[str, Any]]


def load_torch_model(config: CracksConfig, device: torch.device) -> nn.Module:
    paths = run_paths(config)
    model = build_model(config.architecture, pretrained=False)
    state = torch.load(paths.model, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval()


@torch.no_grad()
def predict_probabilities(
    model: nn.Module,
    dataset: CrackImageDataset,
    config: CracksConfig,
    device: torch.device,
) -> NDArray[np.float64]:
    loader = make_loader(
        dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        seed=config.seed,
    )
    outputs = [torch.sigmoid(model(images.to(device)).squeeze(1)).cpu() for images, _ in loader]
    return torch.cat(outputs).double().numpy()


def _rounded(metrics: dict[str, float]) -> dict[str, float]:
    return {key: round(value, 6) for key, value in metrics.items()}


def _robustness(
    model: nn.Module,
    test: pd.DataFrame,
    threshold: float,
    config: CracksConfig,
    device: torch.device,
) -> list[dict[str, Any]]:
    subset = stratified_head(test, config.robustness_max_images, config.seed)
    data_dir = project_path(config.data_dir)
    labels = subset["label"].to_numpy(dtype=np.int64)
    corruptions: list[tuple[str, Perturbation | None]] = [("none", None)]
    corruptions += [(c.label, c) for c in default_corruptions(config.seed)]
    rows: list[dict[str, Any]] = []
    for label, corruption in corruptions:
        dataset = CrackImageDataset.from_frame(
            data_dir, subset, EvalTransform(config.image_size), corruption
        )
        probabilities = predict_probabilities(model, dataset, config, device)
        metrics = binary_metrics(labels, probabilities, threshold)
        rows.append(
            {"corruption": label, "images": len(subset)}
            | {
                k: round(metrics[k], 4)
                for k in ("accuracy", "precision", "recall", "f1", "roc_auc")
            }
        )
        logger.info("robustness %s: F1 %.4f", label, metrics["f1"])
    clean = rows[0]
    for row in rows:
        row["delta_f1"] = round(row["f1"] - clean["f1"], 4)
        row["delta_recall"] = round(row["recall"] - clean["recall"], 4)
    return rows


def _gallery(
    model: nn.Module,
    config: CracksConfig,
    device: torch.device,
    frame: pd.DataFrame,
    title: str,
    name: str,
) -> None:
    """Original patch above its Grad-CAM overlay for each selected test image."""
    data_dir = project_path(config.data_dir)
    dataset = CrackImageDataset.from_frame(data_dir, frame, EvalTransform(config.image_size))
    tiles: list[Image.Image] = []
    captions: list[str] = []
    if len(frame):
        batch = torch.stack([dataset[i][0] for i in range(len(frame))]).to(device)
        with GradCAM(model, gradcam_layer(model, config.architecture)) as cam:
            heatmaps = cam(batch)
        for i, (_, row) in enumerate(frame.iterrows()):
            image = dataset.load_image(i)
            overlay = overlay_heatmap(image, heatmaps[i])
            stacked = Image.new("RGB", (image.width, image.height * 2))
            stacked.paste(image, (0, 0))
            stacked.paste(overlay, (0, image.height))
            tiles.append(stacked)
            truth = "crack" if row["label"] == 1 else "no crack"
            captions.append(
                f"{row['path'].split('/')[-1]} ({truth})\nP(crack)={row['probability']:.3f}"
            )
    figures.gallery(tiles, captions, title, figures.targets(config, name))


def evaluate(config: CracksConfig) -> EvaluationResult:
    """Pick the threshold on validation, apply it once to test and write all evaluation output."""
    device = configure_runtime(config)
    paths = run_paths(config)
    metadata = ModelMetadata.read(paths.model_dir)
    manifest = ensure_splits(config)
    data_dir = project_path(config.data_dir)
    model = load_torch_model(config, device)
    transform = EvalTransform(config.image_size)

    frames: dict[str, pd.DataFrame] = {}
    for split in ("val", "test"):
        frame = manifest[manifest["split"] == split].reset_index(drop=True)
        dataset = CrackImageDataset.from_frame(data_dir, frame, transform)
        frames[split] = frame.assign(
            probability=predict_probabilities(model, dataset, config, device)
        )
        logger.info("predicted %d %s images", len(frame), split)
    predictions = pd.concat([frames["val"], frames["test"]]).assign(
        split=["val"] * len(frames["val"]) + ["test"] * len(frames["test"])
    )
    predictions[["path", "label", "split", "probability"]].to_csv(
        paths.root / "predictions.csv", index=False, lineterminator="\n"
    )

    val_labels = frames["val"]["label"].to_numpy(dtype=np.int64)
    val_probs = frames["val"]["probability"].to_numpy()
    threshold = choose_threshold(val_labels, val_probs, config.threshold_strategy)
    test_labels = frames["test"]["label"].to_numpy(dtype=np.int64)
    test_probs = frames["test"]["probability"].to_numpy()
    val_metrics = binary_metrics(val_labels, val_probs, threshold)
    test_metrics = binary_metrics(test_labels, test_probs, threshold)
    intervals = bootstrap_intervals(
        test_labels,
        test_probs,
        threshold,
        resamples=config.bootstrap_resamples,
        seed=config.seed,
    )
    test_calibration = calibration(test_labels, test_probs, config.calibration_bins)
    val_calibration = calibration(val_labels, val_probs, config.calibration_bins)

    figures.confusion_matrix(
        test_metrics,
        figures.targets(config, "confusion_matrix_test.png"),
        f"Test set, threshold {threshold:.3f}",
    )
    figures.roc_pr(test_labels, test_probs, figures.targets(config, "roc_pr_test.png"))
    figures.reliability(
        test_calibration, figures.targets(config, "reliability_test.png"), "Test calibration"
    )

    robustness = _robustness(model, frames["test"], threshold, config, device)
    figures.robustness(robustness, figures.targets(config, "robustness.png"))

    test = frames["test"]
    predicted = test["probability"] >= threshold
    positive = test["label"] == 1
    k = config.gallery_size
    examples = pd.concat(
        [
            test[predicted & positive].sample(
                min(k // 2, int((predicted & positive).sum())), random_state=config.seed
            ),
            test[~predicted & ~positive].sample(
                min(k // 2, int((~predicted & ~positive).sum())), random_state=config.seed
            ),
        ]
    )
    _gallery(
        model,
        config,
        device,
        examples,
        "Grad-CAM on random true positives and true negatives (top: patch, bottom: overlay)",
        "gradcam_examples.png",
    )
    errors = pd.concat(
        [
            test[predicted & ~positive].sort_values("probability", ascending=False).head(k // 2),
            test[~predicted & positive].sort_values("probability").head(k // 2),
        ]
    )
    _gallery(
        model,
        config,
        device,
        errors,
        "Hardest test errors: false positives (highest P) then false negatives (lowest P)",
        "error_gallery.png",
    )
    test[(predicted != positive)][["path", "label", "probability"]].to_csv(
        paths.root / "test_errors.csv", index=False, lineterminator="\n"
    )

    result = EvaluationResult(
        threshold=threshold,
        threshold_strategy=config.threshold_strategy,
        val=_rounded(val_metrics),
        test=_rounded(test_metrics),
        test_ci95={k: (round(lo, 6), round(hi, 6)) for k, (lo, hi) in intervals.items()},
        test_ece=round(test_calibration.ece, 6),
        robustness=robustness,
    )
    report = asdict(result) | {
        "val_ece": round(val_calibration.ece, 6),
        "test_calibration": asdict(test_calibration),
        "bootstrap_resamples": config.bootstrap_resamples,
        "val_images": len(frames["val"]),
        "test_images": len(frames["test"]),
    }
    (paths.root / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    metadata.threshold = threshold
    metadata.threshold_strategy = config.threshold_strategy
    metadata.metrics = {
        "val": result.val,
        "test": result.test,
        "test_ci95": result.test_ci95,
        "test_ece": result.test_ece,
    }
    metadata.write(paths.model_dir)
    return result
