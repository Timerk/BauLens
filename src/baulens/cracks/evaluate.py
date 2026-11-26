"""Validation-chosen threshold, test metrics with bootstrap CIs, calibration, robustness probe,
Grad-CAM galleries and the error gallery."""

import logging
import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from PIL import Image
from torch import nn

from baulens.artifacts import write_json_atomic
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
from baulens.cracks.metadata import load_published, publish_model, run_paths, verify_training_data
from baulens.cracks.metrics import (
    binary_metrics,
    bootstrap_intervals,
    calibration,
    choose_threshold,
    leave_one_out_thresholds,
    mean_confidence,
    optimal_threshold_ranges,
)
from baulens.cracks.model import gradcam_layer, model_from_weights
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
            | {k: round(metrics[k], 4) for k in ("accuracy", "precision", "recall", "f1")}
            | {
                "roc_auc": round(metrics["roc_auc"], 6),
                "mean_confidence": round(mean_confidence(probabilities), 4),
                "ece": round(calibration(labels, probabilities, config.calibration_bins).ece, 5),
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
    published = load_published(paths.model_dir)
    metadata = published.metadata
    manifest = ensure_splits(config)
    verify_training_data(metadata, manifest, config)
    data_dir = project_path(config.data_dir)
    model = model_from_weights(metadata.architecture, published.weights, device)
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
    plateau = optimal_threshold_ranges(val_labels, val_probs, config.threshold_strategy)
    loo = leave_one_out_thresholds(val_labels, val_probs, config.threshold_strategy)
    loo_counts = pd.Series(loo).value_counts().sort_index()
    image_intervals = bootstrap_intervals(
        test_labels,
        test_probs,
        threshold,
        resamples=config.bootstrap_resamples,
        seed=config.seed,
    )
    intervals = bootstrap_intervals(
        test_labels,
        test_probs,
        threshold,
        resamples=config.bootstrap_resamples,
        seed=config.seed,
        groups=frames["test"]["group"].to_numpy(dtype=np.int64),
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
        "ci_method": (
            "percentile bootstrap over detected duplicate groups (singletons are their own "
            "group), threshold fixed; source-photo dependence is not captured"
        ),
        "test_ci95_image_level": {
            k: (round(lo, 6), round(hi, 6)) for k, (lo, hi) in image_intervals.items()
        },
        "test_groups": int(frames["test"]["group"].nunique()),
        "threshold_optimal_ranges": plateau,
        "threshold_leave_one_out": {
            "method": "exact, every validation image removed once",
            "min": float(loo.min()),
            "max": float(loo.max()),
            "counts": {f"{t:.3f}": int(n) for t, n in loo_counts.items()},
        },
        "val_ece": round(val_calibration.ece, 6),
        "test_calibration": asdict(test_calibration),
        "bootstrap_resamples": config.bootstrap_resamples,
        "val_images": len(frames["val"]),
        "test_images": len(frames["test"]),
    }
    write_json_atomic(paths.root / "evaluation.json", json_safe(report))
    metadata.threshold = threshold
    metadata.threshold_strategy = config.threshold_strategy
    metadata.metrics = {
        "val": result.val,
        "test": result.test,
        "test_ci95": result.test_ci95,
        "test_ece": result.test_ece,
    }
    # A new threshold invalidates the graph's recorded label parity: publish weights and
    # metadata as a new generation without a graph; `export` adds a checked graph.
    metadata.onnx = {}
    publish_model(paths.model_dir, metadata, published.weights)
    return result


def json_safe(value: Any) -> Any:
    """Replace NaN and infinities (e.g. AUC of a one-class subset) by JSON null."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe(item) for item in value]
    return value
