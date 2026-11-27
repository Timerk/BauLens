"""Static report figures (PNG) for the crack classifier."""

import io
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from numpy.typing import NDArray
from PIL import Image

from baulens.config import CracksConfig
from baulens.cracks.metrics import Calibration
from baulens.paths import project_path

SERIES = ("#257de0", "#f47230", "#00b894")
TEXT = "#191919"
MUTED = "#666666"
GRID = "#e5e5e5"
MAX_PNG_BYTES = 290_000

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "font.size": 9,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": TEXT,
        "axes.titlesize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "lines.linewidth": 2,
        "legend.frameon": False,
    }
)


def targets(config: CracksConfig, name: str) -> list[Path]:
    """Run directory figure plus, when configured, the committed docs copy."""
    paths = [project_path(config.artifacts_dir) / "figures" / name]
    if config.docs_figures_dir is not None:
        paths.append(project_path(config.docs_figures_dir) / name)
    return paths


def save(figure: Figure, paths: Sequence[Path]) -> None:
    """Write PNGs, palette-quantising any that would exceed the size budget for docs."""
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(figure)
    payload = buffer.getvalue()
    if len(payload) > MAX_PNG_BYTES:
        with Image.open(io.BytesIO(payload)) as image:
            quantised = image.convert("RGB").quantize(colors=256, method=Image.Quantize.MEDIANCUT)
            out = io.BytesIO()
            quantised.save(out, format="PNG", optimize=True)
            payload = out.getvalue()
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)


def learning_curves(history: pd.DataFrame, best_epoch: int, paths: Sequence[Path]) -> None:
    figure, (loss_ax, f1_ax) = plt.subplots(1, 2, figsize=(8, 3.2))
    loss_ax.plot(history["epoch"], history["train_loss"], color=SERIES[0], label="train")
    loss_ax.plot(
        history["epoch"],
        history["val_loss"],
        color=SERIES[1],
        marker="o",
        markersize=4,
        label="validation",
    )
    loss_ax.set_title("BCE loss")
    f1_ax.plot(
        history["epoch"],
        history["train_accuracy"],
        color=SERIES[0],
        linestyle="--",
        label="train accuracy",
    )
    f1_ax.plot(
        history["epoch"],
        history["val_f1"],
        color=SERIES[1],
        marker="o",
        markersize=4,
        label="validation F1",
    )
    f1_ax.set_title("Accuracy / F1 at threshold 0.5")
    heads = history.loc[history["phase"] == "head", "epoch"]
    for ax in (loss_ax, f1_ax):
        ax.axvline(best_epoch, color=MUTED, linewidth=1, linestyle=":")
        if len(heads):
            ax.axvspan(0.5, heads.max() + 0.5, color=GRID, alpha=0.6, linewidth=0)
        ax.set_xlabel("epoch")
        ax.legend()
    figure.text(
        0.5,
        -0.04,
        "shaded: frozen backbone (head only); dotted: restored best checkpoint",
        ha="center",
        color=MUTED,
        fontsize=8,
    )
    save(figure, paths)


def confusion_matrix(metrics: dict[str, float], paths: Sequence[Path], title: str) -> None:
    matrix = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    figure, ax = plt.subplots(figsize=(3.4, 3))
    ax.imshow(matrix, cmap="Blues")
    ax.grid(False)
    for (row, col), value in np.ndenumerate(matrix):
        colour = "white" if value > matrix.max() / 2 else TEXT
        ax.text(col, row, f"{int(value)}", ha="center", va="center", color=colour, fontsize=11)
    ax.set_xticks([0, 1], ["no crack", "crack"])
    ax.set_yticks([0, 1], ["no crack", "crack"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title(title)
    save(figure, paths)


def roc_pr(
    labels: NDArray[np.int64], probabilities: NDArray[np.float64], paths: Sequence[Path]
) -> None:
    from sklearn.metrics import precision_recall_curve, roc_curve  # type: ignore[import-untyped]

    fpr, tpr, _ = roc_curve(labels, probabilities)
    precision, recall, _ = precision_recall_curve(labels, probabilities)
    figure, (roc_ax, pr_ax) = plt.subplots(1, 2, figsize=(7, 3))
    roc_ax.plot(fpr, tpr, color=SERIES[0])
    roc_ax.set(xlabel="false positive rate", ylabel="true positive rate", title="ROC (test)")
    pr_ax.plot(recall, precision, color=SERIES[0])
    pr_ax.set(xlabel="recall", ylabel="precision", title="Precision-recall (test)")
    save(figure, paths)


def reliability(result: Calibration, paths: Sequence[Path], title: str) -> None:
    figure, ax = plt.subplots(figsize=(3.6, 3.4))
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--", label="perfect")
    points = [
        (c, f)
        for c, f in zip(result.mean_confidence, result.fraction_positive, strict=True)
        if c is not None and f is not None
    ]
    ax.plot(
        [p[0] for p in points],
        [p[1] for p in points],
        color=SERIES[0],
        marker="o",
        markersize=5,
        label="model (labels: images per bin)",
    )
    counts = [
        n for n, c in zip(result.counts, result.mean_confidence, strict=True) if c is not None
    ]
    for (x, y), count in zip(points, counts, strict=True):
        ax.annotate(
            str(count), (x, y), textcoords="offset points", xytext=(4, -10), fontsize=7, color=MUTED
        )
    ax.set(
        xlabel="mean predicted P(crack)",
        ylabel="observed crack fraction",
        title=f"{title} (ECE {result.ece:.4f})",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=8)
    save(figure, paths)


def gallery(
    images: Sequence[Image.Image],
    captions: Sequence[str],
    title: str,
    paths: Sequence[Path],
    columns: int = 4,
) -> None:
    """Grid of (overlay) images; an empty selection renders a short note instead."""
    rows = max(1, -(-len(images) // columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(1.9 * columns, 2.15 * rows + 0.35),
        squeeze=False,
        layout="constrained",
    )
    for ax in axes.ravel():
        ax.axis("off")
    for ax, image, caption in zip(axes.ravel(), images, captions, strict=False):
        ax.imshow(image)
        ax.set_title(caption, fontsize=7.5, color=TEXT)
    if not images:
        axes[0, 0].text(0, 0.5, "none on the test set", fontsize=10, color=MUTED)
    figure.suptitle(title, fontsize=10, color=TEXT)
    save(figure, paths)


def robustness(rows: Sequence[dict[str, Any]], paths: Sequence[Path]) -> None:
    """Dot plot of recall and precision per corruption (zoomed axis; dots, not bars)."""
    frame = pd.DataFrame(rows)
    figure, ax = plt.subplots(figsize=(7, 3.8))
    positions = np.arange(len(frame))
    ax.scatter(frame["recall"], positions, color=SERIES[0], s=36, label="recall", zorder=3)
    ax.scatter(
        frame["precision"],
        positions,
        color=SERIES[1],
        s=36,
        marker="D",
        label="precision",
        zorder=3,
    )
    ax.set_yticks(positions, frame["corruption"])
    ax.invert_yaxis()
    low = float(min(frame["recall"].min(), frame["precision"].min()))
    ax.set_xlim(min(0.95, low - 0.02), 1.005)
    ax.set_xlabel("at the validation-chosen threshold (note the zoomed axis)")
    ax.set_title(
        f"Corruption probe on a stratified test subset ({int(frame['images'].iloc[0])} images)"
    )
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower left")
    save(figure, paths)


def leakage(
    correlations: NDArray[np.float64],
    threshold: float,
    radius: int,
    pairs: Sequence[tuple[Image.Image, Image.Image, str]],
    paths: Sequence[Path],
) -> None:
    """Thumbnail correlation of perceptual-hash neighbours and borderline example pairs."""
    columns = max(1, len(pairs))
    figure = plt.figure(figsize=(max(7.0, 1.4 * columns), 5.4))
    grid = figure.add_gridspec(2, columns, height_ratios=(1, 1.35), hspace=0.55)
    hist_ax = figure.add_subplot(grid[0, :])
    hist_ax.hist(correlations, bins=60, range=(-0.2, 1.0), color=SERIES[0])
    hist_ax.axvline(threshold, color=TEXT, linewidth=1, linestyle="--")
    if len(correlations):
        hist_ax.set_yscale("log")
    hist_ax.set(
        xlabel=f"Pearson r of 32x32 grey thumbnails (dashed: duplicate threshold {threshold:g})",
        ylabel="image pairs",
        title=f"Pairs within {radius} bits of the perceptual hash (byte-identical files excluded)",
    )
    for column, (first, second, caption) in enumerate(pairs):
        ax = figure.add_subplot(grid[1, column])
        ax.imshow(np.concatenate([np.asarray(first), np.asarray(second)], axis=0))
        ax.set_title(caption, fontsize=7)
        ax.axis("off")
    save(figure, paths)
