"""Two-phase transfer learning with early stopping on validation loss."""

import copy
import io
import json
import logging
import math
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

import pandas as pd
import torch
import torchvision  # type: ignore[import-untyped]
from torch import nn
from torch.utils.data import DataLoader

from baulens.config import CracksConfig
from baulens.cracks import figures
from baulens.cracks.data import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    CrackImageDataset,
    EvalTransform,
    Sample,
    make_loader,
    stratified_sample,
    train_transform,
)
from baulens.cracks.metadata import ModelMetadata, git_revision, publish_model, run_paths
from baulens.cracks.model import (
    build_model,
    configure_trainable,
    count_parameters,
    resolve_device,
    set_train_mode,
    weights_name,
)
from baulens.cracks.splits import data_checksum, ensure_splits, split_checksum, split_settings
from baulens.paths import project_path
from baulens.seed import seed_everything

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EpochLog:
    phase: str
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    val_f1: float
    learning_rate: float
    seconds: float


@dataclass(frozen=True)
class TrainResult:
    model_dir: str
    train_images: int
    val_images: int
    epochs_run: int
    best_epoch: int
    best_val_loss: float
    best_val_f1: float
    wall_clock_seconds: float
    device: str
    threads: int


def configure_runtime(config: CracksConfig) -> torch.device:
    """Seed, set the thread count and pick the device."""
    seed_everything(config.seed)
    if config.num_threads is not None:
        torch.set_num_threads(config.num_threads)
    device = resolve_device(config.device)
    if device.type == "cuda":
        # Some CUDA/ROCm kernels (e.g. adaptive pooling backward) have no deterministic variant.
        torch.use_deterministic_algorithms(True, warn_only=True)
    return device


def run_epoch(
    model: nn.Module,
    loader: DataLoader[Sample],
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> tuple[float, float, float]:
    """Mean loss, accuracy and F1 at 0.5 over one pass; trains when an optimizer is given."""
    loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
    total_loss, count, correct, tp, fp, fn = 0.0, 0, 0, 0, 0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images).squeeze(1)
            loss = loss_fn(logits, labels)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                (loss / len(labels)).backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            predicted = logits.detach() >= 0
            truth = labels.bool()
            total_loss += float(loss.item())
            count += len(labels)
            correct += int((predicted == truth).sum())
            tp += int((predicted & truth).sum())
            fp += int((predicted & ~truth).sum())
            fn += int((~predicted & truth).sum())
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0
    return total_loss / count, correct / count, f1


def _optimizer(
    model: nn.Module, learning_rate: float, weight_decay: float, steps: int
) -> tuple[torch.optim.AdamW, torch.optim.lr_scheduler.LRScheduler]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(steps, 1))
    return optimizer, scheduler


def train(config: CracksConfig) -> TrainResult:
    """Phase 1 trains the head on a frozen backbone, phase 2 fine-tunes the last stage(s).

    Each phase uses AdamW with a per-step cosine decay. The checkpoint with the lowest
    validation loss over both phases is restored and saved; phase 2 stops after ``patience``
    epochs without improvement.
    """
    started = time.perf_counter()
    git_sha, git_dirty = git_revision()  # the code that trains, not the HEAD at the end
    device = configure_runtime(config)
    paths = run_paths(config)
    paths.model_dir.mkdir(parents=True, exist_ok=True)
    manifest = ensure_splits(config)
    data_dir = project_path(config.data_dir)
    train_frame = stratified_sample(
        manifest[manifest["split"] == "train"], config.train_fraction, config.seed
    )
    val_frame = manifest[manifest["split"] == "val"]
    train_frame[["path", "label"]].to_csv(
        paths.root / "train_subset.csv", index=False, lineterminator="\n"
    )

    train_loader = make_loader(
        CrackImageDataset.from_frame(data_dir, train_frame, train_transform(config)),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        seed=config.seed,
    )
    val_loader = make_loader(
        CrackImageDataset.from_frame(data_dir, val_frame, EvalTransform(config.image_size)),
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        seed=config.seed,
    )

    model = build_model(config.architecture, config.pretrained).to(device)
    history: list[EpochLog] = []
    best_state = copy.deepcopy(model.state_dict())
    best_loss, best_f1, best_epoch = math.inf, 0.0, 0
    phases = [
        ("head", 0, config.head_epochs, config.lr_head),
        ("finetune", config.finetune_blocks, config.epochs, config.lr_finetune),
    ]
    epoch = 0
    for phase, blocks, epochs, learning_rate in phases:
        if epochs == 0:
            continue
        trainable = configure_trainable(model, config.architecture, blocks)
        optimizer, scheduler = _optimizer(
            model, learning_rate, config.weight_decay, epochs * len(train_loader)
        )
        stale = 0
        for _ in range(epochs):
            epoch += 1
            epoch_start = time.perf_counter()
            set_train_mode(model, trainable)
            train_loss, train_acc, _ = run_epoch(model, train_loader, device, optimizer, scheduler)
            model.eval()
            val_loss, val_acc, val_f1 = run_epoch(model, val_loader, device)
            history.append(
                EpochLog(
                    phase=phase,
                    epoch=epoch,
                    train_loss=train_loss,
                    train_accuracy=train_acc,
                    val_loss=val_loss,
                    val_accuracy=val_acc,
                    val_f1=val_f1,
                    learning_rate=float(optimizer.param_groups[0]["lr"]),
                    seconds=time.perf_counter() - epoch_start,
                )
            )
            logger.info(
                "[%s] epoch %d: train_loss=%.4f val_loss=%.4f val_f1=%.4f (%.0f s)",
                phase,
                epoch,
                train_loss,
                val_loss,
                val_f1,
                history[-1].seconds,
            )
            if val_loss < best_loss:
                best_loss, best_f1, best_epoch, stale = val_loss, val_f1, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
                # Best-so-far weights for inspection after an interruption. This file is
                # never published: only the complete artifact below becomes a generation.
                torch.save(
                    {k: v.cpu() for k, v in best_state.items()}, paths.checkpoint_in_progress
                )
            else:
                stale += 1
                if phase == "finetune" and stale >= config.patience:
                    break

    model.load_state_dict(best_state)
    weights = io.BytesIO()
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, weights)
    wall_clock = time.perf_counter() - started

    frame = pd.DataFrame([asdict(row) for row in history])
    frame.to_csv(paths.root / "history.csv", index=False, lineterminator="\n")
    figures.learning_curves(frame, best_epoch, figures.targets(config, "learning_curves.png"))

    result = TrainResult(
        model_dir=str(paths.model_dir),
        train_images=len(train_frame),
        val_images=len(val_frame),
        epochs_run=epoch,
        best_epoch=best_epoch,
        best_val_loss=best_loss,
        best_val_f1=best_f1,
        wall_clock_seconds=wall_clock,
        device=str(device),
        threads=torch.get_num_threads(),
    )
    created = datetime.now(UTC)
    training: dict[str, Any] = asdict(result) | {
        "parameters": count_parameters(model),
        "history": [asdict(row) for row in history],
    }
    metadata = ModelMetadata(
        model_name=config.architecture,
        model_version=f"{config.architecture}-{created:%Y%m%d}-{git_sha[:7]}",
        architecture=config.architecture,
        image_size=config.image_size,
        normalization_mean=IMAGENET_MEAN,
        normalization_std=IMAGENET_STD,
        pretrained_weights=weights_name(config.architecture, config.pretrained),
        data_checksum=data_checksum(manifest),
        split_checksum=split_checksum(manifest),
        split_settings=split_settings(config),
        git_sha=git_sha,
        git_dirty=git_dirty,
        torch_version=torch.__version__,
        torchvision_version=torchvision.__version__,
        created_at=created.isoformat(timespec="seconds"),
        training=training,
        config=config.model_dump(mode="json"),
    )
    # Weights, their digest and the split/input provenance are published as one generation.
    publish_model(paths.model_dir, metadata, weights.getvalue())
    paths.checkpoint_in_progress.unlink(missing_ok=True)
    (paths.root / "train_result.json").write_text(
        json.dumps(asdict(result), indent=2), encoding="utf-8"
    )
    return result
