"""One-off migration of a checkpoint saved before model generations existed.

Legacy run directories hold ``model/model.pt`` and ``model/metadata.json`` side by side, with
nothing that ties the weights to the sidecar. A sidecar's checksums alone therefore do not prove
that the weights next to it are the ones trained on that split (an interrupted later run could
have replaced them). Before publishing such a checkpoint as a generation, this module requires
independent evidence:

1. the recorded data and split checksums match the current content-keyed manifest, and
2. re-scoring the validation split with the weights reproduces the validation loss that the
   training run recorded for its restored best epoch (``training.best_val_loss`` and the
   matching ``history`` row) to within ``LOSS_TOLERANCE``.

Weights from another run (other seed, other epoch, other split) would not reproduce a loss
recorded to sixteen digits. Only then are the weights, their SHA-256 and the metadata
published together, and the legacy files are moved to ``legacy_model/`` for audit.
"""

import logging
import math
import shutil
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any

from baulens.artifacts import write_json_atomic
from baulens.config import CracksConfig
from baulens.cracks.data import CrackImageDataset, EvalTransform, make_loader
from baulens.cracks.metadata import (
    LegacyMigrationError,
    ModelMetadata,
    legacy_files,
    publish_model,
    run_paths,
    verify_training_data,
)
from baulens.cracks.model import model_from_weights
from baulens.cracks.splits import ensure_splits, split_settings
from baulens.cracks.train import configure_runtime, run_epoch
from baulens.paths import project_path

logger = logging.getLogger(__name__)
LOSS_TOLERANCE = 1e-6


def migrate_legacy(config: CracksConfig) -> dict[str, Any]:
    """Verify a legacy checkpoint as described in the module docstring and publish it."""
    device = configure_runtime(config)
    paths = run_paths(config)
    legacy = legacy_files(paths.model_dir)
    if legacy is None:
        raise LegacyMigrationError(f"no unpublished legacy checkpoint in {paths.model_dir}")
    weights_path, metadata_path = legacy
    weights = weights_path.read_bytes()
    metadata = ModelMetadata.parse(metadata_path.read_bytes())
    manifest = ensure_splits(config)
    verify_training_data(metadata, manifest, config)

    recorded = metadata.training.get("best_val_loss")
    best_epoch = metadata.training.get("best_epoch")
    history = {row["epoch"]: row for row in metadata.training.get("history", [])}
    if not isinstance(recorded, float) or best_epoch not in history:
        raise LegacyMigrationError("metadata lacks the recorded best validation loss")
    if history[best_epoch]["val_loss"] != recorded:
        raise LegacyMigrationError("best_val_loss disagrees with the training history")

    model = model_from_weights(metadata.architecture, weights, device)
    val = manifest[manifest["split"] == "val"]
    loader = make_loader(
        CrackImageDataset.from_frame(
            project_path(config.data_dir), val, EvalTransform(metadata.image_size)
        ),
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        seed=config.seed,
    )
    reproduced, _, _ = run_epoch(model, loader, device)
    difference = abs(reproduced - recorded)
    logger.info("validation loss: recorded %.12f, reproduced %.12f", recorded, reproduced)
    if not math.isfinite(reproduced) or difference > LOSS_TOLERANCE:
        raise LegacyMigrationError(
            f"weights do not reproduce the recorded validation loss ({reproduced:.10f} vs "
            f"{recorded:.10f}); refusing to publish them with this metadata"
        )

    if metadata.split_settings is None:
        metadata.split_settings = split_settings(config)
    # A legacy graph is not carried over; `export` rebuilds and re-checks it.
    metadata.onnx = {}
    generation = publish_model(paths.model_dir, metadata, weights)
    archive = paths.root / "legacy_model"
    archive.mkdir(exist_ok=True)
    for path in (weights_path, metadata_path, paths.model_dir / "model.onnx"):
        if path.exists():
            shutil.move(path, archive / path.name)
    record = {
        "migrated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "generation": generation.name,
        "weights_sha256": sha256(weights).hexdigest(),
        "recorded_best_val_loss": recorded,
        "reproduced_val_loss": reproduced,
        "absolute_difference": difference,
        "tolerance": LOSS_TOLERANCE,
        "data_checksum": metadata.data_checksum,
        "split_checksum": metadata.split_checksum,
        "legacy_files_moved_to": str(archive),
    }
    write_json_atomic(paths.root / "migration.json", record)
    return record
