"""ONNX export with a dynamic batch axis and a PyTorch-vs-ONNX Runtime parity check."""

import json
import logging
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort  # type: ignore[import-untyped]
import torch
from torch import nn

from baulens.config import CracksConfig
from baulens.cracks.data import CrackImageDataset, EvalTransform, stratified_head
from baulens.cracks.evaluate import load_torch_model
from baulens.cracks.metadata import (
    METADATA_FILE,
    ONNX_FILE,
    ModelMetadata,
    file_sha256,
    run_paths,
    verify_training_data,
)
from baulens.cracks.splits import ensure_splits
from baulens.paths import project_path
from baulens.seed import seed_everything

logger = logging.getLogger(__name__)
INPUT_NAME = "image"
OUTPUT_NAMES = ("logit", "probability")


class ExportError(RuntimeError):
    """The exported graph is invalid or disagrees with the PyTorch model."""


class _WithProbability(nn.Module):
    """Export wrapper: one logit and its sigmoid per image, shape (N,)."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logit = self.model(image).reshape(-1)
        return logit, torch.sigmoid(logit)


@dataclass(frozen=True)
class ExportResult:
    onnx_path: str
    opset: int
    file_bytes: int
    parity_images: int
    max_abs_diff_logit: float
    max_abs_diff_probability: float
    labels_identical: bool
    tolerance: float
    threshold: float
    batch_sizes_checked: list[int]


def parity(
    model: nn.Module, onnx_path: Path, batch: torch.Tensor, threshold: float
) -> tuple[float, float, bool]:
    """Max |logit| and |probability| differences (batch N and batch 1) and label agreement."""
    with torch.no_grad():
        torch_logit, torch_prob = (t.numpy() for t in model(batch))
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_logit, onnx_prob = session.run(list(OUTPUT_NAMES), {INPUT_NAME: batch.numpy()})
    single_logit = session.run([OUTPUT_NAMES[0]], {INPUT_NAME: batch[:1].numpy()})[0]
    diff_logit = float(np.max(np.abs(np.asarray(onnx_logit) - torch_logit)))
    diff_logit = max(diff_logit, float(np.max(np.abs(np.asarray(single_logit) - torch_logit[:1]))))
    diff_prob = float(np.max(np.abs(np.asarray(onnx_prob) - torch_prob)))
    same = bool(np.array_equal(np.asarray(onnx_prob) >= threshold, torch_prob >= threshold))
    return diff_logit, diff_prob, same


def export(config: CracksConfig) -> ExportResult:
    """Export ``model.pt`` to ONNX, check it, and publish it only if parity holds.

    The candidate graph and its metadata are written to a staging directory next to the
    model. Only after ``onnx.checker`` and the parity check pass are they moved into place
    (atomic same-directory renames: graph first, then metadata). ``metadata.json`` records
    the graph's SHA-256 and ``CrackClassifier`` refuses a graph that does not match it, so an
    interrupted publish cannot be served silently. A failed check leaves the previously
    published model and metadata untouched.
    """
    seed_everything(config.seed)
    if config.num_threads is not None:
        torch.set_num_threads(config.num_threads)
    paths = run_paths(config)
    metadata = ModelMetadata.read(paths.model_dir)
    if metadata.threshold is None:
        raise ExportError("no validation threshold in metadata.json; run evaluate first")
    manifest = ensure_splits(config)
    verify_training_data(metadata, manifest, config)
    model = _WithProbability(load_torch_model(config, torch.device("cpu"))).eval()
    test = stratified_head(manifest[manifest["split"] == "test"], config.parity_images, config.seed)
    dataset = CrackImageDataset.from_frame(
        project_path(config.data_dir), test, EvalTransform(config.image_size)
    )
    batch = torch.stack([dataset[i][0] for i in range(len(dataset))])

    with tempfile.TemporaryDirectory(prefix=".staging-", dir=paths.model_dir) as staging_dir:
        staged = Path(staging_dir) / ONNX_FILE
        torch.onnx.export(
            model,
            (torch.zeros(2, 3, config.image_size, config.image_size),),
            str(staged),
            input_names=[INPUT_NAME],
            output_names=list(OUTPUT_NAMES),
            dynamic_shapes={INPUT_NAME: {0: torch.export.Dim("batch", min=1, max=4096)}},
            opset_version=config.onnx_opset,
            dynamo=True,
            external_data=False,
            verbose=False,
        )
        onnx.checker.check_model(onnx.load(str(staged)), full_check=True)
        diff_logit, diff_prob, labels_identical = parity(model, staged, batch, metadata.threshold)
        result = ExportResult(
            onnx_path=str(paths.onnx),
            opset=config.onnx_opset,
            file_bytes=staged.stat().st_size,
            parity_images=len(dataset),
            max_abs_diff_logit=diff_logit,
            max_abs_diff_probability=diff_prob,
            labels_identical=labels_identical,
            tolerance=config.parity_tolerance,
            threshold=metadata.threshold,
            batch_sizes_checked=[1, len(dataset)],
        )
        logger.info("ONNX parity: max |dlogit| %.2e, max |dprob| %.2e", diff_logit, diff_prob)
        if max(diff_logit, diff_prob) > config.parity_tolerance or not labels_identical:
            (paths.root / "export_failed.json").write_text(
                json.dumps(asdict(result), indent=2), encoding="utf-8"
            )
            raise ExportError(
                f"ONNX parity failed: max |dlogit| {diff_logit:.2e}, max |dprob| "
                f"{diff_prob:.2e}, labels identical: {labels_identical}; "
                "the previously published model was left unchanged"
            )
        metadata.onnx = asdict(result) | {
            "sha256": file_sha256(staged),
            "onnx_version": onnx.__version__,
            "ort_version": ort.__version__,
        }
        metadata.write(Path(staging_dir))
        staged.replace(paths.onnx)
        (Path(staging_dir) / METADATA_FILE).replace(paths.metadata)
    (paths.root / "export.json").write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    return result
