"""ONNX export with a dynamic batch axis and a PyTorch-vs-ONNX Runtime parity check."""

import hashlib
import logging
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort  # type: ignore[import-untyped]
import torch
from torch import nn

from baulens.artifacts import write_json_atomic
from baulens.config import CracksConfig
from baulens.cracks.data import CrackImageDataset, EvalTransform, stratified_head
from baulens.cracks.metadata import (
    ONNX_FILE,
    load_published,
    publish_model,
    run_paths,
    verify_training_data,
)
from baulens.cracks.model import model_from_weights
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
    model: nn.Module, graph: bytes, batch: torch.Tensor, threshold: float
) -> tuple[float, float, bool]:
    """Max |logit| and |probability| differences (batch N and batch 1) and label agreement."""
    with torch.no_grad():
        torch_logit, torch_prob = (t.numpy() for t in model(batch))
    session = ort.InferenceSession(graph, providers=["CPUExecutionProvider"])
    onnx_logit, onnx_prob = session.run(list(OUTPUT_NAMES), {INPUT_NAME: batch.numpy()})
    single_logit = session.run([OUTPUT_NAMES[0]], {INPUT_NAME: batch[:1].numpy()})[0]
    diff_logit = float(np.max(np.abs(np.asarray(onnx_logit) - torch_logit)))
    diff_logit = max(diff_logit, float(np.max(np.abs(np.asarray(single_logit) - torch_logit[:1]))))
    diff_prob = float(np.max(np.abs(np.asarray(onnx_prob) - torch_prob)))
    same = bool(np.array_equal(np.asarray(onnx_prob) >= threshold, torch_prob >= threshold))
    return diff_logit, diff_prob, same


def export(config: CracksConfig) -> ExportResult:
    """Export the current generation's weights to ONNX and publish weights, graph and metadata
    together as a new generation, only if ``onnx.checker`` and the parity check pass.

    The candidate graph is built in a scratch directory. On any failure nothing is published
    and the current generation stays as it is. On success :func:`publish_model` writes all
    three files into a fresh generation and switches ``current.json`` in one replace, so a
    reader sees either the old or the new set, never a mix.
    """
    seed_everything(config.seed)
    if config.num_threads is not None:
        torch.set_num_threads(config.num_threads)
    paths = run_paths(config)
    published = load_published(paths.model_dir)
    metadata = published.metadata
    if metadata.threshold is None:
        raise ExportError("no validation threshold in metadata.json; run evaluate first")
    manifest = ensure_splits(config)
    verify_training_data(metadata, manifest, config)
    model = _WithProbability(model_from_weights(metadata.architecture, published.weights)).eval()
    test = stratified_head(manifest[manifest["split"] == "test"], config.parity_images, config.seed)
    dataset = CrackImageDataset.from_frame(
        project_path(config.data_dir), test, EvalTransform(config.image_size)
    )
    batch = torch.stack([dataset[i][0] for i in range(len(dataset))])

    with tempfile.TemporaryDirectory(prefix="baulens-onnx-") as scratch:
        candidate = Path(scratch) / ONNX_FILE
        torch.onnx.export(
            model,
            (torch.zeros(2, 3, config.image_size, config.image_size),),
            str(candidate),
            input_names=[INPUT_NAME],
            output_names=list(OUTPUT_NAMES),
            dynamic_shapes={INPUT_NAME: {0: torch.export.Dim("batch", min=1, max=4096)}},
            opset_version=config.onnx_opset,
            dynamo=True,
            external_data=False,
            verbose=False,
        )
        graph = candidate.read_bytes()
    onnx.checker.check_model(onnx.load_from_string(graph), full_check=True)
    diff_logit, diff_prob, labels_identical = parity(model, graph, batch, metadata.threshold)
    result = ExportResult(
        onnx_path=ONNX_FILE,
        opset=config.onnx_opset,
        file_bytes=len(graph),
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
        write_json_atomic(paths.root / "export_failed.json", asdict(result))
        raise ExportError(
            f"ONNX parity failed: max |dlogit| {diff_logit:.2e}, max |dprob| "
            f"{diff_prob:.2e}, labels identical: {labels_identical}; "
            "the previously published model was left unchanged"
        )
    metadata.onnx = asdict(result) | {
        "sha256": hashlib.sha256(graph).hexdigest(),
        "onnx_version": onnx.__version__,
        "ort_version": ort.__version__,
    }
    generation = publish_model(paths.model_dir, metadata, published.weights, graph)
    write_json_atomic(paths.root / "export.json", asdict(result) | {"generation": generation.name})
    return result
