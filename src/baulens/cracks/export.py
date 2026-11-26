"""ONNX export with a dynamic batch axis and a PyTorch-vs-ONNX Runtime parity check."""

import json
import logging
from dataclasses import asdict, dataclass

import numpy as np
import onnx
import onnxruntime as ort  # type: ignore[import-untyped]
import torch
from torch import nn

from baulens.config import CracksConfig
from baulens.cracks.data import CrackImageDataset, EvalTransform, stratified_head
from baulens.cracks.evaluate import load_torch_model
from baulens.cracks.metadata import ModelMetadata, run_paths
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


def export(config: CracksConfig) -> ExportResult:
    """Export ``model.pt`` to ``model.onnx``, run ``onnx.checker`` and verify parity on test."""
    seed_everything(config.seed)
    if config.num_threads is not None:
        torch.set_num_threads(config.num_threads)
    paths = run_paths(config)
    metadata = ModelMetadata.read(paths.model_dir)
    if metadata.threshold is None:
        raise ExportError("no validation threshold in metadata.json; run evaluate first")
    model = _WithProbability(load_torch_model(config, torch.device("cpu"))).eval()
    dummy = torch.zeros(2, 3, config.image_size, config.image_size)
    torch.onnx.export(
        model,
        (dummy,),
        str(paths.onnx),
        input_names=[INPUT_NAME],
        output_names=list(OUTPUT_NAMES),
        dynamic_shapes={INPUT_NAME: {0: torch.export.Dim("batch", min=1, max=4096)}},
        opset_version=config.onnx_opset,
        dynamo=True,
        external_data=False,
        verbose=False,
    )
    onnx.checker.check_model(onnx.load(str(paths.onnx)), full_check=True)

    manifest = ensure_splits(config)
    test = stratified_head(manifest[manifest["split"] == "test"], config.parity_images, config.seed)
    dataset = CrackImageDataset.from_frame(
        project_path(config.data_dir), test, EvalTransform(config.image_size)
    )
    batch = torch.stack([dataset[i][0] for i in range(len(dataset))])
    with torch.no_grad():
        torch_logit, torch_prob = (t.numpy() for t in model(batch))
    session = ort.InferenceSession(str(paths.onnx), providers=["CPUExecutionProvider"])
    onnx_logit, onnx_prob = session.run(list(OUTPUT_NAMES), {INPUT_NAME: batch.numpy()})
    single_logit = session.run([OUTPUT_NAMES[0]], {INPUT_NAME: batch[:1].numpy()})[0]
    diff_logit = float(np.max(np.abs(np.asarray(onnx_logit) - torch_logit)))
    diff_prob = float(np.max(np.abs(np.asarray(onnx_prob) - torch_prob)))
    diff_logit = max(diff_logit, float(np.max(np.abs(np.asarray(single_logit) - torch_logit[:1]))))
    labels_identical = bool(
        np.array_equal(
            np.asarray(onnx_prob) >= metadata.threshold, torch_prob >= metadata.threshold
        )
    )
    result = ExportResult(
        onnx_path=str(paths.onnx),
        opset=config.onnx_opset,
        file_bytes=paths.onnx.stat().st_size,
        parity_images=len(dataset),
        max_abs_diff_logit=diff_logit,
        max_abs_diff_probability=diff_prob,
        labels_identical=labels_identical,
        tolerance=config.parity_tolerance,
        threshold=metadata.threshold,
        batch_sizes_checked=[1, len(dataset)],
    )
    (paths.root / "export.json").write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    metadata.onnx = asdict(result) | {
        "onnx_version": onnx.__version__,
        "ort_version": ort.__version__,
    }
    metadata.write(paths.model_dir)
    logger.info("ONNX parity: max |dlogit| %.2e, max |dprob| %.2e", diff_logit, diff_prob)
    if max(diff_logit, diff_prob) > config.parity_tolerance or not labels_identical:
        raise ExportError(
            f"ONNX parity failed: max |dlogit| {diff_logit:.2e}, max |dprob| {diff_prob:.2e}, "
            f"labels identical: {labels_identical}"
        )
    return result
