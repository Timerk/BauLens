"""Tiny, locally built model artifacts for API tests (CPU only, no downloads)."""

import hashlib
import io
from pathlib import Path

import torch
import torchvision
from PIL import Image
from torch import nn

from baulens.config import StrengthConfig, load_config
from baulens.cracks.data import IMAGENET_MEAN, IMAGENET_STD
from baulens.cracks.export import INPUT_NAME, OUTPUT_NAMES
from baulens.cracks.metadata import ModelMetadata, publish_model
from baulens.cracks.model import build_model
from baulens.strength import train

IMAGE_SIZE = 64


class _WithProbability(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logit = self.model(image).reshape(-1)
        return logit, torch.sigmoid(logit)


def build_strength_model(artifacts_dir: Path) -> Path:
    """Train the smoke configuration on the 60-row fixture into ``artifacts_dir/strength``."""
    settings = load_config(Path("configs/smoke/strength.yaml"), StrengthConfig)
    root = artifacts_dir / "strength"
    result = train(
        settings.model_copy(
            update={
                "artifacts_dir": root,
                "results_path": root / "results.md",
                "figures_dir": root / "figures",
            }
        )
    )
    return result.model_dir


def build_crack_model(artifacts_dir: Path) -> Path:
    """Publish an untrained ResNet-18 with an ONNX graph into ``artifacts_dir/cracks/model``.

    The predictions are meaningless; the API tests only need the real loading, verification
    and inference code paths.
    """
    torch.manual_seed(0)
    model = build_model("resnet18", pretrained=False).eval()
    weights = io.BytesIO()
    torch.save(model.state_dict(), weights)
    graph_path = artifacts_dir / "model.onnx"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        _WithProbability(model).eval(),
        (torch.zeros(2, 3, IMAGE_SIZE, IMAGE_SIZE),),
        str(graph_path),
        input_names=[INPUT_NAME],
        output_names=list(OUTPUT_NAMES),
        dynamic_shapes={INPUT_NAME: {0: torch.export.Dim("batch", min=1, max=64)}},
        opset_version=18,
        dynamo=True,
        external_data=False,
        verbose=False,
    )
    graph = graph_path.read_bytes()
    graph_path.unlink()
    metadata = ModelMetadata(
        model_name="resnet18",
        model_version="resnet18-test",
        architecture="resnet18",
        image_size=IMAGE_SIZE,
        normalization_mean=IMAGENET_MEAN,
        normalization_std=IMAGENET_STD,
        pretrained_weights=None,
        threshold=0.5,
        threshold_strategy="f1",
        data_checksum="0" * 64,
        split_checksum="0" * 64,
        git_sha="test",
        git_dirty=False,
        torch_version=torch.__version__,
        torchvision_version=torchvision.__version__,
        created_at="2026-01-01T00:00:00+00:00",
        metrics={"test": {"f1": 0.5, "roc_auc": 0.5, "tp": 1.0}},
        onnx={"sha256": hashlib.sha256(graph).hexdigest()},
    )
    model_dir = artifacts_dir / "cracks" / "model"
    publish_model(model_dir, metadata, weights.getvalue(), graph)
    return model_dir


def image_bytes(fmt: str = "JPEG", size: tuple[int, int] = (80, 60), **save: object) -> bytes:
    image = Image.new("RGB", size, (150, 150, 150))
    for x in range(size[0] // 2, size[0] // 2 + 3):
        for y in range(size[1]):
            image.putpixel((x, y), (30, 30, 30))
    buffer = io.BytesIO()
    image.save(buffer, format=fmt, **save)
    return buffer.getvalue()
