"""Inference API used by the service layer: load once, classify PIL images, explain them."""

import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import onnxruntime as ort  # type: ignore[import-untyped]
import torch
from numpy.typing import NDArray
from PIL import Image
from torch import nn

from baulens.cracks.data import preprocess
from baulens.cracks.gradcam import GradCAM, overlay_heatmap
from baulens.cracks.metadata import MODEL_FILE, ONNX_FILE, ModelMetadata, file_sha256
from baulens.cracks.model import build_model, gradcam_layer
from baulens.schemas import CrackPrediction

type Backend = Literal["onnx", "torch"]

__all__ = ["Backend", "CrackClassifier", "overlay_heatmap"]


class CrackClassifier:
    """Patch-level crack classifier with the decision threshold chosen on validation.

    ``backend="onnx"`` serves predictions with ONNX Runtime; Grad-CAM always uses the
    PyTorch checkpoint, which is loaded lazily on first use. One instance may serve
    concurrent requests: lazy initialisation is locked, and Grad-CAM keeps its captures per
    thread (see :class:`~baulens.cracks.gradcam.GradCAM`).
    """

    def __init__(
        self,
        model_dir: Path,
        metadata: ModelMetadata,
        backend: Backend,
        threads: int | None = None,
    ) -> None:
        if metadata.threshold is None:
            raise ValueError(f"{model_dir} has no validation threshold; run evaluate first")
        self.model_dir = model_dir
        self.metadata = metadata
        self.backend = backend
        self.threshold: float = metadata.threshold
        self._threads = threads
        self._session: ort.InferenceSession | None = None
        self._torch: nn.Module | None = None
        self._cam: GradCAM | None = None
        self._init_lock = threading.Lock()
        if backend == "onnx":
            expected = metadata.onnx.get("sha256")
            if expected is None or file_sha256(model_dir / ONNX_FILE) != expected:
                raise ValueError(
                    f"{model_dir / ONNX_FILE} does not match the SHA-256 recorded by the last "
                    "successful export; rerun `baulens cracks export`"
                )
            options = ort.SessionOptions()
            if threads is not None:
                options.intra_op_num_threads = threads
            self._session = ort.InferenceSession(
                str(model_dir / ONNX_FILE), options, providers=["CPUExecutionProvider"]
            )
        else:
            self._torch_model()

    @classmethod
    def load(
        cls, model_dir: Path, backend: Backend = "onnx", threads: int | None = None
    ) -> "CrackClassifier":
        return cls(model_dir, ModelMetadata.read(model_dir), backend, threads)

    @property
    def model_name(self) -> str:
        return self.metadata.model_name

    @property
    def model_version(self) -> str:
        return self.metadata.model_version

    def _torch_model(self) -> nn.Module:
        with self._init_lock:
            if self._torch is None:
                if self._threads is not None:
                    torch.set_num_threads(self._threads)
                model = build_model(self.metadata.architecture, pretrained=False)
                state = torch.load(
                    self.model_dir / MODEL_FILE, map_location="cpu", weights_only=True
                )
                model.load_state_dict(state)
                self._torch = model.eval()
            return self._torch

    def _gradcam(self) -> GradCAM:
        model = self._torch_model()
        with self._init_lock:
            if self._cam is None:
                # Registered once; the hook list is never changed while requests run.
                self._cam = GradCAM(model, gradcam_layer(model, self.metadata.architecture))
            return self._cam

    def preprocess(self, images: Sequence[Image.Image]) -> NDArray[np.float32]:
        meta = self.metadata
        return np.stack(
            [
                preprocess(image, meta.image_size, meta.normalization_mean, meta.normalization_std)
                for image in images
            ]
        )

    def predict_proba(self, images: Sequence[Image.Image]) -> NDArray[np.float32]:
        """P(crack) per image, shape (N,)."""
        return self.predict_batch(self.preprocess(images))

    def predict_batch(self, batch: NDArray[np.float32]) -> NDArray[np.float32]:
        """P(crack) for an already preprocessed (N, 3, S, S) float32 batch."""
        if self._session is not None:
            (probabilities,) = self._session.run(["probability"], {"image": batch})
            return np.asarray(probabilities, dtype=np.float32)
        with torch.no_grad():
            logits = self._torch_model()(torch.from_numpy(batch)).reshape(-1)
            return np.asarray(torch.sigmoid(logits).numpy(), dtype=np.float32)

    def predict(self, image: Image.Image) -> CrackPrediction:
        probability = float(self.predict_proba([image])[0])
        return CrackPrediction(
            crack_probability=probability,
            threshold=self.threshold,
            label="crack" if probability >= self.threshold else "no_crack",
            model_name=self.model_name,
            model_version=self.model_version,
        )

    def gradcam(self, image: Image.Image) -> NDArray[np.float32]:
        """Grad-CAM heatmap in [0, 1] at the model input resolution (image_size x image_size).

        Use :func:`overlay_heatmap` to blend it over the original image at any size.
        """
        heatmaps = self._gradcam()(torch.from_numpy(self.preprocess([image])))
        return np.asarray(heatmaps[0], dtype=np.float32)
