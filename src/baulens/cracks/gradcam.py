"""Grad-CAM (Selvaraju et al., 2017) for the single crack logit, built on module hooks."""

import threading
from types import TracebackType

import matplotlib as mpl
import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from torch import nn
from torch.nn import functional


class GradCAM:
    """Class-activation maps for the crack logit at ``layer``'s output.

    The map is ReLU(sum_k w_k A_k) with w_k the spatially averaged gradient of the logit with
    respect to activation channel A_k, upsampled to the input size and scaled to [0, 1] per
    image. Images whose map is all zero (no positive evidence for "crack") stay zero instead
    of having noise stretched to full range.

    Safe for concurrent requests on one shared model: the forward hook is registered once
    and never mutated while serving, it stores activations only for the thread that is
    inside :meth:`__call__` (thread-local), and gradients come from ``torch.autograd.grad``,
    so no shared ``.grad`` buffers or module state are written. Other threads may run plain
    predictions on the same model at the same time.
    """

    def __init__(self, model: nn.Module, layer: nn.Module) -> None:
        self.model = model
        self._local = threading.local()
        self._handle = layer.register_forward_hook(self._capture)

    def _capture(
        self, module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        if getattr(self._local, "active", False):
            self._local.activations = output

    def __call__(self, images: torch.Tensor) -> NDArray[np.float32]:
        """Return maps of shape (N, H, W) for a normalised batch of shape (N, 3, H, W)."""
        if self.model.training:
            raise RuntimeError("Grad-CAM expects the model in eval mode")
        self._local.active = True
        self._local.activations = None
        try:
            with torch.enable_grad():
                # A request-local leaf that requires grad keeps the graph alive even when all
                # parameters are frozen.
                inputs = images.detach().clone().requires_grad_(True)
                logits = self.model(inputs).reshape(-1)
                activations: torch.Tensor | None = self._local.activations
                if activations is None or not activations.requires_grad:
                    raise RuntimeError("Grad-CAM hook did not capture a differentiable output")
                # Samples are independent in eval mode, so the gradient of the summed logits
                # gives every image the gradient of its own logit.
                (gradients,) = torch.autograd.grad(logits.sum(), activations)
        finally:
            self._local.active = False
            self._local.activations = None
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = functional.relu((weights * activations.detach()).sum(dim=1, keepdim=True))
        cam = functional.interpolate(
            cam, size=images.shape[-2:], mode="bilinear", align_corners=False
        ).squeeze(1)
        peak = cam.flatten(1).amax(dim=1).clamp_min(1e-12)[:, None, None]
        cam = torch.where(peak > 1e-8, cam / peak, torch.zeros_like(cam))
        return cam.cpu().numpy().astype(np.float32)

    def close(self) -> None:
        self._handle.remove()

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def overlay_heatmap(
    image: Image.Image, heatmap: NDArray[np.floating], alpha: float = 0.45
) -> Image.Image:
    """Blend a [0, 1] heatmap (any resolution) over an RGB image with the 'jet' colour map."""
    base = image.convert("RGB")
    scaled = Image.fromarray((np.clip(heatmap, 0, 1) * 255).astype(np.uint8)).resize(
        base.size, Image.Resampling.BILINEAR
    )
    colours = mpl.colormaps["jet"](np.asarray(scaled, dtype=np.float32) / 255.0)[..., :3]
    blended = (1 - alpha) * np.asarray(base, dtype=np.float32) / 255.0 + alpha * colours
    return Image.fromarray((np.clip(blended, 0, 1) * 255).astype(np.uint8))
