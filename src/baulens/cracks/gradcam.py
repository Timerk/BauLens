"""Grad-CAM (Selvaraju et al., 2017) for the single crack logit, built on module hooks."""

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
    """

    def __init__(self, model: nn.Module, layer: nn.Module) -> None:
        self.model = model
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._handle = layer.register_forward_hook(self._capture)

    def _capture(
        self, module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        self._activations = output

        def store(gradient: torch.Tensor) -> None:
            self._gradients = gradient

        if output.requires_grad:  # plain no-grad inference with the hook attached is fine
            output.register_hook(store)  # type: ignore[no-untyped-call]

    def __call__(self, images: torch.Tensor) -> NDArray[np.float32]:
        """Return maps of shape (N, H, W) for a normalised batch of shape (N, 3, H, W)."""
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        inputs = images.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            logits = self.model(inputs).reshape(-1)
            # Samples are independent in eval mode, so one backward pass of the sum gives
            # every image the gradient of its own logit.
            logits.sum().backward()
        if self._activations is None or self._gradients is None:
            raise RuntimeError("Grad-CAM hooks did not fire")
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam = functional.relu((weights * self._activations).sum(dim=1, keepdim=True))
        cam = functional.interpolate(
            cam, size=images.shape[-2:], mode="bilinear", align_corners=False
        ).squeeze(1)
        peak = cam.flatten(1).amax(dim=1).clamp_min(1e-12)[:, None, None]
        cam = torch.where(peak > 1e-8, cam / peak, torch.zeros_like(cam))
        self._activations = self._gradients = None
        return cam.detach().cpu().numpy().astype(np.float32)

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
