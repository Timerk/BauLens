"""Synthetic image corruptions as a cheap probe of sensitivity to capture conditions.

These are not a substitute for a second dataset: they mimic camera and processing effects
(defocus, compression, exposure, sensor noise, low resolution) on the same lab-like images.
"""

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


@dataclass(frozen=True)
class Corruption:
    """A named, deterministic corruption; ``seed`` makes the noise reproducible per image."""

    name: str
    level: float
    seed: int = 0

    @property
    def label(self) -> str:
        return f"{self.name} {self.level:g}"

    def __call__(self, image: Image.Image, index: int) -> Image.Image:
        match self.name:
            case "gaussian_blur":
                return image.filter(ImageFilter.GaussianBlur(radius=self.level))
            case "jpeg":
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=int(self.level))
                buffer.seek(0)
                with Image.open(buffer) as compressed:
                    return compressed.convert("RGB")
            case "brightness":
                return ImageEnhance.Brightness(image).enhance(self.level)
            case "contrast":
                return ImageEnhance.Contrast(image).enhance(self.level)
            case "gaussian_noise":
                rng = np.random.default_rng([self.seed, index])
                array = np.asarray(image, dtype=np.float32)
                noisy = array + rng.normal(0, self.level, array.shape)
                return Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8))
            case "downscale":
                width, height = image.size
                small = image.resize(
                    (max(1, round(width * self.level)), max(1, round(height * self.level))),
                    Image.Resampling.BILINEAR,
                )
                return small.resize((width, height), Image.Resampling.BILINEAR)
            case _:
                raise ValueError(f"unknown corruption {self.name!r}")


def default_corruptions(seed: int) -> list[Corruption]:
    """Two severities per corruption: mild (plausible field photo) and strong."""
    levels = {
        "gaussian_blur": (1.0, 2.5),  # radius in px at 227 px
        "jpeg": (30, 10),  # quality
        "brightness": (0.6, 1.5),  # factor
        "contrast": (0.6, 0.3),  # factor
        "gaussian_noise": (8.0, 20.0),  # sigma in grey levels
        "downscale": (0.5, 0.25),  # factor, upsampled back
    }
    return [Corruption(name, level, seed) for name, pair in levels.items() for level in pair]
