"""Tiny synthetic crack images for offline smoke tests and CI runs."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def make_synthetic_cracks(
    root: Path, *, per_class: int = 16, image_size: int = 64, seed: int = 42
) -> Path:
    """Write ``per_class`` noisy RGB images into ``root/Positive`` and ``root/Negative``.

    Positive images get a dark polyline as a stand-in crack. The layout matches the
    Özgenel dataset, so the same loaders work on both.
    """
    rng = np.random.default_rng(seed)
    for label in ("Positive", "Negative"):
        class_dir = root / label
        class_dir.mkdir(parents=True, exist_ok=True)
        for image_number in range(per_class):
            noise = rng.normal(loc=150, scale=12, size=(image_size, image_size))
            image = Image.fromarray(noise.clip(0, 255).astype(np.uint8)).convert("RGB")
            if label == "Positive":
                scale = image_size / 64
                x = int(rng.integers(12, 44))
                points = [(x, 0), (x + 7, 18), (x + 2, 34), (x + 11, 63)]
                ImageDraw.Draw(image).line(
                    [(px * scale, py * scale) for px, py in points],
                    fill=(35, 35, 35),
                    width=max(2, round(2 * scale)),
                )
            image.save(class_dir / f"{image_number:05}.jpg")
    return root
