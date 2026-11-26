"""Small, local datasets for pipeline smoke tests."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw


@pytest.fixture
def tiny_crack_images(tmp_path: Path) -> Path:
    """Create 16 cracked and 16 intact synthetic 64-by-64 RGB images."""
    root = tmp_path / "cracks"
    rng = np.random.default_rng(42)
    for label in ("Positive", "Negative"):
        class_dir = root / label
        class_dir.mkdir(parents=True)
        for image_number in range(16):
            noise = rng.normal(loc=150, scale=12, size=(64, 64))
            image = Image.fromarray(noise.clip(0, 255).astype(np.uint8)).convert("RGB")
            if label == "Positive":
                x = int(rng.integers(12, 44))
                ImageDraw.Draw(image).line(
                    [(x, 0), (x + 7, 18), (x + 2, 34), (x + 11, 63)],
                    fill=(35, 35, 35),
                    width=2,
                )
            image.save(class_dir / f"{image_number:05}.jpg")
    return root
