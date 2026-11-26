"""Small, local datasets for pipeline smoke tests."""

from pathlib import Path

import pytest

from baulens.data.synthetic import make_synthetic_cracks


@pytest.fixture
def tiny_crack_images(tmp_path: Path) -> Path:
    """Create 16 cracked and 16 intact synthetic 64-by-64 RGB images."""
    return make_synthetic_cracks(tmp_path / "cracks", per_class=16, image_size=64, seed=42)
