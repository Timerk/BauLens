from pathlib import Path

import pytest
from PIL import Image

from baulens.config import CracksConfig, StrengthConfig, load_config
from baulens.data.concrete import CONCRETE_COLUMNS, load_concrete
from baulens.paths import project_path


@pytest.mark.smoke
def test_load_concrete_fixture() -> None:
    config = load_config(Path("configs/smoke/strength.yaml"), StrengthConfig)
    frame = load_concrete(project_path(config.data_path))
    assert frame.shape == (60, 9)
    assert tuple(frame.columns) == CONCRETE_COLUMNS
    assert frame["age_days"].between(1, 365).all()
    assert (frame["compressive_strength_mpa"] > 0).all()


@pytest.mark.smoke
def test_load_synthetic_cracks(tiny_crack_images: Path) -> None:
    config = load_config(Path("configs/smoke/cracks.yaml"), CracksConfig)
    assert not config.pretrained
    # The smoke profile's data_dir is generated on demand; tests point it at the fixture.
    config = config.model_copy(update={"data_dir": tiny_crack_images})
    for label in ("Positive", "Negative"):
        files = sorted((config.data_dir / label).glob("*.jpg"))
        assert len(files) == 16
        with Image.open(files[0]) as image:
            assert image.mode == "RGB"
            assert image.size == (config.image_size, config.image_size)
