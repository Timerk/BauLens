"""Image preprocessing, augmentation, datasets and seeded data loaders."""

import random
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2  # type: ignore[import-untyped]

from baulens.config import CracksConfig

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

type Sample = tuple[torch.Tensor, torch.Tensor]
type ImageTransform = Callable[[Image.Image], torch.Tensor]
type Perturbation = Callable[[Image.Image, int], Image.Image]


def preprocess(
    image: Image.Image,
    image_size: int,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> NDArray[np.float32]:
    """Resize the whole image to a square and normalise it to a CHW float32 array.

    Evaluation, export and the inference API all use this function, so they see identical
    inputs. Non-square images are squashed rather than cropped.
    """
    resized = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
    pixels = np.asarray(resized, dtype=np.float32) / 255.0
    mean_array = np.asarray(mean, dtype=np.float32)
    std_array = np.asarray(std, dtype=np.float32)
    normalised = ((pixels - mean_array) / std_array).astype(np.float32)
    return np.ascontiguousarray(normalised.transpose(2, 0, 1))


class EvalTransform:
    def __init__(self, image_size: int) -> None:
        self.image_size = image_size

    def __call__(self, image: Image.Image) -> torch.Tensor:
        return torch.from_numpy(preprocess(image, self.image_size))


class RandomRot90(torch.nn.Module):
    """Rotate by a random multiple of 90 degrees: crack orientation carries no class signal."""

    def forward(self, image: Image.Image) -> Image.Image:
        turns = int(torch.randint(0, 4, ()).item())
        if turns == 0:
            return image
        transposes = (
            Image.Transpose.ROTATE_90,
            Image.Transpose.ROTATE_180,
            Image.Transpose.ROTATE_270,
        )
        return image.transpose(transposes[turns - 1])


def train_transform(config: CracksConfig) -> v2.Compose:
    """Augmentations that keep the surface physically plausible.

    Flips and 90-degree rotations: patches have no canonical orientation. Mild random resized
    crops (aspect ratio near 1): distance to the surface varies, a crack may sit off-centre.
    Mild colour jitter: lighting and concrete tone vary; saturation and hue stay small because
    concrete is nearly grey and colour is not a crack cue.
    """
    jitter = config.color_jitter
    return v2.Compose(
        [
            v2.RandomResizedCrop(
                config.image_size, scale=(config.crop_scale_min, 1.0), ratio=(0.8, 1.25)
            ),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            RandomRot90(),
            v2.ColorJitter(brightness=jitter, contrast=jitter, saturation=jitter / 2, hue=0.02),
            v2.PILToTensor(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class CrackImageDataset(Dataset[Sample]):
    """Images listed by relative path below ``root`` with binary labels (1 = crack)."""

    def __init__(
        self,
        root: Path,
        paths: Sequence[str],
        labels: Sequence[int],
        transform: ImageTransform,
        perturbation: Perturbation | None = None,
    ) -> None:
        if len(paths) != len(labels):
            raise ValueError("paths and labels differ in length")
        self.root = root
        self.paths = list(paths)
        self.labels = [int(label) for label in labels]
        self.transform = transform
        self.perturbation = perturbation

    @classmethod
    def from_frame(
        cls,
        root: Path,
        frame: pd.DataFrame,
        transform: ImageTransform,
        perturbation: Perturbation | None = None,
    ) -> "CrackImageDataset":
        return cls(root, frame["path"].tolist(), frame["label"].tolist(), transform, perturbation)

    def __len__(self) -> int:
        return len(self.paths)

    def load_image(self, index: int) -> Image.Image:
        with Image.open(self.root / self.paths[index]) as image:
            return image.convert("RGB")

    def __getitem__(self, index: int) -> Sample:
        image = self.load_image(index)
        if self.perturbation is not None:
            image = self.perturbation(image, index)
        return self.transform(image), torch.tensor(self.labels[index], dtype=torch.float32)


def _seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def make_loader(
    dataset: Dataset[Sample],
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader[Sample]:
    """Seeded loader; worker processes are spawn-safe on Windows and kept alive across epochs."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        worker_init_fn=_seed_worker,
        generator=generator,
    )


def stratified_sample(frame: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    """Deterministic per-class subsample; returns the frame unchanged when fraction is 1."""
    if fraction >= 1:
        return frame
    parts = [
        part.sample(frac=fraction, random_state=seed)
        for _, part in frame.groupby("label", sort=True)
    ]
    return pd.concat(parts).sort_values("path").reset_index(drop=True)


def stratified_head(frame: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    """At most ``count`` rows with the class ratio of ``frame``, drawn reproducibly."""
    if count >= len(frame):
        return frame.reset_index(drop=True)
    return stratified_sample(frame, count / len(frame), seed)
