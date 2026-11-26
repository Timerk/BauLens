"""Split by the exact formulation and keep entire age series together."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.model_selection import GroupKFold, KFold, train_test_split  # type: ignore[import-untyped]

from baulens.config import StrengthConfig
from baulens.data.concrete import load_concrete
from baulens.strength.features import INGREDIENTS, TARGET

type Indices = NDArray[np.intp]
type Fold = tuple[Indices, Indices]


@dataclass(frozen=True)
class Partitions:
    fit: Indices
    calibration: Indices
    test: Indices
    strength_bins: int


def prepare_data(path: Path) -> tuple[pd.DataFrame, int]:
    """Remove exact duplicates while keeping distinct measurements of the same mix and age."""
    raw = load_concrete(path)
    frame = raw.drop_duplicates(ignore_index=True)
    return frame, len(raw) - len(frame)


def mix_groups(frame: pd.DataFrame) -> pd.Series[int]:
    """Assign a deterministic integer to each exact tuple of seven ingredients."""
    return frame.groupby(list(INGREDIENTS), sort=True).ngroup()


def split_partitions(
    frame: pd.DataFrame, groups: pd.Series[int], config: StrengthConfig
) -> Partitions:
    """Stratify the frozen holdout by group-mean strength, then randomise calibration groups."""
    means = frame[TARGET].groupby(groups).mean()
    test_count = int(np.ceil(config.test_fraction * len(means)))
    labels = None
    bin_count = 1
    for proposed_bins in range(min(5, test_count, len(means) - test_count), 1, -1):
        bins = pd.qcut(means, q=proposed_bins, labels=False, duplicates="drop")
        counts = bins.value_counts()
        if len(counts) >= 2 and counts.min() >= 2:
            labels, bin_count = bins, len(counts)
            break
    dev, test = train_test_split(
        means.index.to_numpy(),
        test_size=config.test_fraction,
        random_state=config.seed,
        stratify=labels,
    )
    fit, cal = train_test_split(
        dev, test_size=config.calibration_fraction, random_state=config.seed
    )
    if len(fit) < max(3, config.cv_splits):
        raise ValueError("Too few fitting mixes for grouped CV")
    return Partitions(
        fit=np.flatnonzero(groups.isin(fit)),
        calibration=np.flatnonzero(groups.isin(cal)),
        test=np.flatnonzero(groups.isin(test)),
        strength_bins=bin_count,
    )


def fold_indices(groups: pd.Series[int], splits: int, seed: int, *, grouped: bool) -> list[Fold]:
    """Use shuffled GroupKFold for generalisation and shuffled KFold for the leakage comparison."""
    if grouped:
        return list(
            GroupKFold(n_splits=splits, shuffle=True, random_state=seed).split(
                groups, groups=groups
            )
        )
    return list(KFold(n_splits=splits, shuffle=True, random_state=seed).split(groups))
