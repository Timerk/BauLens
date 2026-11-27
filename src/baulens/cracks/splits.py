"""Image index, duplicate (leakage) check and group-aware stratified split.

Özgenel's 227 px patches were cropped from 458 high-resolution photos whose identifiers are not
published, so patches from the same photo cannot be grouped exactly. As a proxy we link images
that are byte-identical, or whose 64-bit DCT perceptual hashes differ in at most
``near_duplicate_max_distance`` bits *and* whose 32x32 grey thumbnails correlate with Pearson
r >= ``near_duplicate_min_correlation``. The hash alone is not specific enough on crack patches:
two different cracks with a similar course share most low-frequency bits. Connected components
of the confirmed pairs become groups, and whole groups are assigned to one split.

This removes (near-)identical copies across splits. It does not separate distinct,
non-overlapping patches of the same photo, which share lighting, camera and surface, and it
does not detect rotated or mirrored copies.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from PIL import Image
from scipy.fft import dctn  # type: ignore[import-untyped]
from scipy.sparse import coo_matrix  # type: ignore[import-untyped]
from scipy.sparse.csgraph import connected_components  # type: ignore[import-untyped]
from sklearn.model_selection import train_test_split  # type: ignore[import-untyped]

from baulens.config import CracksConfig
from baulens.paths import project_path

CLASS_DIRS: dict[str, int] = {"Negative": 0, "Positive": 1}
SPLITS = ("train", "val", "test")
_HASH_BITS = 64
_DISTANCE_GRID = (0, 2, 4, 6, 8, 10, 12)
_CORRELATION_GRID = (0.8, 0.85, 0.9, 0.95, 0.98)
_THUMB = 32


@dataclass(frozen=True)
class SplitFiles:
    directory: Path
    index: Path
    thumbnails: Path
    manifest: Path
    leakage: Path
    settings: Path


def split_files(config: CracksConfig) -> SplitFiles:
    directory = project_path(config.artifacts_dir) / "splits"
    return SplitFiles(
        directory=directory,
        index=directory / "image_index.csv",
        thumbnails=directory / "thumbnails.npy",
        manifest=directory / "manifest.csv",
        leakage=directory / "leakage.json",
        settings=directory / "split_settings.json",
    )


def thumbnail(image: Image.Image) -> NDArray[np.float32]:
    """32x32 grey thumbnail used for both the perceptual hash and pair confirmation."""
    grey = image.convert("L").resize((_THUMB, _THUMB), Image.Resampling.LANCZOS)
    return np.asarray(grey, dtype=np.float32)


def perceptual_hash(thumb: NDArray[np.float32]) -> int:
    """64-bit DCT hash: signs of the 8x8 lowest frequencies against their median."""
    coefficients = dctn(thumb.astype(np.float64), norm="ortho")[:8, :8].ravel()
    bits = coefficients > np.median(coefficients[1:])
    return int.from_bytes(np.packbits(bits).tobytes(), "big")


def list_images(data_dir: Path) -> list[tuple[Path, int]]:
    images: list[tuple[Path, int]] = []
    for class_name, label in CLASS_DIRS.items():
        class_dir = data_dir / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"missing class directory {class_dir}")
        images.extend((path, label) for path in sorted(class_dir.glob("*.jpg")))
    if not images:
        raise FileNotFoundError(f"no JPEG images below {data_dir}")
    return images


def scan_contents(data_dir: Path) -> pd.DataFrame:
    """Path, label and SHA-256 of every image currently on disk (reads all bytes)."""
    rows = []
    for path, label in list_images(data_dir):
        with path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        rows.append(
            {"path": path.relative_to(data_dir).as_posix(), "label": label, "sha256": digest}
        )
    return pd.DataFrame(rows)


def index_images(
    data_dir: Path, entries: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, NDArray[np.float32]]:
    """Read images once: hashes, size, mode, intensity statistics and a grey thumbnail.

    ``entries`` (columns path, label) restricts indexing to those files; default: all images.
    """
    if entries is None:
        listed = [(path, label) for path, label in list_images(data_dir)]
    else:
        listed = [
            (data_dir / path, int(label))
            for path, label in zip(entries["path"], entries["label"], strict=True)
        ]
    rows: list[dict[str, Any]] = []
    thumbs: list[NDArray[np.float32]] = []
    for path, label in listed:
        payload = path.read_bytes()
        with Image.open(path) as image:
            mode = image.mode
            rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
            thumbs.append(thumbnail(image))
        rows.append(
            {
                "path": path.relative_to(data_dir).as_posix(),
                "label": label,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "phash": f"{perceptual_hash(thumbs[-1]):016x}",
                "width": rgb.shape[1],
                "height": rgb.shape[0],
                "mode": mode,
                "mean_r": float(rgb[..., 0].mean()),
                "mean_g": float(rgb[..., 1].mean()),
                "mean_b": float(rgb[..., 2].mean()),
                "grey_std": float(rgb.mean(axis=2).std()),
            }
        )
    empty = np.zeros((0, _THUMB, _THUMB), dtype=np.float32)
    return pd.DataFrame(rows), np.stack(thumbs) if thumbs else empty


def load_or_index(
    data_dir: Path, files: SplitFiles, contents: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, NDArray[np.float32]]:
    """Index the images on disk, reusing cached rows only where path, label *and* SHA-256 match.

    Changed, added or replaced files are re-indexed, so a cached hash can never describe
    different bytes than the file now on disk.
    """
    contents = scan_contents(data_dir) if contents is None else contents
    cached: pd.DataFrame | None = None
    cached_thumbs: NDArray[np.float32] | None = None
    if files.index.exists() and files.thumbnails.exists():
        cached = pd.read_csv(files.index, dtype={"phash": str})
        cached_thumbs = np.load(files.thumbnails)
        if len(cached) != len(cached_thumbs):
            cached = cached_thumbs = None
    key = ["path", "label", "sha256"]
    if cached is not None and cached_thumbs is not None:
        position = pd.Series(np.arange(len(cached)), index=pd.MultiIndex.from_frame(cached[key]))
        found = position.reindex(pd.MultiIndex.from_frame(contents[key]))
    else:
        found = pd.Series(np.nan, index=pd.MultiIndex.from_frame(contents[key]))
    missing = contents[found.isna().to_numpy()]
    if cached is not None and cached_thumbs is not None and missing.empty:
        order = found.to_numpy(dtype=np.int64)
        if np.array_equal(order, np.arange(len(cached))):
            return cached, cached_thumbs
    fresh, fresh_thumbs = index_images(data_dir, missing[["path", "label"]])
    parts, thumb_parts = [], []
    if cached is not None and cached_thumbs is not None:
        reuse = found.dropna().to_numpy(dtype=np.int64)
        parts.append(cached.iloc[reuse])
        thumb_parts.append(cached_thumbs[reuse])
    parts.append(fresh)
    thumb_parts.append(fresh_thumbs)
    combined = pd.concat(parts, ignore_index=True)
    combined_thumbs = np.concatenate(thumb_parts)
    order = combined.reset_index().set_index("path").loc[contents["path"], "index"].to_numpy()
    index = combined.iloc[order].reset_index(drop=True)
    thumbs = combined_thumbs[order]
    files.directory.mkdir(parents=True, exist_ok=True)
    index.to_csv(files.index, index=False, lineterminator="\n")
    np.save(files.thumbnails, thumbs)
    return index, thumbs


def thumbnail_correlation(
    thumbs: NDArray[np.float32], left: NDArray[np.int64], right: NDArray[np.int64]
) -> NDArray[np.float64]:
    """Pearson correlation of the grey thumbnails of each pair (1 = same up to gain/offset)."""
    flat = thumbs.reshape(len(thumbs), -1).astype(np.float64)
    flat = flat - flat.mean(axis=1, keepdims=True)
    flat /= np.maximum(np.linalg.norm(flat, axis=1, keepdims=True), 1e-9)
    return np.asarray(np.einsum("ij,ij->i", flat[left], flat[right]), dtype=np.float64)


def hash_array(index: pd.DataFrame) -> NDArray[np.uint64]:
    return np.array([int(value, 16) for value in index["phash"]], dtype=np.uint64)


def hamming_neighbours(
    hashes: NDArray[np.uint64], max_distance: int, chunk_size: int = 256
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.uint8]]:
    """Return all pairs ``i < j`` within ``max_distance`` bits and each image's NN distance."""
    count = len(hashes)
    nearest = np.full(count, _HASH_BITS, dtype=np.uint8)
    left: list[NDArray[np.int64]] = []
    right: list[NDArray[np.int64]] = []
    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        distances = np.bitwise_count(hashes[start:stop, None] ^ hashes[None, :]).astype(np.uint8)
        rows = np.arange(stop - start)
        distances[rows, rows + start] = _HASH_BITS + 1
        nearest[start:stop] = distances.min(axis=1)
        i, j = np.nonzero(distances <= max_distance)
        keep = j > i + start
        left.append((i[keep] + start).astype(np.int64))
        right.append(j[keep].astype(np.int64))
    return np.concatenate(left), np.concatenate(right), nearest


@dataclass(frozen=True)
class CandidatePairs:
    """Image pairs within the hash radius, with the evidence used to confirm them."""

    left: NDArray[np.int64]
    right: NDArray[np.int64]
    distance: NDArray[np.int64]
    correlation: NDArray[np.float64]
    identical: NDArray[np.bool_]
    nearest: NDArray[np.uint8]

    def subset(self, mask: NDArray[np.bool_]) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        return self.left[mask], self.right[mask]


def candidate_pairs(
    index: pd.DataFrame, thumbs: NDArray[np.float32], max_distance: int
) -> CandidatePairs:
    hashes = hash_array(index)
    left, right, nearest = hamming_neighbours(hashes, max_distance)
    sha = index["sha256"].to_numpy()
    return CandidatePairs(
        left=left,
        right=right,
        distance=np.bitwise_count(hashes[left] ^ hashes[right]).astype(np.int64),
        correlation=thumbnail_correlation(thumbs, left, right),
        identical=np.asarray(sha[left] == sha[right], dtype=np.bool_),
        nearest=nearest,
    )


def confirmed(
    pairs: CandidatePairs, max_distance: int, min_correlation: float
) -> NDArray[np.bool_]:
    """Byte-identical files, or hash neighbours whose thumbnails correlate strongly."""
    near = (pairs.distance <= max_distance) & (pairs.correlation >= min_correlation)
    return np.asarray(pairs.identical | near, dtype=np.bool_)


def duplicate_groups(
    count: int, pairs: tuple[NDArray[np.int64], NDArray[np.int64]]
) -> NDArray[np.int64]:
    """Connected components over confirmed duplicate pairs (singletons get their own group)."""
    left, right = pairs
    graph = coo_matrix((np.ones(len(left), dtype=np.int8), (left, right)), shape=(count, count))
    _, labels = connected_components(graph, directed=False)
    return np.asarray(labels, dtype=np.int64)


def group_stratified_split(
    labels: NDArray[np.int64],
    groups: NDArray[np.int64],
    *,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> NDArray[np.str_]:
    """Assign whole groups to train/val/test, stratified by each group's majority label."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({"label": labels, "group": groups})
    by_group = frame.groupby("group")["label"].agg(["mean", "size"])
    group_label = (by_group["mean"] >= 0.5).astype(int)
    assignment: dict[int, str] = {}
    for label in sorted(group_label.unique()):
        members = by_group.index[group_label == label].to_numpy()
        members = members[rng.permutation(len(members))]
        sizes = by_group.loc[members, "size"].to_numpy()
        total = sizes.sum()
        test_target, val_target = test_fraction * total, val_fraction * total
        filled = {"test": 0, "val": 0}
        for group, size in zip(members, sizes, strict=True):
            if filled["test"] < test_target:
                split = "test"
            elif filled["val"] < val_target:
                split = "val"
            else:
                split = "train"
            if split != "train":
                filled[split] += int(size)
            assignment[int(group)] = split
    return np.array([assignment[int(group)] for group in groups])


def naive_split(labels: NDArray[np.int64], config: CracksConfig) -> NDArray[np.str_]:
    """Plain stratified random split, for comparison only."""
    held_out = config.val_fraction + config.test_fraction
    train, rest = train_test_split(
        np.arange(len(labels)), test_size=held_out, stratify=labels, random_state=config.seed
    )
    val, test = train_test_split(
        rest,
        test_size=config.test_fraction / held_out,
        stratify=labels[rest],
        random_state=config.seed,
    )
    splits = np.empty(len(labels), dtype="<U5")
    splits[train], splits[val], splits[test] = "train", "val", "test"
    return splits


def _cross(splits: NDArray[np.str_], pairs: tuple[NDArray[np.int64], NDArray[np.int64]]) -> int:
    left, right = pairs
    return int(np.count_nonzero(splits[left] != splits[right]))


def leakage_report(
    index: pd.DataFrame,
    config: CracksConfig,
    pairs: CandidatePairs,
    groups: NDArray[np.int64],
    splits: NDArray[np.str_],
) -> dict[str, Any]:
    """Duplicates and near-duplicates, how many would cross a naive split, image statistics."""
    labels = index["label"].to_numpy(dtype=np.int64)
    naive = naive_split(labels, config)

    def summary(mask: NDArray[np.bool_]) -> dict[str, int]:
        pair = pairs.subset(mask)
        return {
            "pairs": int(mask.sum()),
            "images_involved": len(np.unique(np.concatenate(pair))),
            "label_conflicting_pairs": int(np.count_nonzero(labels[pair[0]] != labels[pair[1]])),
            "cross_split_pairs_naive": _cross(naive, pair),
            "cross_split_pairs_grouped": _cross(splits, pair),
        }

    radius = config.near_duplicate_max_distance
    hash_only = [{"max_distance": d} | summary(pairs.distance <= d) for d in _DISTANCE_GRID]
    by_correlation = [
        {"min_correlation": c} | summary(confirmed(pairs, radius, c)) for c in _CORRELATION_GRID
    ]
    duplicates = confirmed(pairs, radius, config.near_duplicate_min_correlation)
    group_sizes = np.bincount(groups)
    duplicated = index["sha256"].duplicated(keep=False)
    per_class = {}
    for name, label in CLASS_DIRS.items():
        part = index[index["label"] == label]
        per_class[name] = {
            "images": len(part),
            "mean_rgb": [round(float(part[c].mean()), 2) for c in ("mean_r", "mean_g", "mean_b")],
            "mean_grey_std": round(float(part["grey_std"].mean()), 2),
            "exact_duplicate_files": int(duplicated[part.index].sum()),
        }
    split_counts = pd.crosstab(
        pd.Series(splits, name="split"), index["label"].map({0: "Negative", 1: "Positive"})
    ).reindex(SPLITS)
    return {
        "images": len(index),
        "sizes": sorted({f"{w}x{h}" for w, h in zip(index["width"], index["height"], strict=True)}),
        "modes": sorted(index["mode"].unique().tolist()),
        "per_class": per_class,
        "exact_duplicate_files": int(duplicated.sum()),
        "exact_duplicate_sets": int(index.loc[duplicated, "sha256"].nunique()),
        "exact": summary(pairs.identical),
        "hash_radius": radius,
        "min_correlation": config.near_duplicate_min_correlation,
        "confirmed": summary(duplicates),
        "confirmed_not_identical": summary(duplicates & ~pairs.identical),
        "hash_only_sensitivity": hash_only,
        "correlation_sensitivity": by_correlation,
        "groups": len(group_sizes),
        "groups_larger_than_one": int(np.count_nonzero(group_sizes > 1)),
        "largest_group": int(group_sizes.max()),
        "images_in_multi_image_groups": int(group_sizes[group_sizes > 1].sum()),
        "nearest_neighbour_hamming_histogram": np.bincount(
            pairs.nearest, minlength=_HASH_BITS + 1
        ).tolist(),
        "split_counts": {
            str(split): {str(name): int(count) for name, count in row.items()}
            for split, row in split_counts.to_dict(orient="index").items()
        },
    }


def _leakage_figure(
    config: CracksConfig, index: pd.DataFrame, data_dir: Path, pairs: CandidatePairs
) -> None:
    """Correlation of hash candidates and borderline accepted / rejected pairs."""
    from baulens.cracks import figures

    def load(i: int) -> Image.Image:
        with Image.open(data_dir / index["path"].iloc[i]) as image:
            return image.convert("RGB")

    names = {0: "neg", 1: "pos"}
    threshold = config.near_duplicate_min_correlation
    within = (pairs.distance <= config.near_duplicate_max_distance) & ~pairs.identical
    accepted = np.flatnonzero(within & (pairs.correlation >= threshold))
    rejected = np.flatnonzero(within & (pairs.correlation < threshold))
    examples = [
        *accepted[np.argsort(pairs.correlation[accepted])][:3],
        *rejected[np.argsort(-pairs.correlation[rejected])][:3],
    ]
    shown = []
    for position in examples:
        a, b = int(pairs.left[position]), int(pairs.right[position])
        verdict = "duplicate" if pairs.correlation[position] >= threshold else "distinct"
        kinds = "/".join(names[int(index["label"].iloc[i])] for i in (a, b))
        caption = (
            f"r={pairs.correlation[position]:.4f}, d={pairs.distance[position]}\n{kinds}: {verdict}"
        )
        shown.append((load(a), load(b), caption))
    figures.leakage(
        pairs.correlation[within],
        threshold,
        config.near_duplicate_max_distance,
        shown,
        figures.targets(config, "near_duplicates.png"),
    )


def split_settings(config: CracksConfig) -> dict[str, Any]:
    """Settings that determine the split, recorded with the manifest and the checkpoint."""
    return {
        "seed": config.seed,
        "val_fraction": config.val_fraction,
        "test_fraction": config.test_fraction,
        "near_duplicate_max_distance": config.near_duplicate_max_distance,
        "near_duplicate_min_correlation": config.near_duplicate_min_correlation,
    }


def _settings(config: CracksConfig, data_dir: Path, contents: pd.DataFrame) -> dict[str, Any]:
    return {"data_dir": str(data_dir), "data_checksum": data_checksum(contents)} | split_settings(
        config
    )


def build_splits(config: CracksConfig, contents: pd.DataFrame | None = None) -> pd.DataFrame:
    """Index the images, group duplicates, split and write manifest and leakage report."""
    files = split_files(config)
    data_dir = project_path(config.data_dir)
    contents = scan_contents(data_dir) if contents is None else contents
    index, thumbs = load_or_index(data_dir, files, contents)
    pairs = candidate_pairs(index, thumbs, max(config.near_duplicate_max_distance, *_DISTANCE_GRID))
    duplicates = confirmed(
        pairs, config.near_duplicate_max_distance, config.near_duplicate_min_correlation
    )
    groups = duplicate_groups(len(index), pairs.subset(duplicates))
    splits = group_stratified_split(
        index["label"].to_numpy(dtype=np.int64),
        groups,
        val_fraction=config.val_fraction,
        test_fraction=config.test_fraction,
        seed=config.seed,
    )
    manifest = index[["path", "label", "sha256", "phash"]].assign(group=groups, split=splits)
    manifest.to_csv(files.manifest, index=False, lineterminator="\n")
    report = leakage_report(index, config, pairs, groups, splits)
    files.leakage.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _leakage_figure(config, index, data_dir, pairs)
    settings = _settings(config, data_dir, contents)
    files.settings.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    return manifest


def ensure_splits(config: CracksConfig) -> pd.DataFrame:
    """Return the manifest for the current file *contents* and split settings.

    Every call hashes all images. The cached manifest is reused only when the content
    checksum and the split settings both match what it was built from; otherwise the index is
    refreshed for changed files and the grouping and split are rebuilt.
    """
    files = split_files(config)
    data_dir = project_path(config.data_dir)
    contents = scan_contents(data_dir)
    expected = _settings(config, data_dir, contents)
    if (
        files.manifest.exists()
        and files.settings.exists()
        and json.loads(files.settings.read_text(encoding="utf-8")) == expected
    ):
        manifest = pd.read_csv(files.manifest, dtype={"phash": str})
        if data_checksum(manifest) == expected["data_checksum"]:
            return manifest
    return build_splits(config, contents)


def data_checksum(manifest: pd.DataFrame) -> str:
    """SHA-256 over the sorted (path, file hash) pairs, independent of the split."""
    digest = hashlib.sha256()
    for path, sha in sorted(zip(manifest["path"], manifest["sha256"], strict=True)):
        digest.update(f"{path}\t{sha}\n".encode())
    return digest.hexdigest()


def split_checksum(manifest: pd.DataFrame) -> str:
    """SHA-256 over the sorted (path, split) assignment."""
    digest = hashlib.sha256()
    for path, split in sorted(zip(manifest["path"], manifest["split"], strict=True)):
        digest.update(f"{path}\t{split}\n".encode())
    return digest.hexdigest()


def ordered_samples_checksum(train: pd.DataFrame, val: pd.DataFrame) -> str:
    """SHA-256 of the ordered path and label pairs supplied to the two loaders."""
    digest = hashlib.sha256()
    for split, frame in (("train", train), ("val", val)):
        digest.update(f"{split}\n".encode())
        for path, label in zip(frame["path"], frame["label"], strict=True):
            digest.update(json.dumps([path, int(label)]).encode() + b"\n")
    return digest.hexdigest()
