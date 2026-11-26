"""Keep run outputs reproducible and publish completed files atomically."""

import json
import re
from collections.abc import Iterator
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path, PureWindowsPath
from shutil import rmtree
from tempfile import NamedTemporaryFile
from uuid import uuid4

from baulens.config import StrengthConfig

MODEL_PATH = "pipeline.joblib"
METADATA_PATH = "metadata.json"
CV_SUMMARY_PATH = "cv_summary.json"
CV_FOLDS_PATH = "cv_folds.csv"
MANIFEST_PATH = "manifest.json"
CURRENT_PATH = "current.json"
FIT_FILES = (MODEL_PATH, METADATA_PATH, CV_SUMMARY_PATH, CV_FOLDS_PATH)


def run_identity(config: StrengthConfig, data_path: Path) -> tuple[str, str]:
    data_sha256 = sha256(data_path.read_bytes()).hexdigest()
    # Canonicalise separators so checkout-relative YAML paths hash the same on Windows and Linux.
    settings = config.model_dump()
    config_json = json.dumps(
        {
            key: str(PureWindowsPath(value)) if isinstance(value, Path) else value
            for key, value in settings.items()
        },
        sort_keys=True,
    )
    version = sha256(f"{config_json}\n{data_sha256}".encode()).hexdigest()[:12]
    return version, data_sha256


@contextmanager
def atomic_file(destination: Path) -> Iterator[Path]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=destination.parent, suffix=destination.suffix, delete=False
    ) as file:
        staged = Path(file.name)
    try:
        yield staged
        staged.replace(destination)
    finally:
        staged.unlink(missing_ok=True)


def write_json(destination: Path, content: object) -> None:
    with atomic_file(destination) as staged:
        staged.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def publish_fit(staging: Path, model_dir: Path) -> None:
    """Publish a complete generation before switching the reader's pointer."""
    model_dir = model_dir.resolve()
    staging = staging.resolve()
    generations = model_dir / "generations"
    generations.mkdir(exist_ok=True)
    if staging.parent != model_dir or generations.resolve() != generations:
        raise ValueError("Fit staging and generations must be inside the model directory")

    previous = _current_generations(model_dir)
    checksums = {name: sha256((staging / name).read_bytes()).hexdigest() for name in FIT_FILES}
    write_json(staging / MANIFEST_PATH, checksums)
    generation = uuid4().hex
    destination = generations / generation
    staging.rename(destination)
    retained = [generation, *previous[:1]]
    try:
        write_json(model_dir / CURRENT_PATH, {"generations": retained})
    except OSError:
        if destination.resolve() == destination:
            rmtree(destination, ignore_errors=True)
        raise

    # Only prune after switching the pointer. A later publish can retry a failed cleanup.
    for old in generations.iterdir():
        if (
            _is_generation_id(old.name)
            and old.name not in retained
            and old.resolve() == old
            and old.is_dir()
        ):
            rmtree(old, ignore_errors=True)


def read_fit_files(model_dir: Path, paths: tuple[str, ...]) -> dict[str, bytes]:
    """Verify files from one published generation, or an existing pre-generation fit."""
    try:
        generations = _current_generations(model_dir)
        while generations:
            directory = model_dir / "generations" / generations[0]
            try:
                return _verified_files(directory, paths)
            except FileNotFoundError:
                # Two successive refits may have pruned our snapshot before the read finished.
                latest = _current_generations(model_dir)
                if latest == generations:
                    raise
                generations = latest
        return _verified_files(model_dir, paths, legacy=True)
    except (OSError, ValueError) as error:
        raise ValueError(
            "Fitted artifact files do not match the manifest. Run strength train again"
        ) from error


def _current_generations(model_dir: Path) -> list[str]:
    try:
        pointer = json.loads((model_dir / CURRENT_PATH).read_bytes())
    except FileNotFoundError:
        return []
    ids = pointer.get("generations") if isinstance(pointer, dict) else None
    if (
        not isinstance(ids, list)
        or not 1 <= len(ids) <= 2
        or not all(_is_generation_id(item) for item in ids)
        or len(set(ids)) != len(ids)
    ):
        raise ValueError("Invalid fitted model generation pointer")
    return list(ids)


def _is_generation_id(name: object) -> bool:
    return isinstance(name, str) and re.fullmatch(r"[0-9a-f]{32}", name) is not None


def _verified_files(
    directory: Path, paths: tuple[str, ...], *, legacy: bool = False
) -> dict[str, bytes]:
    checksums = json.loads((directory / MANIFEST_PATH).read_bytes())
    files = {}
    for path in paths:
        file_path, checksum_key = directory / path, path
        if legacy:
            if path in (MODEL_PATH, METADATA_PATH):
                checksum_key = f"model/{path}"
            else:
                file_path = directory.parent / path
        content = file_path.read_bytes()
        if not isinstance(checksums, dict) or sha256(content).hexdigest() != checksums.get(
            checksum_key
        ):
            raise ValueError("checksums do not match")
        files[path] = content
    return files
