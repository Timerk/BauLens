"""Keep run outputs reproducible and publish completed files atomically."""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path, PureWindowsPath
from tempfile import NamedTemporaryFile

from baulens.config import StrengthConfig

MODEL_PATH = "model/pipeline.joblib"
METADATA_PATH = "model/metadata.json"
CV_SUMMARY_PATH = "cv_summary.json"
CV_FOLDS_PATH = "cv_folds.csv"
MANIFEST_PATH = "model/manifest.json"
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


def publish_fit(staging: Path, root: Path) -> None:
    """Commit the checksums last so readers can reject a partially published fit."""
    checksums = {name: sha256((staging / name).read_bytes()).hexdigest() for name in FIT_FILES}
    write_json(staging / MANIFEST_PATH, checksums)
    (root / "model").mkdir(exist_ok=True)
    for name in (*FIT_FILES, MANIFEST_PATH):
        (staging / name).replace(root / name)


def read_fit_files(model_dir: Path, paths: tuple[str, ...]) -> dict[str, bytes]:
    """Read each requested file once and verify it against the committed manifest."""
    try:
        checksums = json.loads((model_dir / Path(MANIFEST_PATH).name).read_bytes())
        files = {
            path: (
                model_dir / Path(path).name
                if path.startswith("model/")
                else model_dir.parent / path
            ).read_bytes()
            for path in paths
        }
        if not isinstance(checksums, dict) or any(
            sha256(content).hexdigest() != checksums.get(path) for path, content in files.items()
        ):
            raise ValueError("checksums do not match")
    except (OSError, ValueError) as error:
        raise ValueError(
            "Fitted artifact files do not match the manifest. Run strength train again"
        ) from error
    return files
