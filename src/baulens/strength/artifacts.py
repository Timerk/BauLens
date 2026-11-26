"""Keep run outputs reproducible and publish completed files atomically."""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path, PureWindowsPath
from tempfile import NamedTemporaryFile

from baulens.config import StrengthConfig


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
