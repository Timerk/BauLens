"""Immutable model generations behind a single atomically replaced pointer.

Layout below a model directory::

    model/
      current.json              {"generations": ["<newest id>", "<previous id>"]}
      generations/<id>/         immutable once published
        manifest.json           SHA-256 of every other file in the generation
        ...                     weights, graphs, metadata

A writer prepares all files in a staging directory inside ``model/``, then
:func:`publish_generation` writes the manifest, renames the staging directory to
``generations/<id>`` and replaces ``current.json`` with ``os.replace`` semantics
(``Path.replace``). Readers resolve the pointer **once** and read every file of that one
generation, verifying each against its manifest. A crash before the pointer switch leaves the
previous generation current. A crash after it leaves a complete new one. Old generations are
pruned after the switch; the newest ``keep`` stay so that a reader holding the previous
pointer can finish.

This mirrors ``baulens.strength.artifacts``; moving strength onto this module is left to a
follow-up.
"""

import json
import re
import shutil
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

CURRENT_FILE = "current.json"
MANIFEST_FILE = "manifest.json"
GENERATIONS_DIR = "generations"


class ArtifactError(RuntimeError):
    """No valid published generation, or its files do not match their manifest."""


def write_json_atomic(destination: Path, content: object) -> None:
    """Write strict JSON (no NaN) to a temporary file next to ``destination``, then replace."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=destination.parent, suffix=".tmp", delete=False, encoding="utf-8"
    ) as handle:
        staged = Path(handle.name)
        handle.write(json.dumps(content, indent=2, allow_nan=False) + "\n")
    try:
        staged.replace(destination)
    finally:
        staged.unlink(missing_ok=True)


@contextmanager
def staging_directory(model_dir: Path) -> Iterator[Path]:
    """A fresh directory inside ``model_dir``; removed again unless it was published."""
    model_dir.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=model_dir))
    try:
        yield staging
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _is_generation_id(name: object) -> bool:
    return isinstance(name, str) and re.fullmatch(r"[0-9a-f]{32}", name) is not None


def current_generations(model_dir: Path) -> list[str]:
    """Generation ids from the pointer, newest first; empty if nothing was published."""
    try:
        pointer = json.loads((model_dir / CURRENT_FILE).read_bytes())
    except FileNotFoundError:
        return []
    ids = pointer.get("generations") if isinstance(pointer, dict) else None
    if (
        not isinstance(ids, list)
        or not ids
        or not all(_is_generation_id(item) for item in ids)
        or len(set(ids)) != len(ids)
    ):
        raise ArtifactError(f"invalid generation pointer in {model_dir / CURRENT_FILE}")
    return list(ids)


def publish_generation(staging: Path, model_dir: Path, keep: int = 2) -> Path:
    """Checksum the staged files, publish them as a new generation and switch the pointer."""
    model_dir = model_dir.resolve()
    staging = staging.resolve()
    if staging.parent != model_dir:
        raise ArtifactError("staging directory must be inside the model directory")
    files = sorted(p.name for p in staging.iterdir() if p.is_file() and p.name != MANIFEST_FILE)
    if not files:
        raise ArtifactError("nothing staged to publish")
    checksums = {name: sha256((staging / name).read_bytes()).hexdigest() for name in files}
    write_json_atomic(staging / MANIFEST_FILE, checksums)
    previous = current_generations(model_dir)
    generations = model_dir / GENERATIONS_DIR
    generations.mkdir(exist_ok=True)
    generation = uuid4().hex
    destination = generations / generation
    staging.rename(destination)
    retained = [generation, *previous[: keep - 1]]
    try:
        write_json_atomic(model_dir / CURRENT_FILE, {"generations": retained})
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    for old in generations.iterdir():
        if _is_generation_id(old.name) and old.name not in retained and old.is_dir():
            shutil.rmtree(old, ignore_errors=True)
    return destination


def read_generation(
    model_dir: Path, names: Sequence[str], optional: Sequence[str] = ()
) -> tuple[Path, dict[str, bytes]]:
    """Resolve the pointer once and return verified bytes of ``names`` from that generation.

    ``optional`` files are returned when the generation contains them. Every returned file
    must match the generation's manifest.
    """
    ids = current_generations(model_dir)
    if not ids:
        raise ArtifactError(f"no published model generation in {model_dir}")
    directory = model_dir / GENERATIONS_DIR / ids[0]
    try:
        checksums = json.loads((directory / MANIFEST_FILE).read_bytes())
    except (OSError, ValueError) as error:
        raise ArtifactError(f"unreadable manifest in {directory}") from error
    if not isinstance(checksums, dict):
        raise ArtifactError(f"invalid manifest in {directory}")
    wanted = [*names, *(name for name in optional if name in checksums)]
    files: dict[str, bytes] = {}
    for name in wanted:
        try:
            content = (directory / name).read_bytes()
        except OSError as error:
            raise ArtifactError(f"{name} missing from generation {ids[0]}") from error
        if sha256(content).hexdigest() != checksums.get(name):
            raise ArtifactError(f"{name} in generation {ids[0]} does not match its manifest")
        files[name] = content
    return directory, files
