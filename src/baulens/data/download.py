"""Download and verify the official dataset archives."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Literal
from zipfile import BadZipFile, ZipFile

import httpx
from filelock import FileLock, Timeout
from pydantic import BaseModel, ConfigDict, HttpUrl, PositiveInt, RootModel, StringConstraints
from rich.console import Console
from rich.progress import DownloadColumn, Progress, TransferSpeedColumn
from yaml import YAMLError

from baulens.config import load_config
from baulens.paths import CONFIG_DIR, RAW_DATA_DIR

type DatasetName = Literal["concrete", "cracks"]
type Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
MANIFEST_PATH = CONFIG_DIR / "datasets.yaml"
DOWNLOAD_TIMEOUT = httpx.Timeout(60.0, connect=10.0)
EXTRACTION_MANIFEST = ".baulens-extraction.json"


class DatasetSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: HttpUrl
    filename: Annotated[str, StringConstraints(pattern=r"^[\w.-]+\.(zip|rar)$")]
    sha256: Sha256
    files: dict[str, Sha256] = {}
    jpeg_counts: dict[str, PositiveInt] = {}


class DatasetManifest(RootModel[dict[DatasetName, DatasetSpec]]):
    """Archive sources and the expected layout of each dataset."""


class DatasetError(RuntimeError):
    """A dataset could not be fetched, verified or extracted."""


def verify_checksum(path: Path, expected_sha256: str) -> bool:
    """Compare a file's SHA-256 digest with the expected hexadecimal value."""
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest() == expected_sha256.lower()


def download_dataset(
    name: DatasetName,
    *,
    raw_dir: Path = RAW_DATA_DIR,
    manifest_path: Path = MANIFEST_PATH,
    client: httpx.Client | None = None,
    console: Console | None = None,
) -> Path:
    """Return the local dataset directory, reusing the verified archive when present."""
    archive_dir = raw_dir.resolve()
    destination = archive_dir / name
    try:
        spec = load_config(manifest_path, DatasetManifest).root[name]
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive = archive_dir / spec.filename
        with FileLock(archive_dir / f".{name}.lock", timeout=300):
            if archive.exists():
                if not verify_checksum(archive, spec.sha256):
                    raise DatasetError(f"SHA-256 mismatch for {archive}. Move it aside and retry.")
            else:
                _download_archive(spec, archive, client, console)

            if _valid_extraction(destination, spec):
                if console is not None:
                    console.print(
                        f"{name}: verified archive and existing extraction.", markup=False
                    )
            else:
                if console is not None:
                    console.print(f"{name}: extracting {archive.name}.", markup=False)
                _install_extraction(archive, destination, spec)
    except (
        OSError,
        ValueError,
        KeyError,
        YAMLError,
        httpx.HTTPError,
        BadZipFile,
        subprocess.SubprocessError,
        Timeout,
    ) as error:
        raise DatasetError(f"{name}: {error}") from error
    return destination


def _download_archive(
    spec: DatasetSpec, archive: Path, client: httpx.Client | None, console: Console | None
) -> None:
    if console is not None:
        console.print(f"Downloading {archive.name}.", markup=False)
    connection = nullcontext(client) if client is not None else httpx.Client()
    with TemporaryDirectory(prefix=".download-", dir=archive.parent) as temp_dir:
        partial = Path(temp_dir) / archive.name
        with (
            connection as http,
            http.stream(
                "GET", str(spec.url), follow_redirects=True, timeout=DOWNLOAD_TIMEOUT
            ) as response,
            partial.open("wb") as target,
            Progress(
                *Progress.get_default_columns(),
                DownloadColumn(),
                TransferSpeedColumn(),
                console=console,
                disable=console is None,
            ) as progress,
        ):
            response.raise_for_status()
            length = response.headers.get("Content-Length")
            task = progress.add_task(archive.name, total=int(length) if length else None)
            for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                target.write(chunk)
                progress.advance(task, len(chunk))
        if not verify_checksum(partial, spec.sha256):
            raise DatasetError(f"SHA-256 mismatch for the download of {archive.name}.")
        partial.replace(archive)


def _valid_extraction(destination: Path, spec: DatasetSpec) -> bool:
    if not destination.is_dir():
        return False
    for filename, expected in spec.files.items():
        path = destination / filename
        if not path.is_file() or not verify_checksum(path, expected):
            return False
    for dirname, expected_count in spec.jpeg_counts.items():
        directory = destination / dirname
        if not directory.is_dir():
            return False
        with os.scandir(directory) as entries:
            count = sum(entry.name.endswith(".jpg") and entry.is_file() for entry in entries)
        if count != expected_count:
            return False
    return _matches_extraction_manifest(destination, spec)


def _hash_tree(root: Path) -> dict[str, str]:
    """Return SHA-256 digests of every extracted file keyed by POSIX-relative path."""
    digests: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != EXTRACTION_MANIFEST:
            with path.open("rb") as source:
                digests[path.relative_to(root).as_posix()] = hashlib.file_digest(
                    source, "sha256"
                ).hexdigest()
    return digests


def _write_extraction_manifest(root: Path, spec: DatasetSpec) -> None:
    """Record per-file digests of an extraction that came from the verified archive."""
    manifest = {"archive_sha256": spec.sha256, "files": _hash_tree(root)}
    (root / EXTRACTION_MANIFEST).write_text(json.dumps(manifest, indent=1), encoding="utf-8")


def _matches_extraction_manifest(root: Path, spec: DatasetSpec) -> bool:
    """Check that every file still matches the digests recorded at extraction time.

    A missing, stale or unreadable manifest means the extraction is re-created from the
    checksum-verified archive, so a truncated or edited image is never reused silently.
    """
    manifest_path = root / EXTRACTION_MANIFEST
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(manifest, dict) or manifest.get("archive_sha256") != spec.sha256:
        return False
    return bool(manifest.get("files") == _hash_tree(root))


def _install_extraction(archive: Path, destination: Path, spec: DatasetSpec) -> None:
    with TemporaryDirectory(prefix=".extract-", dir=destination.parent) as temp_dir:
        staged = Path(temp_dir) / "extracted"
        staged.mkdir()
        if archive.suffix == ".zip":
            with ZipFile(archive) as zip_file:
                zip_file.extractall(staged)
        else:
            _extract_rar(archive, staged)
        _write_extraction_manifest(staged, spec)
        if not _valid_extraction(staged, spec):
            raise DatasetError(f"{archive.name} does not contain the expected dataset files.")

        previous = Path(temp_dir) / "previous"
        if destination.exists() or destination.is_symlink():
            destination.rename(previous)
        try:
            staged.rename(destination)
        except OSError:
            if previous.exists() or previous.is_symlink():
                previous.rename(destination)
            raise


def _extract_rar(archive: Path, destination: Path) -> None:
    tool = shutil.which("bsdtar")
    if tool is None and sys.platform == "win32":
        system_root = os.environ.get("SYSTEMROOT")
        if system_root:
            system_tar = Path(system_root) / "System32" / "tar.exe"
            if system_tar.is_file():
                tool = str(system_tar)
    if tool is None and sys.platform == "darwin":
        tool = "/usr/bin/tar"
    if tool is None:
        raise DatasetError(
            "RAR extraction requires bsdtar (libarchive-tools on Ubuntu) or Windows system tar. "
            "See the README for manual extraction."
        )
    result = subprocess.run(
        [tool, "-xf", str(archive), "-C", str(destination)],
        check=False,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=600,
    )
    if result.returncode:
        raise DatasetError(f"Could not extract {archive.name}: {result.stderr.strip()}")
