from hashlib import sha256
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import httpx
import pytest
import yaml

from baulens.data.download import DatasetError, download_dataset, verify_checksum

PAYLOAD = b"concrete dataset fixture"


@pytest.fixture
def archive_and_manifest(tmp_path: Path) -> tuple[bytes, Path]:
    buffer = BytesIO()
    with ZipFile(buffer, mode="w") as archive:
        archive.writestr("sample.txt", PAYLOAD)
    archive_bytes = buffer.getvalue()
    manifest = tmp_path / "datasets.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "concrete": {
                    "url": "https://example.test/concrete.zip",
                    "filename": "concrete.zip",
                    "sha256": sha256(archive_bytes).hexdigest(),
                    "files": {"sample.txt": sha256(PAYLOAD).hexdigest()},
                }
            }
        ),
        encoding="utf-8",
    )
    return archive_bytes, manifest


def test_verify_checksum(tmp_path: Path) -> None:
    path = tmp_path / "sample.txt"
    path.write_bytes(b"abc")
    expected = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert verify_checksum(path, expected)
    path.write_bytes(b"abcd")
    assert not verify_checksum(path, expected)


def test_download_reuses_archive_and_repairs_extraction(
    tmp_path: Path, archive_and_manifest: tuple[bytes, Path]
) -> None:
    archive, manifest = archive_and_manifest
    requests = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        assert request.method == "GET"
        assert request.url == httpx.URL("https://example.test/concrete.zip")
        requests += 1
        return httpx.Response(200, content=archive)

    raw_dir = tmp_path / "raw"
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        directory = download_dataset(
            "concrete", raw_dir=raw_dir, manifest_path=manifest, client=client
        )
        extracted = directory / "sample.txt"
        original_stat = extracted.stat()
        assert extracted.read_bytes() == PAYLOAD
        assert download_dataset(
            "concrete", raw_dir=raw_dir, manifest_path=manifest, client=client
        ) == (directory)
        assert extracted.stat().st_mtime_ns == original_stat.st_mtime_ns
        extracted.write_bytes(b"damaged")
        download_dataset("concrete", raw_dir=raw_dir, manifest_path=manifest, client=client)

    assert requests == 1
    assert (raw_dir / "concrete.zip").read_bytes() == archive
    assert extracted.read_bytes() == PAYLOAD


def test_download_rejects_wrong_hash_before_publishing(
    tmp_path: Path, archive_and_manifest: tuple[bytes, Path]
) -> None:
    _, manifest = archive_and_manifest
    raw_dir = tmp_path / "raw"

    def respond(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"unexpected response")

    with (
        httpx.Client(transport=httpx.MockTransport(respond)) as client,
        pytest.raises(DatasetError, match="SHA-256 mismatch"),
    ):
        download_dataset("concrete", raw_dir=raw_dir, manifest_path=manifest, client=client)

    assert not (raw_dir / "concrete.zip").exists()
    assert not (raw_dir / "concrete").exists()
    assert not list(raw_dir.glob(".download-*"))


def test_download_rejects_wrong_hash_for_manual_archive(
    tmp_path: Path, archive_and_manifest: tuple[bytes, Path]
) -> None:
    _, manifest = archive_and_manifest
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "concrete.zip").write_bytes(b"corrupted archive")

    def respond(_request: httpx.Request) -> httpx.Response:
        pytest.fail("a locally supplied archive should be verified before any request")

    with (
        httpx.Client(transport=httpx.MockTransport(respond)) as client,
        pytest.raises(DatasetError, match="SHA-256 mismatch"),
    ):
        download_dataset("concrete", raw_dir=raw_dir, manifest_path=manifest, client=client)


def test_download_repairs_damaged_file_not_listed_in_manifest(
    tmp_path: Path, archive_and_manifest: tuple[bytes, Path]
) -> None:
    """Files without a pinned hash (e.g. 40k crack JPEGs) are checked via the extraction record."""
    _, manifest = archive_and_manifest
    buffer = BytesIO()
    with ZipFile(buffer, mode="w") as zip_file:
        zip_file.writestr("sample.txt", PAYLOAD)
        zip_file.writestr("images/00001.jpg", b"jpeg bytes")
    archive = buffer.getvalue()
    spec = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    spec["concrete"]["sha256"] = sha256(archive).hexdigest()
    manifest.write_text(yaml.safe_dump(spec), encoding="utf-8")

    transport = httpx.MockTransport(lambda _: httpx.Response(200, content=archive))
    raw_dir = tmp_path / "raw"
    with httpx.Client(transport=transport) as client:
        directory = download_dataset(
            "concrete", raw_dir=raw_dir, manifest_path=manifest, client=client
        )
        image = directory / "images" / "00001.jpg"
        image.write_bytes(b"")
        download_dataset("concrete", raw_dir=raw_dir, manifest_path=manifest, client=client)

    assert image.read_bytes() == b"jpeg bytes"
