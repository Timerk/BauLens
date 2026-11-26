"""Validate uploaded images before any model sees them.

Checks, in order: declared content type, byte size (streamed, so an oversized body is never
held in memory), Pillow's structural ``verify()``, the decoded format, the pixel count (a
guard against decompression bombs that is checked from the header, before decoding) and
finally a full decode. EXIF orientation is applied and the result converted to RGB, so the
model sees the image the way a person would.
"""

import hashlib
import io
import re
from dataclasses import dataclass
from pathlib import PurePath

from fastapi import UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError

from baulens.schemas import ImageMetadata

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}
ALLOWED_FORMATS = {"JPEG", "PNG"}
_CHUNK = 64 * 1024


class UploadRejectedError(ValueError):
    """The upload is not an acceptable image; ``status_code`` is the HTTP answer."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


@dataclass(frozen=True)
class ValidatedImage:
    image: Image.Image
    metadata: ImageMetadata


def safe_filename(name: str | None) -> str:
    """Base name only, without control characters; never a client-supplied path."""
    base = PurePath((name or "").replace("\\", "/")).name
    cleaned = re.sub(r"[\x00-\x1f\x7f]", "", base).strip()
    return cleaned[:255] or "upload"


async def read_limited(upload: UploadFile, max_bytes: int) -> bytes:
    buffer = bytearray()
    while chunk := await upload.read(_CHUNK):
        buffer += chunk
        if len(buffer) > max_bytes:
            raise UploadRejectedError(413, f"image exceeds the upload limit of {max_bytes} bytes")
    return bytes(buffer)


def decode_image(
    content: bytes, content_type: str | None, filename: str | None, max_pixels: int
) -> ValidatedImage:
    declared = (content_type or "").split(";")[0].strip().lower()
    if declared not in ALLOWED_CONTENT_TYPES:
        raise UploadRejectedError(415, "expected a JPEG or PNG image (image/jpeg or image/png)")
    if not content:
        raise UploadRejectedError(422, "the uploaded file is empty")
    try:
        with Image.open(io.BytesIO(content)) as probe:
            probe.verify()
        image = Image.open(io.BytesIO(content))
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as error:
        raise UploadRejectedError(422, "the file is not a readable image") from error
    except Image.DecompressionBombError as error:
        raise UploadRejectedError(413, "the image has too many pixels") from error
    if image.format not in ALLOWED_FORMATS:
        raise UploadRejectedError(415, f"unsupported image format {image.format}; use JPEG or PNG")
    if image.width * image.height > max_pixels:
        raise UploadRejectedError(413, f"the image has more than {max_pixels} pixels")
    try:
        image.load()
        rgb = ImageOps.exif_transpose(image).convert("RGB")
    except (OSError, SyntaxError, ValueError) as error:
        raise UploadRejectedError(422, "the image data is truncated or corrupt") from error
    metadata = ImageMetadata(
        filename=safe_filename(filename),
        width_px=rgb.width,
        height_px=rgb.height,
        sha256=hashlib.sha256(content).hexdigest(),
    )
    return ValidatedImage(rgb, metadata)
