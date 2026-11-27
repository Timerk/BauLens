"""Validate uploaded images before any model sees them.

Checks, in order:

1. The declared content type is ``image/jpeg`` or ``image/png`` (415 otherwise). The request
   body as a whole is capped earlier by
   :class:`~baulens.api.observability.BodySizeLimitMiddleware`.
2. Pillow opens the bytes with only the JPEG and PNG decoders enabled, and the detected
   format must match the declared type in both directions (415).
3. The pixel count is read from the header and checked **before** ``verify()`` and before
   any decoding, as a guard against decompression bombs (413).
4. ``verify()`` and a full decode catch truncated or corrupt data (422).

Supported image modes and how they reach the model as 8-bit RGB:

- ``1``, ``L``, ``RGB``, ``CMYK`` (JPEG), ``P`` without transparency: converted directly.
- ``I;16``/``I;16B``/``I;16L``/``I`` (16-bit greyscale PNG): scaled linearly from
  0-65535 to 0-255, so a 16-bit image keeps its contrast instead of saturating to white.
- ``LA``, ``RGBA``, ``PA`` and ``P`` with a transparency entry: composited on a white
  background, so fully transparent pixels are white and cannot influence the
  classification as hidden content.
- Anything else (e.g. ``F``, ``I;16`` with alpha, ``YCbCr`` PNGs) is rejected with 415.

EXIF orientation is applied last, so the model and every preview see the image upright.
"""

import base64
import hashlib
import io
import re
from dataclasses import dataclass
from pathlib import PurePath

import numpy as np
from fastapi import UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError

from baulens.schemas import ImageMetadata

FORMAT_FOR_TYPE = {"image/jpeg": "JPEG", "image/png": "PNG"}
ALLOWED_CONTENT_TYPES = set(FORMAT_FOR_TYPE)
_DIRECT_MODES = {"1", "L", "RGB", "CMYK", "P"}
_SIXTEEN_BIT_MODES = {"I;16", "I;16B", "I;16L", "I"}
_ALPHA_MODES = {"LA", "RGBA", "PA"}
_BACKGROUND = (255, 255, 255)
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


def to_rgb(image: Image.Image) -> Image.Image:
    """8-bit RGB following the mode contract in the module docstring."""
    mode = image.mode
    if mode == "P" and "transparency" in image.info:
        image, mode = image.convert("RGBA"), "RGBA"
    if mode in _SIXTEEN_BIT_MODES:
        samples = np.asarray(image, dtype=np.float64)
        scaled = np.clip(np.rint(samples / 257.0), 0, 255).astype(np.uint8)
        return Image.fromarray(scaled, mode="L").convert("RGB")
    if mode in _ALPHA_MODES:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (*_BACKGROUND, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")
    if mode in _DIRECT_MODES:
        return image.convert("RGB")
    raise UploadRejectedError(415, f"unsupported image mode {mode}")


def decode_image(
    content: bytes, content_type: str | None, filename: str | None, max_pixels: int
) -> ValidatedImage:
    declared = (content_type or "").split(";")[0].strip().lower()
    expected = FORMAT_FOR_TYPE.get(declared)
    if expected is None:
        raise UploadRejectedError(415, "expected a JPEG or PNG image (image/jpeg or image/png)")
    if not content:
        raise UploadRejectedError(422, "the uploaded file is empty")
    formats = list(FORMAT_FOR_TYPE.values())
    try:
        probe = Image.open(io.BytesIO(content), formats=formats)
    except UnidentifiedImageError as error:
        raise UploadRejectedError(422, "the file is not a readable JPEG or PNG image") from error
    except Image.DecompressionBombError as error:
        raise UploadRejectedError(413, "the image has too many pixels") from error
    except (OSError, SyntaxError, ValueError) as error:
        raise UploadRejectedError(422, "the file is not a readable image") from error
    with probe:
        if probe.format != expected:
            raise UploadRejectedError(415, f"declared {declared} but the content is {probe.format}")
        if probe.width * probe.height > max_pixels:
            raise UploadRejectedError(413, f"the image has more than {max_pixels} pixels")
        try:
            probe.verify()
        except (OSError, SyntaxError, ValueError) as error:
            raise UploadRejectedError(422, "the image data is truncated or corrupt") from error
    try:
        with Image.open(io.BytesIO(content), formats=[expected]) as image:
            image.load()
            upright = ImageOps.exif_transpose(image)
            rgb = to_rgb(upright)
    except UploadRejectedError:
        raise
    except (OSError, SyntaxError, ValueError, Image.DecompressionBombError) as error:
        raise UploadRejectedError(422, "the image data is truncated or corrupt") from error
    metadata = ImageMetadata(
        filename=safe_filename(filename),
        width_px=rgb.width,
        height_px=rgb.height,
        sha256=hashlib.sha256(content).hexdigest(),
    )
    return ValidatedImage(rgb, metadata)


def encode_preview(image: Image.Image, max_side: int, fmt: str = "JPEG") -> str:
    """Base64 of a bounded copy of the normalised image, as the model saw it (upright RGB)."""
    preview = image.copy()
    preview.thumbnail((max_side, max_side))
    buffer = io.BytesIO()
    if fmt == "JPEG":
        preview.save(buffer, format="JPEG", quality=88)
    else:
        preview.save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")
