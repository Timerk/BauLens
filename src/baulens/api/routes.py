"""Versioned HTTP endpoints. Inference runs in the threadpool, never on the event loop."""

from dataclasses import dataclass
from importlib.metadata import version
from typing import Annotated, Any

import numpy as np
from fastapi import APIRouter, Body, File, HTTPException, Query, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from PIL import Image

from baulens.api.models import (
    ComponentName,
    ComponentStatus,
    CrackAnalysis,
    ErrorResponse,
    HealthResponse,
    ModelsResponse,
    RecordedCase,
    ReportRequest,
    ReportResponse,
)
from baulens.api.report_service import ReportUnavailableError
from baulens.api.services import CrackService, Unavailable
from baulens.api.state import Cracks, State, Strength
from baulens.api.uploads import (
    UploadRejectedError,
    decode_image,
    encode_preview,
    read_limited,
)
from baulens.cracks.inference import overlay_heatmap
from baulens.schemas import ConcreteMix, CrackPrediction, StrengthPrediction

router = APIRouter()

_ERRORS: dict[int | str, dict[str, Any]] = {
    503: {"model": ErrorResponse, "description": "Model or module not available"}
}
_JSON_ERRORS: dict[int | str, dict[str, Any]] = {
    422: {"model": ErrorResponse, "description": "Invalid request body"},
    **_ERRORS,
}
_UPLOAD_ERRORS: dict[int | str, dict[str, Any]] = {
    413: {"model": ErrorResponse, "description": "File or image dimensions too large"},
    415: {"model": ErrorResponse, "description": "Not a JPEG or PNG image"},
    422: {"model": ErrorResponse, "description": "Unreadable or corrupt image"},
    **_ERRORS,
}

EXAMPLE_MIX = {
    "cement": 380.0,
    "blast_furnace_slag": 95.0,
    "fly_ash": 0.0,
    "water": 185.0,
    "superplasticizer": 4.0,
    "coarse_aggregate": 1000.0,
    "fine_aggregate": 760.0,
    "age_days": 28,
}


@router.get("/health", tags=["system"])
def health(state: State) -> HealthResponse:
    """Readiness of each component with the loaded model versions. Always 200."""
    components: dict[ComponentName, ComponentStatus] = {
        "strength": state.strength.status(),
        "cracks": state.cracks.status(),
        "report": state.report.status(),
    }
    ready = all(component.ready for component in components.values())
    return HealthResponse(
        status="ok" if ready else "degraded",
        version=version("baulens"),
        llm_mode=state.settings.llm_mode,
        components=components,
    )


@router.get("/models", tags=["system"])
def models(state: State) -> ModelsResponse:
    """Model summaries: versions, test metrics and the training-domain ranges for forms."""
    strength = None if isinstance(state.strength, Unavailable) else state.strength.info
    cracks = None if isinstance(state.cracks, Unavailable) else state.cracks.info
    return ModelsResponse(strength=strength, cracks=cracks)


@router.post("/strength/predict", tags=["strength"], responses=_JSON_ERRORS)
async def predict_strength(
    service: Strength,
    mix: Annotated[
        ConcreteMix,
        Body(openapi_examples={"typical": {"summary": "C30/37-like mix", "value": EXAMPLE_MIX}}),
    ],
) -> StrengthPrediction:
    """Compressive strength with a 90 % conformal interval and the top SHAP contributions.

    A mix that passes the schema bounds but makes an engineered ratio non-finite (e.g. a
    vanishingly small cement content) is rejected with 422.
    """
    try:
        return await run_in_threadpool(service.predict, mix)
    except ValueError as error:
        # ConcreteMix is already validated, so a ValueError here is about the input values.
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(error)) from error


@dataclass(frozen=True)
class _Classified:
    prediction: CrackPrediction
    preview: str | None
    overlay: str | None


def _classify(
    service: CrackService, image: Image.Image, gradcam: bool, max_side: int
) -> _Classified:
    prediction = service.classifier.predict(image)
    if not gradcam:
        return _Classified(prediction, None, None)
    # Preview and overlay come from the same upright image, so both panels match.
    preview = image.copy()
    preview.thumbnail((max_side, max_side))
    overlay = overlay_heatmap(preview, np.asarray(service.classifier.gradcam(image)))
    return _Classified(
        prediction,
        encode_preview(preview, max_side, "JPEG"),
        encode_preview(overlay, max_side, "PNG"),
    )


@router.post("/cracks/predict", tags=["cracks"], responses=_UPLOAD_ERRORS)
async def predict_cracks(
    state: State,
    service: Cracks,
    file: Annotated[UploadFile, File(description="JPEG or PNG surface image")],
    gradcam: Annotated[
        bool, Query(description="Also return the upright preview and a Grad-CAM overlay")
    ] = False,
) -> CrackAnalysis:
    """Crack probability against the validation threshold, plus image metadata."""
    config = state.config
    try:
        content = await read_limited(file, config.max_upload_bytes)
        validated = await run_in_threadpool(
            decode_image, content, file.content_type, file.filename, config.max_image_pixels
        )
    except UploadRejectedError as error:
        raise HTTPException(error.status_code, error.detail) from error
    result = await run_in_threadpool(
        _classify, service, validated.image, gradcam, config.gradcam_max_side
    )
    return CrackAnalysis(
        prediction=result.prediction,
        image=validated.metadata,
        input_size_px=service.info.image_size,
        preview_jpeg_base64=result.preview,
        gradcam_png_base64=result.overlay,
    )


@router.post(
    "/report",
    tags=["report"],
    responses={403: {"model": ErrorResponse, "description": "Live mode disabled"}, **_JSON_ERRORS},
)
async def report(state: State, request: ReportRequest) -> ReportResponse:
    """German inspection report draft whose numbers passed the faithfulness check.

    A model draft that cites a number not in the input is never delivered; the deterministic
    template is returned instead and ``status``/``note`` say why.
    """
    if not state.report.status().ready:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, state.report.status().detail)
    if request.source == "live" and state.settings.llm_mode == "replay":
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "live generation is disabled in this deployment (BAULENS_LLM_MODE=replay)",
        )
    try:
        return await run_in_threadpool(state.report.generate, request)
    except ReportUnavailableError as error:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(error)) from error


@router.get("/report/cases", tags=["report"], responses=_ERRORS)
def report_cases(state: State) -> list[RecordedCase]:
    """Cases with recorded model output, selectable in replay mode."""
    try:
        return state.report.recorded_cases()
    except ReportUnavailableError as error:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(error)) from error
