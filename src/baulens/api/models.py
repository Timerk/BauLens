"""Request and response bodies of the HTTP API.

Analysis payloads reuse :mod:`baulens.schemas` unchanged. The models here only add what the
transport needs: health and model summaries, upload metadata and the report envelope. The
report envelope is owned by the API rather than by ``baulens.report`` so that the dashboard
depends on one stable contract while the report module evolves behind
:mod:`baulens.api.report_service`.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from baulens.schemas import CrackPrediction, ImageMetadata, InspectionInput

API_PREFIX = "/api/v1"

ComponentName = Literal["strength", "cracks", "report"]
ReportSource = Literal["live", "replay", "template"]
GenerationMode = Literal["slots", "free"]
ReportStatus = Literal["accepted", "rejected", "invalid", "error", "replay_miss", "template"]


class _Response(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ErrorResponse(_Response):
    """Body of every 4xx/5xx response produced by the API itself."""

    detail: str | list[dict[str, Any]] = Field(
        description="Message, or the list of validation errors for a 422"
    )
    request_id: str | None = None


class ComponentStatus(_Response):
    ready: bool
    detail: str = Field(description="Why a component is unavailable, or what is loaded")
    model_name: str | None = None
    model_version: str | None = None


class HealthResponse(_Response):
    status: Literal["ok", "degraded"]
    version: str
    llm_mode: Literal["live", "replay"]
    components: dict[ComponentName, ComponentStatus]


class FeatureRange(_Response):
    feature: str
    minimum: float
    maximum: float
    unit: str


class StrengthModelInfo(_Response):
    model_name: str
    model_version: str
    interval_coverage: float
    interval_method: str
    interval_half_width_mpa: float
    training_rows: int | None = None
    training_ranges: list[FeatureRange]
    test_metrics: dict[str, float] = Field(
        default_factory=dict,
        description="MAE, RMSE and R² on unseen ingredient tuples, if evaluated for this version",
    )
    model_card: str = "docs/model_cards/strength.md"


class CrackModelInfo(_Response):
    model_name: str
    model_version: str
    architecture: str
    image_size: int
    threshold: float
    threshold_strategy: str | None
    backend: str
    test_metrics: dict[str, float] = Field(default_factory=dict)
    model_card: str = "docs/model_cards/cracks.md"


class ModelsResponse(_Response):
    strength: StrengthModelInfo | None
    cracks: CrackModelInfo | None


class CrackAnalysis(_Response):
    prediction: CrackPrediction
    image: ImageMetadata
    input_size_px: int = Field(description="The image is resized to this square for the model")
    gradcam_png_base64: str | None = Field(
        default=None, description="Grad-CAM overlay as PNG, only if requested with gradcam=true"
    )


class ReportRequest(BaseModel):
    """A report for one inspection case.

    ``source`` selects the generator: ``live`` asks the local Ollama model, ``replay`` serves
    recorded model output for one of the recorded cases, ``template`` renders the
    deterministic text without a model.
    """

    model_config = ConfigDict(extra="forbid")

    case: InspectionInput
    source: ReportSource = "template"
    generation_mode: GenerationMode = "slots"
    model: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z0-9._:/-]{1,64}$",
        description="Ollama model tag; defaults to the first model in the report config",
    )


class NumberVerdict(_Response):
    """Faithfulness verdict for one number in the delivered text."""

    section: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    text: str
    status: Literal["ok", "invented", "misattributed", "inexact", "unbound"]
    reason: str
    matched_fields: list[str] = Field(default_factory=list)


class ReportSection(_Response):
    name: str
    title: str
    text: str


class ReportResponse(_Response):
    case_id: str
    source: ReportSource
    status: ReportStatus
    model: str | None
    generation_mode: GenerationMode | None
    prompt_version: str | None
    sections: list[ReportSection]
    markdown: str
    disclaimer: str
    provenance: str
    checks: list[NumberVerdict] = Field(description="Checks of the delivered text")
    rejected_violations: list[NumberVerdict] = Field(
        default_factory=list, description="Violations of a model draft that was not delivered"
    )
    latency_s: float = Field(ge=0)
    fallback_used: bool
    note: str | None = None


class RecordedCase(_Response):
    case: InspectionInput
    models: list[str] = Field(description="Models with recorded output for this case")
