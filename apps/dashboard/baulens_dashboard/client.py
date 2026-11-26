"""HTTP client for the BauLens API. The dashboard never imports models; it only calls this.

Responses are returned as plain JSON mappings described by the TypedDicts below, which
mirror the API's OpenAPI schema (``/api/v1/openapi.json``) for the fields the UI reads.
"""

import os
import uuid
from typing import Any, Literal, NotRequired, TypedDict, cast

import httpx

DEFAULT_API_URL = "http://localhost:8000"
REPORT_TIMEOUT_S = 300.0  # live generation takes 26-71 s per report on a laptop CPU

transport_override: httpx.BaseTransport | None = None
"""Tests replace the network with ``httpx.MockTransport`` here."""


class ComponentStatus(TypedDict):
    ready: bool
    detail: str
    model_name: str | None
    model_version: str | None


class Health(TypedDict):
    status: Literal["ok", "degraded"]
    version: str
    llm_mode: Literal["live", "replay"]
    components: dict[str, ComponentStatus]


class FeatureRange(TypedDict):
    feature: str
    minimum: float
    maximum: float
    unit: str


class StrengthModel(TypedDict):
    model_name: str
    model_version: str
    interval_coverage: float
    interval_half_width_mpa: float
    training_rows: int | None
    training_ranges: list[FeatureRange]
    test_metrics: dict[str, float]
    model_card: str


class CrackModel(TypedDict):
    model_name: str
    model_version: str
    image_size: int
    threshold: float
    threshold_strategy: str | None
    backend: str
    test_metrics: dict[str, float]
    model_card: str


class Models(TypedDict):
    strength: StrengthModel | None
    cracks: CrackModel | None


class Contribution(TypedDict):
    feature: str
    value: float
    contribution_mpa: float


class StrengthPrediction(TypedDict):
    point_mpa: float
    lower_mpa: float
    upper_mpa: float
    interval_coverage: float
    interval_method: str
    in_training_domain: bool
    top_features: list[Contribution]
    model_name: str
    model_version: str


class CrackPrediction(TypedDict):
    crack_probability: float
    threshold: float
    label: Literal["crack", "no_crack"]
    model_name: str
    model_version: str


class ImageMetadata(TypedDict):
    filename: str
    width_px: int
    height_px: int
    sha256: str | None
    element: NotRequired[str | None]


class CrackAnalysis(TypedDict):
    prediction: CrackPrediction
    image: ImageMetadata
    input_size_px: int
    gradcam_png_base64: str | None


class NumberVerdict(TypedDict):
    section: str
    start: int
    end: int
    text: str
    status: Literal["ok", "invented", "misattributed", "inexact", "unbound"]
    reason: str
    matched_fields: list[str]


class ReportSection(TypedDict):
    name: str
    title: str
    text: str


class Report(TypedDict):
    case_id: str
    source: Literal["live", "replay", "template"]
    status: str
    model: str | None
    generation_mode: str | None
    prompt_version: str | None
    sections: list[ReportSection]
    markdown: str
    disclaimer: str
    provenance: str
    checks: list[NumberVerdict]
    rejected_violations: list[NumberVerdict]
    latency_s: float
    fallback_used: bool
    note: str | None


class RecordedCase(TypedDict):
    case: dict[str, Any]
    models: list[str]


class ApiError(RuntimeError):
    """The API answered with an error, or could not be reached (``status`` is then None)."""

    def __init__(self, status: int | None, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail

    @property
    def unavailable(self) -> bool:
        return self.status is None or self.status == 503


def _detail(response: httpx.Response) -> str:
    try:
        detail = response.json().get("detail")
    except (ValueError, AttributeError):
        return response.text[:300] or response.reason_phrase
    if isinstance(detail, list):
        return "; ".join(
            f"{'.'.join(str(part) for part in item.get('loc', [])[1:])}: {item.get('msg')}"
            for item in detail
            if isinstance(item, dict)
        )
    return str(detail)


class ApiClient:
    def __init__(
        self,
        base_url: str,
        transport: httpx.BaseTransport | None = None,
        public_url: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.public_url = (public_url or base_url).rstrip("/")
        """Where a browser reaches the API (differs from ``base_url`` inside Compose)."""
        self._http = httpx.Client(
            base_url=f"{self.base_url}/api/v1", timeout=30.0, transport=transport
        )

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        headers = {"X-Request-ID": f"dashboard-{uuid.uuid4().hex[:16]}"}
        try:
            response = self._http.request(method, path, headers=headers, **kwargs)
        except httpx.TransportError as error:
            raise ApiError(None, f"API nicht erreichbar unter {self.base_url} ({error})") from error
        if response.is_error:
            raise ApiError(response.status_code, _detail(response))
        return response.json()

    def health(self) -> Health:
        return cast(Health, self._request("GET", "/health", timeout=5.0))

    def models(self) -> Models:
        return cast(Models, self._request("GET", "/models"))

    def predict_strength(self, mix: dict[str, float]) -> StrengthPrediction:
        return cast(StrengthPrediction, self._request("POST", "/strength/predict", json=mix))

    def predict_cracks(
        self, content: bytes, filename: str, content_type: str, *, gradcam: bool = True
    ) -> CrackAnalysis:
        return cast(
            CrackAnalysis,
            self._request(
                "POST",
                "/cracks/predict",
                params={"gradcam": str(gradcam).lower()},
                files={"file": (filename, content, content_type)},
            ),
        )

    def report(self, case: dict[str, Any], source: str, model: str | None = None) -> Report:
        body: dict[str, Any] = {"case": case, "source": source}
        if model is not None:
            body["model"] = model
        return cast(Report, self._request("POST", "/report", json=body, timeout=REPORT_TIMEOUT_S))

    def recorded_cases(self) -> list[RecordedCase]:
        return cast(list[RecordedCase], self._request("GET", "/report/cases"))


def default_client() -> ApiClient:
    """Client for ``BAULENS_API_URL`` (default ``http://localhost:8000``).

    ``BAULENS_API_PUBLIC_URL`` is only used for links shown to the browser.
    """
    return ApiClient(
        os.environ.get("BAULENS_API_URL", DEFAULT_API_URL),
        transport_override,
        os.environ.get("BAULENS_API_PUBLIC_URL"),
    )
