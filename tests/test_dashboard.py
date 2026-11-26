"""Dashboard tests: the HTTP client against httpx.MockTransport and the app via AppTest."""

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from streamlit.testing.v1 import AppTest

from baulens_dashboard import client as client_module
from baulens_dashboard.client import ApiClient, ApiError
from baulens_dashboard.labels import DISCLAIMER, REPLAY_LABEL, number
from baulens_dashboard.report_view import highlighted

APP = Path(__file__).resolve().parents[1] / "apps" / "dashboard" / "app.py"
MIX = {
    "cement": 380.0,
    "blast_furnace_slag": 95.0,
    "fly_ash": 0.0,
    "water": 185.0,
    "superplasticizer": 4.0,
    "coarse_aggregate": 1000.0,
    "fine_aggregate": 760.0,
    "age_days": 28,
}


def _component(ready: bool, version: str | None = None, detail: str = "loaded") -> dict[str, Any]:
    return {"ready": ready, "detail": detail, "model_name": "m", "model_version": version}


def health(report_ready: bool = False, llm_mode: str = "live") -> dict[str, Any]:
    return {
        "status": "ok" if report_ready else "degraded",
        "version": "0.1.0",
        "llm_mode": llm_mode,
        "components": {
            "strength": _component(True, "s1"),
            "cracks": _component(True, "c1"),
            "report": _component(report_ready, None, "report module not available"),
        },
    }


MODELS = {
    "strength": {
        "model_name": "Histogram GBM",
        "model_version": "s1",
        "interval_coverage": 0.9,
        "interval_method": "split conformal",
        "interval_half_width_mpa": 10.9,
        "training_rows": 616,
        "training_ranges": [
            {"feature": name, "minimum": 0.0, "maximum": 2000.0, "unit": "kg/m³"} for name in MIX
        ],
        "test_metrics": {"mae": 3.83},
        "model_card": "docs/model_cards/strength.md",
    },
    "cracks": {
        "model_name": "resnet18",
        "model_version": "c1",
        "architecture": "resnet18",
        "image_size": 128,
        "threshold": 0.954,
        "threshold_strategy": "f1",
        "backend": "onnx",
        "test_metrics": {"f1": 0.999},
        "model_card": "docs/model_cards/cracks.md",
    },
}
PREDICTION = {
    "point_mpa": 51.19,
    "lower_mpa": 40.33,
    "upper_mpa": 62.04,
    "interval_coverage": 0.9,
    "interval_method": "split conformal",
    "in_training_domain": False,
    "top_features": [
        {"feature": "water_binder", "value": 0.39, "contribution_mpa": 5.3},
        {"feature": "water", "value": 185.0, "contribution_mpa": -1.1},
    ],
    "model_name": "Histogram GBM",
    "model_version": "s1",
}
REPORT = {
    "case_id": "PB-001",
    "source": "replay",
    "status": "accepted",
    "model": "llama3.2:3b",
    "generation_mode": "slots",
    "prompt_version": "report_slots_v1",
    "sections": [
        {"name": "druckfestigkeit", "title": "Druckfestigkeit", "text": "Prognose 51,2 MPa."}
    ],
    "markdown": "# Prüfberichtsentwurf PB-001\n",
    "disclaimer": "Hinweis: kein Prüfbericht.",
    "provenance": "Aufgezeichnete Modellausgabe (Replay)",
    "checks": [
        {
            "section": "druckfestigkeit",
            "start": 9,
            "end": 13,
            "text": "51,2",
            "status": "ok",
            "reason": "point estimate",
            "matched_fields": ["strength.point_mpa"],
        }
    ],
    "rejected_violations": [],
    "latency_s": 0.0,
    "fallback_used": False,
    "note": None,
}

Handler = Callable[[httpx.Request], httpx.Response]


def api(report_ready: bool = False, llm_mode: str = "live") -> tuple[Handler, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        routes: dict[tuple[str, str], tuple[int, object]] = {
            ("GET", "/api/v1/health"): (200, health(report_ready, llm_mode)),
            ("GET", "/api/v1/models"): (200, MODELS),
            ("POST", "/api/v1/strength/predict"): (200, PREDICTION),
            ("GET", "/api/v1/report/cases"): (
                (200, [{"case": {"case_id": "PB-001"}, "models": ["llama3.2:3b"]}])
                if report_ready
                else (503, {"detail": "report module not available"})
            ),
            ("POST", "/api/v1/report"): (200, REPORT),
        }
        status, body = routes.get((request.method, request.url.path), (404, {"detail": "nope"}))
        return httpx.Response(status, json=body)

    return handle, seen


@pytest.fixture
def mocked_api() -> Iterator[Callable[..., list[httpx.Request]]]:
    def install(**options: Any) -> list[httpx.Request]:
        handler, seen = api(**options)
        client_module.transport_override = httpx.MockTransport(handler)
        return seen

    yield install
    client_module.transport_override = None


def test_client_parses_responses_and_sends_request_ids() -> None:
    handler, seen = api()
    client = ApiClient("http://api:8000/", httpx.MockTransport(handler))
    assert client.health()["components"]["strength"]["ready"]
    assert client.predict_strength(MIX)["point_mpa"] == 51.19
    assert seen[0].url == "http://api:8000/api/v1/health"
    assert seen[1].headers["X-Request-ID"].startswith("dashboard-")
    assert json.loads(seen[1].content) == MIX


def test_client_uploads_multipart_with_gradcam_flag() -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"prediction": {}})

    ApiClient("http://api", httpx.MockTransport(handler)).predict_cracks(
        b"\xff\xd8", "probe.jpg", "image/jpeg"
    )
    request = captured[0]
    assert request.url.params["gradcam"] == "true"
    assert request.headers["content-type"].startswith("multipart/form-data")
    assert b'filename="probe.jpg"' in request.content


@pytest.mark.parametrize(
    ("response", "status", "detail"),
    [
        (httpx.Response(503, json={"detail": "no strength model"}), 503, "no strength model"),
        (
            httpx.Response(422, json={"detail": [{"loc": ["body", "cement"], "msg": "too low"}]}),
            422,
            "cement: too low",
        ),
        (httpx.Response(502, text="Bad Gateway"), 502, "Bad Gateway"),
    ],
)
def test_client_turns_errors_into_api_errors(
    response: httpx.Response, status: int, detail: str
) -> None:
    client = ApiClient("http://api", httpx.MockTransport(lambda request: response))
    with pytest.raises(ApiError) as raised:
        client.models()
    assert raised.value.status == status
    assert raised.value.detail == detail


def test_unreachable_api_is_reported_as_unavailable() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(ApiError) as raised:
        ApiClient("http://api", httpx.MockTransport(refuse)).health()
    assert raised.value.unavailable
    assert "nicht erreichbar" in raised.value.detail


def test_highlighting_escapes_text_and_marks_checked_numbers() -> None:
    checks = [
        {"section": "s", "start": 4, "end": 8, "text": "51,2", "status": "ok", "reason": "ok"},
        {"section": "s", "start": 13, "end": 15, "text": "99", "status": "invented", "reason": "x"},
    ]
    html = highlighted("<b> 51,2 und 99 MPa", "s", checks)  # type: ignore[arg-type]
    assert html.startswith("&lt;b&gt; ")
    assert '<mark class="bl-ok" title="belegt: ok">51,2</mark>' in html
    assert '<mark class="bl-bad" title="erfunden: x">99</mark>' in html


def test_german_number_format() -> None:
    assert number(1234.56) == "1.234,6"
    assert number(0.954, 3) == "0,954"


def _app() -> AppTest:
    return AppTest.from_file(str(APP), default_timeout=30)


def test_app_renders_and_predicts_strength(mocked_api: Callable[..., list[httpx.Request]]) -> None:
    seen = mocked_api()
    app = _app().run()
    assert not app.exception
    assert [tab.label for tab in app.tabs] == ["Betonrezeptur", "Rissbild", "Prüfbericht"]
    assert any(DISCLAIMER in element.value for element in app.get("html") if element.value)
    app.button[0].click().run()
    assert not app.exception
    assert app.metric[0].value == "51,2 MPa"
    assert "Außerhalb des Trainingsbereichs" in app.warning[0].value
    assert any(request.url.path == "/api/v1/strength/predict" for request in seen)
    # The report tab explains that the module is not wired yet instead of failing.
    assert any("Berichtsmodul" in info.value for info in app.info)


def test_app_labels_replays_unmistakably(mocked_api: Callable[..., list[httpx.Request]]) -> None:
    mocked_api(report_ready=True, llm_mode="replay")
    app = _app().run()
    report_source = app.radio[-1]
    assert report_source.options == ["Vorlage (ohne Sprachmodell)", "Replay (aufgezeichnet)"]
    report_source.set_value("replay").run()
    assert REPLAY_LABEL in [warning.value for warning in app.warning]
    next(button for button in app.button if button.label == "Bericht erstellen").click().run()
    assert not app.exception
    assert [warning.value for warning in app.warning].count(REPLAY_LABEL) == 2


def test_app_explains_an_unreachable_api(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BAULENS_API_URL", "http://127.0.0.1:9")
    app = _app().run()
    assert not app.exception
    assert any("nicht erreichbar" in error.value for error in app.error)
    assert not app.tabs
