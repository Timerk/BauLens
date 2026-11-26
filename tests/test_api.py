"""HTTP API tests against tiny locally built models (no network, GPU, Ollama or real data)."""

import base64
import hashlib
import io
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from api_helpers import build_crack_model, build_strength_model, image_bytes
from fastapi.testclient import TestClient
from PIL import Image
from typer.testing import CliRunner

from baulens.api import create_app
from baulens.api.models import (
    ComponentStatus,
    NumberVerdict,
    RecordedCase,
    ReportRequest,
    ReportResponse,
    ReportSection,
)
from baulens.api.uploads import UploadRejectedError, decode_image, to_rgb
from baulens.cli import app as cli_app
from baulens.config import ApiConfig, Settings, load_config
from baulens.schemas import CrackPrediction, ImageMetadata, StrengthPrediction

V1 = "/api/v1"
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
CRACK = {
    "crack_probability": 0.97,
    "threshold": 0.5,
    "label": "crack",
    "model_name": "resnet18",
    "model_version": "test",
}


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("api-artifacts")
    build_strength_model(root)
    build_crack_model(root)
    return root


def api_config(artifacts_dir: Path, **overrides: object) -> ApiConfig:
    config = load_config(Path("configs/smoke/api.yaml"), ApiConfig)
    return config.model_copy(update={"artifacts_dir": artifacts_dir, **overrides})


def settings(llm_mode: str = "live") -> Settings:
    return Settings(_env_file=None, BAULENS_LLM_MODE=llm_mode)  # type: ignore[call-arg]


@dataclass
class FakeReportService:
    """Stands in for the report adapter so the route logic is tested without the module."""

    requests: list[ReportRequest]

    def status(self) -> ComponentStatus:
        return ComponentStatus(ready=True, detail="fake")

    def recorded_cases(self) -> list[RecordedCase]:
        return []

    def generate(self, request: ReportRequest) -> ReportResponse:
        self.requests.append(request)
        text = "Die Wahrscheinlichkeit beträgt 0,97."
        return ReportResponse(
            case_id=request.case.case_id,
            source=request.source,
            status="template",
            model=None,
            generation_mode=None,
            prompt_version=None,
            sections=[ReportSection(name="rissbefund", title="Rissbefund", text=text)],
            markdown=f"## Rissbefund\n\n{text}\n",
            disclaimer="Hinweis",
            provenance="Vorlage",
            checks=[
                NumberVerdict(
                    section="rissbefund",
                    start=31,
                    end=35,
                    text="0,97",
                    status="ok",
                    reason="matches",
                    matched_fields=["crack.crack_probability"],
                )
            ],
            latency_s=0.0,
            fallback_used=False,
        )


@pytest.fixture(scope="module")
def client(artifacts: Path) -> Iterator[TestClient]:
    with TestClient(create_app(api_config(artifacts), settings())) as test_client:
        yield test_client


def upload(
    client: TestClient,
    content: bytes,
    content_type: str = "image/jpeg",
    name: str = "probe.jpg",
    query: str = "",
) -> object:
    return client.post(f"{V1}/cracks/predict{query}", files={"file": (name, content, content_type)})


def test_health_reports_each_component(client: TestClient) -> None:
    body = client.get(f"{V1}/health").json()
    assert body["status"] == "degraded"  # report module not wired yet
    assert body["components"]["strength"]["ready"]
    assert body["components"]["cracks"]["model_version"] == "resnet18-test"
    assert body["components"]["report"] == {
        "ready": False,
        "detail": "report module not available (Milestone 4 is not merged yet)",
        "model_name": None,
        "model_version": None,
    }


def test_models_lists_training_ranges_and_metrics(client: TestClient) -> None:
    body = client.get(f"{V1}/models").json()
    ranges = {item["feature"]: item for item in body["strength"]["training_ranges"]}
    assert set(ranges) == set(MIX)
    assert ranges["age_days"]["unit"] == "d"
    assert ranges["cement"]["minimum"] < ranges["cement"]["maximum"]
    assert body["cracks"]["threshold"] == 0.5
    assert body["cracks"]["test_metrics"] == {"f1": 0.5, "roc_auc": 0.5}


def test_strength_prediction_round_trips_the_shared_schema(client: TestClient) -> None:
    response = client.post(f"{V1}/strength/predict", json=MIX)
    assert response.status_code == 200
    prediction = StrengthPrediction.model_validate(response.json())
    assert prediction.lower_mpa <= prediction.point_mpa <= prediction.upper_mpa
    assert prediction.interval_coverage == 0.9


@pytest.mark.parametrize(
    "body",
    [MIX | {"cement": -1}, MIX | {"unknown": 1}, {"cement": 300}, MIX | {"age_days": "x"}],
)
def test_strength_rejects_invalid_mixes(client: TestClient, body: dict[str, object]) -> None:
    response = client.post(f"{V1}/strength/predict", json=body)
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert isinstance(detail, list)
    assert all("input" not in item for item in detail)  # inputs are not echoed back


def test_crack_prediction_returns_metadata_and_request_id(client: TestClient) -> None:
    content = image_bytes()
    response = upload(client, content, name="../../secret/dir/Stütze B.jpg")
    assert response.status_code == 200
    body = response.json()
    CrackPrediction.model_validate(body["prediction"])
    image = ImageMetadata.model_validate(body["image"])
    assert image.filename == "Stütze B.jpg"
    assert (image.width_px, image.height_px) == (80, 60)
    assert image.sha256 == hashlib.sha256(content).hexdigest()
    assert body["input_size_px"] == 64
    assert body["gradcam_png_base64"] is None
    assert len(response.headers["X-Request-ID"]) == 32


def _decoded(payload: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(payload)))


def test_gradcam_overlay_and_preview_are_bounded_and_aligned(client: TestClient) -> None:
    response = upload(client, image_bytes("PNG", (600, 300)), "image/png", "a.png", "?gradcam=true")
    assert response.status_code == 200
    overlay = _decoded(response.json()["gradcam_png_base64"])
    preview = _decoded(response.json()["preview_jpeg_base64"])
    assert (overlay.format, preview.format) == ("PNG", "JPEG")
    assert overlay.size == preview.size == (256, 128)  # gradcam_max_side in the smoke config


def test_without_gradcam_no_images_are_returned(client: TestClient) -> None:
    body = upload(client, image_bytes()).json()
    assert body["gradcam_png_base64"] is None
    assert body["preview_jpeg_base64"] is None


def test_exif_orientation_is_applied_to_model_input_and_previews(client: TestClient) -> None:
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90° clockwise when displayed
    content = image_bytes(size=(80, 60), exif=exif.tobytes())
    body = upload(client, content, query="?gradcam=true").json()
    assert (body["image"]["width_px"], body["image"]["height_px"]) == (60, 80)
    assert _decoded(body["preview_jpeg_base64"]).size == (60, 80)
    assert _decoded(body["gradcam_png_base64"]).size == (60, 80)


def _png(image: Image.Image, **save: object) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", **save)
    return buffer.getvalue()


def _pixel(content: bytes) -> object:
    return decode_image(content, "image/png", "probe.png", 10_000).image.getpixel((0, 0))


def test_sixteen_bit_greyscale_is_scaled_not_saturated() -> None:
    dark = Image.fromarray(np.full((8, 8), 256, dtype=np.uint16))  # 256 / 65535: near black
    assert dark.mode.startswith("I;16")
    assert _pixel(_png(dark)) == (1, 1, 1)
    bright = Image.fromarray(np.full((8, 8), 65535, dtype=np.uint16))
    assert _pixel(_png(bright)) == (255, 255, 255)


@pytest.mark.parametrize(("mode", "colour"), [("RGBA", (0, 0, 0, 0)), ("LA", (0, 0))])
def test_transparent_pixels_are_composited_on_white(mode: str, colour: tuple[int, ...]) -> None:
    assert _pixel(_png(Image.new(mode, (8, 8), colour))) == (255, 255, 255)


def test_palette_transparency_is_composited_on_white() -> None:
    image = Image.new("P", (8, 8), 0)
    image.putpalette([0, 0, 0] * 256)
    assert _pixel(_png(image, transparency=0)) == (255, 255, 255)


def test_unsupported_modes_are_rejected() -> None:
    with pytest.raises(UploadRejectedError) as raised:
        to_rgb(Image.new("F", (4, 4)))
    assert raised.value.status_code == 415


def _gif() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buffer, format="GIF")
    return buffer.getvalue()


@pytest.mark.parametrize(
    ("content", "content_type", "status"),
    [
        (b"plain text", "text/plain", 415),
        (image_bytes(), "application/octet-stream", 415),
        (_gif(), "image/gif", 415),
        (_gif(), "image/png", 422),  # only the JPEG and PNG decoders are enabled
        (image_bytes("JPEG"), "image/png", 415),  # declared PNG, content JPEG
        (image_bytes("PNG"), "image/jpeg", 415),  # declared JPEG, content PNG
        (b"not an image at all", "image/jpeg", 422),
        (b"", "image/jpeg", 422),
        (image_bytes()[:200], "image/jpeg", 422),  # truncated
        (image_bytes("PNG", (2100, 2000)), "image/png", 413),  # 4.2 MP > 4 MP limit
    ],
)
def test_upload_validation(
    client: TestClient, content: bytes, content_type: str, status: int
) -> None:
    response = upload(client, content, content_type)
    assert response.status_code == status, response.json()
    assert response.json()["request_id"] == response.headers["X-Request-ID"]


def test_oversized_upload_is_refused_before_parsing(client: TestClient) -> None:
    response = upload(client, b"\xff" * (1024 * 1024 + 1))
    assert response.status_code == 413
    body_limit = 1024 * 1024 + 64 * 1024
    response = client.post(
        f"{V1}/cracks/predict",
        files={"file": ("big.jpg", b"\xff" * (body_limit + 1), "image/jpeg")},
        headers={"Origin": "http://localhost:8501", "X-Request-ID": "upload-12345678"},
    )
    assert response.status_code == 413
    assert "request body exceeds" in response.json()["detail"]
    assert response.json()["request_id"] == "upload-12345678"
    assert response.headers["X-Request-ID"] == "upload-12345678"
    assert response.headers["access-control-allow-origin"] == "http://localhost:8501"


def test_pixel_limit_is_checked_before_verification(client: TestClient) -> None:
    # A 4.2 MP header followed by corrupt data: the pixel budget answers before verify().
    content = bytearray(image_bytes("PNG", (2100, 2000)))
    content[-40:-12] = bytes(28)
    assert upload(client, bytes(content), "image/png").status_code == 413


def test_mix_with_non_finite_ratios_is_a_client_error(client: TestClient) -> None:
    response = client.post(
        f"{V1}/strength/predict",
        json=MIX | {"cement": 5e-324},
        headers={"X-Request-ID": "overflow-1234"},
    )
    assert response.status_code == 422
    assert "non-finite" in response.json()["detail"]
    assert response.json()["request_id"] == "overflow-1234"


class _ExplodingReportService(FakeReportService):
    def generate(self, request: ReportRequest) -> ReportResponse:
        raise RuntimeError("secret internal state at C:/models")


def test_unexpected_errors_keep_request_id_and_cors(artifacts: Path) -> None:
    app = create_app(api_config(artifacts), settings(), report_service=_ExplodingReportService([]))
    with TestClient(app) as wired:
        response = wired.post(
            f"{V1}/report",
            json={"case": {"case_id": "PB-1", "crack": CRACK}},
            headers={"Origin": "http://localhost:8501", "X-Request-ID": "boom-12345678"},
        )
    assert response.status_code == 500
    assert response.json() == {"detail": "internal error", "request_id": "boom-12345678"}
    assert response.headers["X-Request-ID"] == "boom-12345678"
    assert response.headers["access-control-allow-origin"] == "http://localhost:8501"


def test_missing_file_field_is_a_validation_error(client: TestClient) -> None:
    assert client.post(f"{V1}/cracks/predict").status_code == 422


def test_missing_artifacts_make_endpoints_unavailable(tmp_path: Path) -> None:
    with TestClient(create_app(api_config(tmp_path), settings())) as empty:
        health = empty.get(f"{V1}/health").json()
        assert health["status"] == "degraded"
        assert "make strength" in health["components"]["strength"]["detail"]
        assert "make cracks" in health["components"]["cracks"]["detail"]
        assert empty.get(f"{V1}/models").json() == {"strength": None, "cracks": None}
        response = empty.post(f"{V1}/strength/predict", json=MIX)
        assert response.status_code == 503
        assert "no strength model" in response.json()["detail"]
        assert upload(empty, image_bytes()).status_code == 503
        # Public details never reveal where the service looks for its models.
        assert tmp_path.name not in json.dumps(health)


def test_tampered_artifacts_are_reported_not_loaded(artifacts: Path, tmp_path: Path) -> None:
    import shutil

    shutil.copytree(artifacts, tmp_path, dirs_exist_ok=True)
    for metadata in (tmp_path / "cracks" / "model" / "generations").glob("*/metadata.json"):
        metadata.write_text(metadata.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with TestClient(create_app(api_config(tmp_path), settings())) as tampered:
        cracks = tampered.get(f"{V1}/health").json()["components"]["cracks"]
        assert not cracks["ready"]
        assert cracks["detail"].startswith("crack model could not be loaded (see server log)")
        assert "manifest" not in cracks["detail"]
        assert tmp_path.name not in cracks["detail"]
        assert tampered.post(f"{V1}/strength/predict", json=MIX).status_code == 200


@pytest.mark.parametrize("scores", [None, [], "x", {"mae": "high", "rmse": True}])
def test_malformed_optional_metrics_are_ignored(
    artifacts: Path, tmp_path: Path, scores: object
) -> None:
    import shutil

    shutil.copytree(artifacts / "strength", tmp_path / "strength")
    with TestClient(create_app(api_config(tmp_path), settings())) as probe:
        version = probe.get(f"{V1}/health").json()["components"]["strength"]["model_version"]
    metrics = {"model_version": version, "scores": scores, "joint_coverage": None}
    (tmp_path / "strength" / "test_metrics.json").write_text(json.dumps(metrics), "utf-8")
    with TestClient(create_app(api_config(tmp_path), settings())) as service:
        assert service.get(f"{V1}/models").json()["strength"]["test_metrics"] == {}


def test_report_is_unavailable_until_the_module_is_wired(client: TestClient) -> None:
    response = client.post(f"{V1}/report", json={"case": {"case_id": "PB-1", "crack": CRACK}})
    assert response.status_code == 503
    assert response.json()["detail"].startswith("report module not available")
    assert client.get(f"{V1}/report/cases").status_code == 503


def test_report_route_uses_the_adapter(artifacts: Path) -> None:
    fake = FakeReportService([])
    app = create_app(api_config(artifacts), settings(), report_service=fake)
    with TestClient(app) as wired:
        assert wired.get(f"{V1}/health").json()["status"] == "ok"
        case = {"case_id": "PB-1", "crack": CRACK}
        response = wired.post(f"{V1}/report", json={"case": case, "source": "template"})
        assert response.status_code == 200
        assert response.json()["checks"][0]["text"] == "0,97"
        assert fake.requests[0].case.crack is not None
        # A strength result without the mix it came from violates the shared contract.
        invalid = {"case_id": "PB-1", "strength": {"point_mpa": 1}}
        assert wired.post(f"{V1}/report", json={"case": invalid}).status_code == 422
        assert wired.post(f"{V1}/report", json={"case": case, "source": "x"}).status_code == 422


def test_live_reports_are_refused_in_replay_mode(artifacts: Path) -> None:
    fake = FakeReportService([])
    app = create_app(api_config(artifacts), settings("replay"), report_service=fake)
    with TestClient(app) as demo:
        assert demo.get(f"{V1}/health").json()["llm_mode"] == "replay"
        case = {"case_id": "PB-1", "crack": CRACK}
        response = demo.post(f"{V1}/report", json={"case": case, "source": "live"})
        assert response.status_code == 403
        assert demo.post(f"{V1}/report", json={"case": case, "source": "replay"}).is_success
        assert len(fake.requests) == 1


def test_request_ids_are_kept_only_when_well_formed(client: TestClient) -> None:
    kept = client.get(f"{V1}/health", headers={"X-Request-ID": "dashboard-1234"})
    assert kept.headers["X-Request-ID"] == "dashboard-1234"
    replaced = client.get(f"{V1}/health", headers={"X-Request-ID": "bad id\r\nx"})
    assert replaced.headers["X-Request-ID"] != "bad id\r\nx"


def test_cors_allows_only_the_dashboard_origin(client: TestClient) -> None:
    preflight = {"Access-Control-Request-Method": "POST"}
    allowed = client.options(
        f"{V1}/strength/predict", headers={"Origin": "http://localhost:8501", **preflight}
    )
    assert allowed.headers["access-control-allow-origin"] == "http://localhost:8501"
    other = client.options(
        f"{V1}/strength/predict", headers={"Origin": "https://example.org", **preflight}
    )
    assert "access-control-allow-origin" not in other.headers


def test_openapi_documents_examples_and_errors(client: TestClient) -> None:
    schema = client.get(f"{V1}/openapi.json").json()
    strength = schema["paths"][f"{V1}/strength/predict"]["post"]
    examples = strength["requestBody"]["content"]["application/json"]["examples"]
    assert examples["typical"]["value"]["age_days"] == 28
    assert {"413", "415", "422", "503"} <= set(
        schema["paths"][f"{V1}/cracks/predict"]["post"]["responses"]
    )
    error_ref = "#/components/schemas/ErrorResponse"
    for path in ("strength/predict", "report"):
        response = schema["paths"][f"{V1}/{path}"]["post"]["responses"]["422"]
        assert response["content"]["application/json"]["schema"]["$ref"] == error_ref


def test_access_log_has_no_file_names(client: TestClient, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("INFO", logger="baulens.api.access")
    upload(client, image_bytes(), name="Bauteil-Geheim.jpg")
    records = [r for r in caplog.records if r.name == "baulens.api.access"]
    assert records
    fields = records[-1].fields  # type: ignore[attr-defined]
    assert fields["path"] == f"{V1}/cracks/predict"
    assert "Geheim" not in json.dumps(fields)


def test_cli_serve_passes_settings_to_uvicorn(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}
    monkeypatch.setattr("uvicorn.run", lambda app, **kwargs: calls.update(kwargs))
    result = CliRunner().invoke(
        cli_app, ["api", "serve", "--config", "configs/smoke/api.yaml", "--port", "9001"]
    )
    assert result.exit_code == 0, result.output
    assert calls["port"] == 9001
    assert calls["host"] == "127.0.0.1"


@pytest.mark.smoke
def test_api_smoke_end_to_end(artifacts: Path) -> None:
    """Strength and crack inference through HTTP, as the dashboard calls them."""
    with TestClient(create_app(api_config(artifacts), settings())) as smoke:
        assert smoke.get(f"{V1}/health").json()["components"]["strength"]["ready"]
        strength = smoke.post(f"{V1}/strength/predict", json=MIX)
        assert strength.status_code == 200
        cracks = upload(smoke, image_bytes(), query="?gradcam=true")
        assert cracks.status_code == 200
        assert cracks.json()["gradcam_png_base64"]
