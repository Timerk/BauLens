"""Ollama HTTP handling, evaluation cases, replays, comparison and CLI of the report module."""

import json
import os
from collections.abc import Sequence
from pathlib import Path

import httpx
import pytest
from report_helpers import both_case, crack_case
from typer.testing import CliRunner

from baulens.cli import app
from baulens.config import ReportConfig, load_config
from baulens.report.cases import SCENARIOS, generate_cases, load_cases
from baulens.report.clients import FakeClient, LLMError, OllamaClient
from baulens.report.compare import DOC_BEGIN, DOC_END, resumable_metrics, run_comparison
from baulens.report.facts import build_facts
from baulens.report.generate import generate_report, generation_options
from baulens.report.prompting import ChatMessage, initial_messages, load_prompt
from baulens.report.replay import ReplayClient
from baulens.report.template import template_draft
from baulens.schemas import InspectionInput

CONFIG = load_config(Path("configs/report.yaml"), ReportConfig)
CASES_PATH = Path(__file__).parents[1] / "data" / "report_cases" / "cases.jsonl"
REPLAY_DIR = Path(__file__).parents[1] / "data" / "report_replays"
runner = CliRunner()


def _chat_reply(content: str) -> dict[str, object]:
    return {
        "model": "qwen2.5:3b",
        "message": {"role": "assistant", "content": content},
        "done_reason": "stop",
        "eval_count": 120,
        "eval_duration": 6_000_000_000,
        "prompt_eval_count": 900,
        "prompt_eval_duration": 2_000_000_000,
        "load_duration": 10_000_000,
        "total_duration": 8_100_000_000,
    }


def test_ollama_client_sends_structured_output_request() -> None:
    seen: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=_chat_reply('{"a": "b"}'))

    client = OllamaClient(
        "http://ollama:11434/", "qwen2.5:3b", transport=httpx.MockTransport(handler)
    )
    messages = [ChatMessage(role="user", content="hallo")]
    response = client.generate(messages, {"type": "object"}, generation_options(CONFIG))
    assert response.text == '{"a": "b"}'
    assert response.source == "live"
    assert response.eval_count == 120 and response.eval_duration_ns == 6_000_000_000
    (payload,) = seen
    assert payload["stream"] is False
    assert payload["format"] == {"type": "object"}
    assert payload["options"] == {
        "temperature": 0.0,
        "seed": 42,
        "num_predict": 1024,
        "num_ctx": 4096,
    }
    assert payload["messages"] == [{"role": "user", "content": "hallo"}]


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(500, text="boom"),
        httpx.Response(200, text="not json"),
        httpx.Response(200, json={"error": "model not found"}),
        httpx.Response(200, json={"message": {"role": "assistant"}}),
    ],
)
def test_ollama_client_raises_llm_error(response: httpx.Response) -> None:
    client = OllamaClient("http://ollama", "m", transport=httpx.MockTransport(lambda _: response))
    with pytest.raises(LLMError):
        client.generate([], {}, generation_options(CONFIG))


def test_ollama_client_wraps_transport_errors_and_reads_ps() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": [{"name": "m", "size": 2_000_000_000}]})
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.40.18"})
        raise httpx.ConnectError("refused", request=request)

    client = OllamaClient("http://ollama", "m", transport=httpx.MockTransport(handler))
    assert client.loaded_size_bytes() == 2_000_000_000
    assert client.version() == "0.40.18"
    with pytest.raises(LLMError, match="refused"):
        client.unload()
    result = generate_report(crack_case(), client, CONFIG)
    assert result.status == "error" and result.delivered.generator == "template"


def test_committed_cases_match_the_seeded_generator() -> None:
    committed = load_cases(CASES_PATH)
    assert committed == generate_cases()
    assert 24 <= len(committed) <= 30 == len(SCENARIOS) + 2


def test_cases_cover_the_required_situations() -> None:
    cases = load_cases(CASES_PATH)
    strength = [case.strength for case in cases if case.strength is not None]
    crack = [case.crack for case in cases if case.crack is not None]
    assert any(case.crack is None for case in cases)
    assert any(case.strength is None for case in cases)
    assert any(case.crack is not None and case.strength is not None for case in cases)
    assert {prediction.label for prediction in crack} == {"crack", "no_crack"}
    assert sum(abs(p.crack_probability - p.threshold) < 0.05 for p in crack) >= 4
    assert sum(not prediction.in_training_domain for prediction in strength) >= 3
    ages = {case.mix.age_days for case in cases if case.mix is not None}
    assert {1, 2, 3, 365} <= ages
    widths = [prediction.upper_mpa - prediction.lower_mpa for prediction in strength]
    assert min(widths) < 11 and max(widths) > 24
    assert any(prediction.lower_mpa < 0 for prediction in strength)
    elements = [case.image.element or "" for case in cases if case.image is not None]
    assert any(char in "äöüÄÖÜß" for element in elements for char in element)


def test_template_is_faithful_and_complete_on_every_case() -> None:
    from baulens.report.generate import generate_template

    for case in load_cases(CASES_PATH):
        result = generate_template(case)
        assert result.faithfulness is not None and result.faithfulness.faithful, case.case_id
        assert result.completeness is not None and result.completeness.score == 1.0


@pytest.mark.parametrize("model", CONFIG.models)
def test_committed_replays_cover_every_case(model: str) -> None:
    replay = ReplayClient.from_directory(REPLAY_DIR, model)
    if len(replay) == 0:
        pytest.skip(f"no replay recorded for {model}")
    for case in load_cases(CASES_PATH):
        result = generate_report(case, replay, CONFIG)
        assert result.source == "replay", (case.case_id, result.note)
        assert result.status in {"accepted", "rejected", "invalid"}


def _scripted_reply(cases: list[InspectionInput]) -> FakeClient:
    facts = {build_facts(case).render(): case for case in cases}

    def reply(messages: Sequence[ChatMessage]) -> str:
        case = next(c for rendered, c in facts.items() if rendered in messages[1].content)
        draft = template_draft(build_facts(case)).model_dump(exclude_none=True)
        if case.case_id == cases[0].case_id and len(messages) == 2:
            draft["empfehlung"] += " Es sind 3 Bohrkerne zu entnehmen."
        return json.dumps(draft, ensure_ascii=False)

    return FakeClient(reply, model="fake:1b")


@pytest.mark.smoke
def test_compare_writes_results_replays_figure_and_doc(tmp_path: Path) -> None:
    cases = load_cases(CASES_PATH)[:3]
    doc = tmp_path / "report.md"
    doc.write_text(f"# Results\n\n{DOC_BEGIN}\nold\n{DOC_END}\n", encoding="utf-8")
    config = load_config(Path("configs/smoke/report.yaml"), ReportConfig).model_copy(
        update={
            "artifacts_dir": tmp_path / "artifacts",
            "replay_dir": tmp_path / "replays",
            "results_doc": doc,
            "figures_dir": tmp_path / "figures",
        }
    )
    run = run_comparison(
        config, cases, models=["fake:1b"], client_factory=lambda _: _scripted_reply(cases)
    )
    fake, template = run.metrics
    assert fake.cases == template.cases == 3
    assert fake.faithful_initial == pytest.approx(2 / 3)
    assert fake.faithful_final == 1.0
    assert fake.violations_per_report["invented"] == pytest.approx(1 / 3)
    assert template.faithful_initial == 1.0
    run_dir = Path(run.run_dir)
    assert len((run_dir / "results_fake_1b.jsonl").read_text(encoding="utf-8").splitlines()) == 3
    assert (tmp_path / "replays" / "fake_1b.jsonl").exists()
    assert (tmp_path / "figures" / "model_comparison.png").stat().st_size > 10_000
    text = doc.read_text(encoding="utf-8")
    assert "old" not in text and "| fake:1b | 3 |" in text
    assert resumable_metrics(run_dir, "fake:1b", 3) == fake


@pytest.mark.smoke
def test_cli_generate_check_and_reject_path(tmp_path: Path) -> None:
    case_path = tmp_path / "case.json"
    case_path.write_text(both_case().model_dump_json(), encoding="utf-8")
    result = runner.invoke(
        app, ["report", "generate", "--input", str(case_path), "--mode", "template"]
    )
    assert result.exit_code == 0, result.output
    assert "Prüfberichtsentwurf PB-022" in result.stdout and "status: template" in result.stdout

    replay = runner.invoke(
        app,
        [
            "report",
            "generate",
            "--input",
            str(case_path),
            "--mode",
            "replay",
            "--model",
            "none:0b",
            "--json",
        ],
    )
    assert replay.exit_code == 0
    assert json.loads(replay.stdout)["note"].startswith("Keine aufgezeichnete Antwort")

    good = tmp_path / "good.txt"
    good.write_text("Die Druckfestigkeit beträgt 61,9 MPa.", encoding="utf-8")
    bad = tmp_path / "bad.txt"
    bad.write_text("Die Druckfestigkeit beträgt 65,3 MPa bei 28 MPa.", encoding="utf-8")
    ok = runner.invoke(app, ["report", "check", "--input", str(case_path), "--text", str(good)])
    assert ok.exit_code == 0 and "0 violations" in ok.stdout
    flagged = runner.invoke(app, ["report", "check", "--input", str(case_path), "--text", str(bad)])
    assert flagged.exit_code == 1
    assert "invented" in flagged.stdout and "misattributed" in flagged.stdout


@pytest.mark.smoke
def test_fake_generation_rejects_and_falls_back() -> None:
    case = both_case()
    bad = json.dumps(
        {**template_draft(build_facts(case)).model_dump(exclude_none=True), "empfehlung": "C30/37"},
        ensure_ascii=False,
    )
    result = generate_report(case, FakeClient([bad, bad]), CONFIG)
    assert result.status == "rejected" and result.delivered.generator == "template"
    assert {check.text for check in result.faithfulness.violations} == {"30", "37"}  # type: ignore[union-attr]


@pytest.mark.slow
def test_live_ollama_report_is_valid() -> None:
    """Needs a running Ollama with the first configured model; never runs in CI."""
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    client = OllamaClient(host, CONFIG.models[0])
    if client.version() is None:
        pytest.skip("Ollama is not reachable")
    case = load_cases(CASES_PATH)[0]
    assert initial_messages(build_facts(case), load_prompt())
    result = generate_report(case, client, CONFIG)
    assert result.status in {"accepted", "rejected"}
    assert result.candidate is not None
