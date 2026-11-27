"""Ollama HTTP handling, evaluation cases, replays, comparison and CLI of the report module."""

import json
import os
from collections.abc import Sequence
from pathlib import Path

import httpx
import pytest
from report_helpers import both_case, crack_case, template_reply
from typer.testing import CliRunner

from baulens.cli import app
from baulens.config import ReportConfig, load_config
from baulens.report.cases import SCENARIOS, generate_cases, load_cases
from baulens.report.clients import FakeClient, LLMError, OllamaClient
from baulens.report.compare import DOC_BEGIN, DOC_END, load_results, reusable, run_comparison
from baulens.report.facts import build_facts
from baulens.report.generate import generate_report, generation_options
from baulens.report.metrics import compute_metrics
from baulens.report.prompting import ChatMessage, initial_messages, load_prompt
from baulens.report.replay import RecordingClient, ReplayClient, load_records
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


def test_malformed_ollama_metadata_falls_back_to_the_template() -> None:
    # The review's payload: a 200 with an empty object and a list as done_reason.
    payload = {"message": {"content": "{}"}, "done_reason": [], "eval_count": -3}
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json=payload))
    client = OllamaClient("http://ollama", "m", transport=transport)
    response = client.generate([], {}, generation_options(CONFIG))
    assert response.done_reason is None and response.eval_count is None
    result = generate_report(crack_case(), client, CONFIG)
    assert result.status == "invalid" and result.fallback_used
    assert result.delivered.generator == "template"
    assert "Ersatz" in result.delivered.provenance


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


@pytest.mark.parametrize("mode", CONFIG.compare_modes)
@pytest.mark.parametrize("model", CONFIG.models)
def test_committed_replays_cover_every_case(model: str, mode: str) -> None:
    replay = ReplayClient.from_directory(REPLAY_DIR, model)
    assert len(replay) > 0, f"no replay recorded for {model}"
    for case in load_cases(CASES_PATH):
        result = generate_report(case, replay, CONFIG, mode=mode)  # type: ignore[arg-type]
        assert result.source == "replay"
        assert result.status in {"accepted", "rejected", "invalid"}, (case.case_id, result.note)


def _scripted_reply(cases: list[InspectionInput]) -> FakeClient:
    """Template-based replies for both modes; the first case first invents a number."""

    def reply(messages: Sequence[ChatMessage]) -> str:
        case = next(c for c in cases if c.case_id in messages[1].content)
        slots = "Platzhalter" in messages[0].content
        extra = (
            " Es sind 3 Bohrkerne zu entnehmen." if case is cases[0] and len(messages) == 2 else ""
        )
        draft = json.loads(template_reply(case, slots=slots))
        draft["empfehlung"] += extra
        return json.dumps(draft, ensure_ascii=False)

    return FakeClient(reply, model="fake:1b")


def _smoke_config(tmp_path: Path, doc: Path | None = None) -> ReportConfig:
    return load_config(Path("configs/smoke/report.yaml"), ReportConfig).model_copy(
        update={
            "artifacts_dir": tmp_path / "artifacts",
            "replay_dir": tmp_path / "replays",
            "results_doc": doc,
            "figures_dir": tmp_path / "figures",
        }
    )


@pytest.mark.smoke
def test_compare_runs_both_modes_and_writes_outputs(tmp_path: Path) -> None:
    cases = load_cases(CASES_PATH)[:3]
    doc = tmp_path / "report.md"
    doc.write_text("\n".join(["# Results", "", DOC_BEGIN, "old", DOC_END, ""]), encoding="utf-8")
    config = _smoke_config(tmp_path, doc)
    run = run_comparison(
        config, cases, models=["fake:1b"], client_factory=lambda _: _scripted_reply(cases)
    )
    free, slots, template = run.metrics
    assert (free.mode, slots.mode, template.mode) == ("free", "slots", None)
    for row in (free, slots):
        assert row.cases == 3
        assert row.faithful_initial == pytest.approx(2 / 3)
        assert row.faithful_final == 1.0
        assert row.violations_per_report["invented"] == pytest.approx(1 / 3)
    assert template.faithful_initial == template.faithful_final == 1.0
    run_dir = Path(run.run_dir)
    for mode in ("free", "slots"):
        lines = (run_dir / f"results_fake_1b__{mode}.jsonl").read_text(encoding="utf-8")
        assert len(lines.splitlines()) == 3
    records = load_records(tmp_path / "replays" / "fake_1b.jsonl")
    assert len(records) == 8  # 3 cases x 2 modes + one correction per mode
    assert (tmp_path / "figures" / "model_comparison.png").stat().st_size > 10_000
    text = doc.read_text(encoding="utf-8")
    assert "old" not in text and "| fake:1b / slots | 3 |" in text
    assert reusable(run_dir, "fake:1b", "slots", cases, config, "fake")[0] == slots


@pytest.mark.smoke
def test_resume_rejects_changed_inputs_and_settings(tmp_path: Path) -> None:
    cases = load_cases(CASES_PATH)[:2]
    config = _smoke_config(tmp_path).model_copy(update={"compare_modes": ["free"]})
    run = run_comparison(
        config, cases, models=["fake:1b"], client_factory=lambda _: _scripted_reply(cases)
    )
    run_dir = Path(run.run_dir)
    assert reusable(run_dir, "fake:1b", "free", cases, config, "fake")[0] is not None
    assert cases[0].mix is not None
    changed_mix = cases[0].mix.model_copy(update={"cement": 950.0})
    changed = [cases[0].model_copy(update={"mix": changed_mix}), cases[1]]
    assert reusable(run_dir, "fake:1b", "free", changed, config, "fake") == (
        None,
        "inputs differs from the earlier run",
    )
    renamed = [cases[0].model_copy(update={"case_id": "PB-999"}), cases[1]]
    assert reusable(run_dir, "fake:1b", "free", renamed, config, "fake")[0] is None
    # A run recorded from one source is never reused as another (live vs replay).
    assert reusable(run_dir, "fake:1b", "free", cases, config, "replay") == (
        None,
        "source differs from the earlier run",
    )
    hotter = config.model_copy(update={"temperature": 0.7})
    assert (
        reusable(run_dir, "fake:1b", "free", cases, hotter, "fake")[1]
        == "options differs from the earlier run"
    )


@pytest.mark.smoke
def test_replay_misses_are_counted_separately() -> None:
    cases = load_cases(CASES_PATH)[:2]
    results = [generate_report(case, ReplayClient("none", []), CONFIG) for case in cases]
    metrics = compute_metrics("none", results, mode="slots")
    assert metrics.replay_misses == 2
    assert metrics.schema_valid == metrics.faithful_initial == metrics.faithful_final == 0.0
    assert metrics.completeness_mean is None and metrics.calls_per_case == 0.0


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
    result = generate_report(case, FakeClient([bad, bad]), CONFIG, mode="free")
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


@pytest.mark.smoke
def test_compare_from_replay_uses_the_recorded_replies(tmp_path: Path) -> None:
    """`report compare --source replay` must find replays recorded by a live comparison."""
    cases = load_cases(CASES_PATH)[:2]
    config = _smoke_config(tmp_path).model_copy(update={"compare_modes": ["slots"]})
    run_comparison(
        config, cases, models=["fake:1b"], client_factory=lambda _: _scripted_reply(cases)
    )
    replayed = run_comparison(
        config.model_copy(update={"artifacts_dir": tmp_path / "replayed"}),
        cases,
        models=["fake:1b"],
        client_factory=lambda model: ReplayClient.from_directory(tmp_path / "replays", model),
        record_replays=False,
        source="replay",
    )
    row = replayed.metrics[0]
    assert row.replay_misses == 0 and row.faithful_final == 1.0


@pytest.mark.smoke
def test_partial_replay_misses_are_excluded_from_rates(tmp_path: Path) -> None:
    case = load_cases(CASES_PATH)[0]
    recorder = RecordingClient(_scripted_reply([case]))
    generate_report(case, recorder, CONFIG)
    assert len(recorder.records) == 2  # first draft invents a number, the correction is clean
    first_only = ReplayClient("fake:1b", recorder.records[:1])
    partial = generate_report(case, first_only, CONFIG)
    assert partial.status == "replay_miss" and len(partial.attempts) == 1
    metrics = compute_metrics("fake:1b", [partial], mode="slots")
    assert metrics.replay_misses == 1 and metrics.cases == 1
    assert metrics.schema_valid == metrics.faithful_initial == 0.0
    assert metrics.calls_per_case == 0.0 and metrics.latency_median_s is None


@pytest.mark.smoke
def test_cli_compare_from_committed_replays_has_no_misses() -> None:
    """The documented offline command re-scores committed records without a model."""
    result = runner.invoke(
        app,
        ["report", "compare", "--source", "replay", "--model", "qwen2.5:3b", "--limit", "1",
         "--no-replays", "--no-docs"],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    run_dir = Path(result.stdout.strip().splitlines()[-1].removeprefix("results: "))
    for mode in CONFIG.compare_modes:
        rows = load_results(run_dir / f"results_qwen2.5_3b__{mode}.jsonl")
        assert [row.status for row in rows] != ["replay_miss"]
        assert all(row.source == "replay" for row in rows)
