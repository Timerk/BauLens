"""Guarded generation with scripted model replies, including unfaithful ones."""

import json
from pathlib import Path

import pytest
from report_helpers import both_case, crack_case, strength_case, template_reply

from baulens.config import ReportConfig, load_config
from baulens.report.clients import FakeClient, LLMError
from baulens.report.completeness import check_completeness
from baulens.report.facts import build_facts
from baulens.report.faithfulness import check_sections
from baulens.report.generate import (
    generate_report,
    generate_template,
    generation_options,
    input_sha256,
)
from baulens.report.prompting import (
    ChatMessage,
    DraftError,
    draft_schema,
    initial_messages,
    load_prompt,
    parse_draft,
    required_sections,
)
from baulens.report.replay import (
    RecordingClient,
    ReplayClient,
    load_records,
    replay_path,
    write_records,
)
from baulens.report.schemas import DISCLAIMER, InspectionReport, ReportDraft
from baulens.report.template import template_draft
from baulens.schemas import InspectionInput

CONFIG = load_config(Path("configs/report.yaml"), ReportConfig)
FREE = CONFIG.model_copy(update={"generation_mode": "free"})


def faithful_reply(case: InspectionInput, **changes: str) -> str:
    draft = template_draft(build_facts(case)).model_dump(exclude_none=True)
    draft.update(changes)
    return json.dumps(draft, ensure_ascii=False)


def test_prompt_lists_the_facts_and_sections() -> None:
    sheet = build_facts(both_case(in_training_domain=False))
    system, user = initial_messages(sheet, load_prompt())
    assert system.role == "system" and "Erfinde keine Zahlen" in system.content
    assert "Prognostizierte Druckfestigkeit: 61,9 MPa" in user.content
    assert "Risswahrscheinlichkeit: 93,1 %" in user.content
    assert "außerhalb des Wertebereichs der Trainingsdaten" in user.content
    assert "- rissbefund:" in user.content
    assert load_prompt().prompt_version.startswith("report-v2+")
    assert "Alle Eingaben liegen im Wertebereich" not in user.content


def test_schema_requires_exactly_the_sections_of_the_case() -> None:
    assert required_sections(build_facts(strength_case())) == (
        "zusammenfassung",
        "druckfestigkeit",
        "einschraenkungen",
        "empfehlung",
    )
    schema = draft_schema(build_facts(crack_case()))
    assert schema["required"] == ["zusammenfassung", "rissbefund", "einschraenkungen", "empfehlung"]
    assert schema["additionalProperties"] is False


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("not json", "kein gültiges JSON"),
        ("[]", "JSON-Objekt erwartet"),
        ('{"zusammenfassung": "x"}', "fehlende Felder"),
        (
            '{"zusammenfassung": "x", "rissbefund": "y", "einschraenkungen": "z", '
            '"empfehlung": "w", "druckfestigkeit": "v"}',
            "unzulässige Felder",
        ),
        (
            '{"zusammenfassung": " ", "rissbefund": "y", "einschraenkungen": "z", '
            '"empfehlung": "w"}',
            "zusammenfassung",
        ),
    ],
)
def test_parse_draft_rejects_invalid_replies(text: str, message: str) -> None:
    with pytest.raises(DraftError, match=message):
        parse_draft(text, build_facts(crack_case()))


def test_faithful_reply_is_accepted_and_delivered() -> None:
    case = both_case()
    client = FakeClient([faithful_reply(case)], model="fake-a")
    result = generate_report(case, client, FREE)
    assert result.status == "accepted"
    assert result.source == "fake"
    assert result.delivered.generator == "fake-a"
    assert result.delivered.disclaimer == DISCLAIMER
    assert not result.fallback_used
    assert result.completeness is not None and result.completeness.score == 1.0
    assert len(client.calls) == 1
    assert result.input_sha256 == input_sha256(case)


def test_corrective_retry_lists_the_offending_numbers() -> None:
    case = both_case()
    bad = faithful_reply(case, empfehlung="Es sind 3 Bohrkerne nach 7 Tagen zu entnehmen.")
    client = FakeClient([bad, faithful_reply(case)])
    result = generate_report(case, client, FREE)
    assert result.status == "accepted"
    assert [attempt.kind for attempt in result.attempts] == ["initial", "correction"]
    assert result.initial_faithfulness is not None
    assert len(result.initial_faithfulness.violations) == 2
    correction = client.calls[1][-1].content
    assert "„3“" in correction and "„7 Tagen“" in correction
    assert client.calls[1][-2] == ChatMessage(role="assistant", content=bad)


def test_persistently_unfaithful_report_is_rejected_never_delivered() -> None:
    case = strength_case()
    bad = faithful_reply(case, zusammenfassung="Die Druckfestigkeit beträgt 65,3 MPa.")
    result = generate_report(case, FakeClient([bad, bad]), FREE)
    assert result.status == "rejected"
    assert result.fallback_used
    assert result.delivered.generator == "template"
    assert result.candidate is not None and "65,3" in result.candidate.draft.zusammenfassung
    assert result.faithfulness is not None and not result.faithfulness.faithful
    assert result.note is not None and "Vorlagenbericht" in result.note


def test_invalid_json_gets_one_schema_retry_then_fails_cleanly() -> None:
    case = crack_case()
    recovered = generate_report(case, FakeClient(["{", faithful_reply(case)]), FREE)
    assert recovered.status == "accepted"
    assert [attempt.kind for attempt in recovered.attempts] == ["initial", "schema_retry"]
    assert recovered.attempts[0].schema_error is not None

    failed = generate_report(case, FakeClient(["{", "{"]), FREE)
    assert failed.status == "invalid"
    assert failed.candidate is None and failed.completeness is None
    assert failed.delivered.generator == "template"


def test_invalid_correction_keeps_the_rejection() -> None:
    case = crack_case()
    bad = faithful_reply(case, empfehlung="Wiederholung in 6 Monaten.")
    result = generate_report(case, FakeClient([bad, "{", "{"]), FREE)
    assert result.status == "rejected"
    assert [attempt.kind for attempt in result.attempts] == [
        "initial",
        "correction",
        "schema_retry",
    ]


def test_service_errors_fall_back_to_the_template() -> None:
    result = generate_report(crack_case(), FakeClient([]), FREE)
    assert result.status == "error"
    assert result.delivered.generator == "template"
    assert result.note is not None and "fehlgeschlagen" in result.note


def test_template_report_is_faithful_and_complete() -> None:
    for case in (strength_case(), crack_case(0.2), both_case(in_training_domain=False)):
        result = generate_template(case)
        assert result.status == "template"
        assert result.faithfulness is not None and result.faithfulness.faithful
        assert result.completeness is not None and result.completeness.score == 1.0


def test_completeness_flags_missing_facts_and_caveats() -> None:
    case = both_case(in_training_domain=False)
    sheet = build_facts(case)
    draft = ReportDraft(
        zusammenfassung="Die Druckfestigkeit beträgt 61,9 MPa.",
        druckfestigkeit="Prognose 61,9 MPa.",
        rissbefund="Es wurde kein Riss festgestellt.",
        einschraenkungen="Keine.",
        empfehlung="Keine.",
    )
    report = InspectionReport(case_id=case.case_id, draft=draft, generator="fake")
    result = check_completeness(report, check_sections(draft.sections(), sheet), sheet)
    missing = set(result.missing)
    assert {
        "interval_bounds",
        "interval_coverage",
        "age",
        "out_of_domain_warning",
        "testing_reference",
        "crack_probability",
        "crack_label",
        "patch_caveat",
        "visual_inspection",
    } <= missing
    assert "strength_point" not in missing and "estimate_caveat" not in missing
    assert 0 < result.score < 0.5


def test_recorded_replies_replay_the_same_path(tmp_path: Path) -> None:
    case = both_case()
    bad = faithful_reply(case, empfehlung="Wiederholung in 6 Monaten.")
    recorder = RecordingClient(FakeClient([bad, faithful_reply(case)], model="m:1b"))
    live = generate_report(case, recorder, FREE)
    path = replay_path(tmp_path, "m:1b")
    assert path.name == "m_1b.jsonl"
    write_records(path, recorder.records)
    records = load_records(path)
    assert [record.call for record in records] == [0, 1]
    assert records[0].prompt_version == load_prompt().prompt_version
    assert records[0].input_sha256 == input_sha256(case)

    replay = ReplayClient.from_directory(tmp_path, "m:1b")
    replayed = generate_report(case, replay, FREE)
    assert replayed.source == "replay"
    assert replayed.status == live.status == "accepted"
    assert replayed.delivered.draft == live.delivered.draft
    assert "Replay" in replayed.delivered.provenance
    assert [a.kind for a in replayed.attempts] == [a.kind for a in live.attempts]


def test_unknown_input_in_replay_mode_falls_back_and_says_so(tmp_path: Path) -> None:
    result = generate_report(crack_case(), ReplayClient.from_directory(tmp_path, "m"), FREE)
    assert result.status == "replay_miss"
    assert result.source == "replay"
    assert result.fallback_used and result.delivered.generator == "template"
    assert "Ersatz" in result.delivered.to_markdown()
    assert result.note is not None and "Keine aufgezeichnete Antwort" in result.note


def test_fake_client_rejects_missing_script() -> None:
    client = FakeClient(["a"])
    options = generation_options(CONFIG)
    client.generate([], {}, options)
    with pytest.raises(LLMError):
        client.generate([], {}, options)


@pytest.mark.parametrize(
    ("sentence", "consistent"),
    [
        ("Die Prognose liegt außerhalb des Wertebereichs der Trainingsdaten.", False),
        ("Die Prognose ist unsicher, wenn Eingaben außerhalb des Trainingsbereichs liegen.", True),
        ("Die Eingaben liegen nicht außerhalb des Wertebereichs der Trainingsdaten.", True),
        ("Die Eingaben liegen innerhalb des Wertebereichs der Trainingsdaten.", True),
    ],
)
def test_in_domain_case_must_not_be_declared_out_of_domain(sentence: str, consistent: bool) -> None:
    case = strength_case()
    sheet = build_facts(case)
    draft = template_draft(sheet).model_copy(update={"einschraenkungen": sentence})
    report = InspectionReport(case_id=case.case_id, draft=draft, generator="fake")
    result = check_completeness(report, check_sections(draft.sections(), sheet), sheet)
    item = next(item for item in result.items if item.name == "no_false_domain_warning")
    assert item.satisfied is consistent


def slot_reply(case: InspectionInput, **changes: str) -> str:
    return template_reply(case, slots=True, **changes)


def test_slot_mode_inserts_numbers_by_code() -> None:
    case = both_case()
    reply = slot_reply(case)
    assert "61,9" not in reply and "{druckfestigkeit}" in reply and "MPa" not in reply
    client = FakeClient([reply], model="fake-s")
    result = generate_report(case, client, CONFIG)
    assert CONFIG.generation_mode == "slots" and result.mode == "slots"
    assert result.status == "accepted", result.faithfulness
    assert "61,9 MPa" in result.delivered.draft.zusammenfassung
    assert result.attempts[0].raw_draft is not None
    assert "{druckfestigkeit}" in result.attempts[0].raw_draft["zusammenfassung"]
    assert "{druckfestigkeit}" in client.calls[0][1].content
    assert "Modus slots" in result.delivered.to_markdown()


def test_slot_mode_rejects_literal_numbers_and_misplaced_slots() -> None:
    case = strength_case()
    literal = slot_reply(case, empfehlung="Es sind drei Bohrkerne nach 28 Tagen zu prüfen.")
    corrected = generate_report(case, FakeClient([literal, slot_reply(case)]), CONFIG)
    assert corrected.status == "accepted"
    first = corrected.initial_faithfulness
    assert first is not None
    assert [check.reason for check in first.violations] == ["Zahl außerhalb eines Platzhalters"] * 2

    misplaced = slot_reply(case, zusammenfassung="Die Druckfestigkeit beträgt {obere_grenze}.")
    rejected = generate_report(case, FakeClient([misplaced, misplaced]), CONFIG)
    assert rejected.status == "rejected" and rejected.fallback_used
    assert rejected.faithfulness is not None
    assert [check.status for check in rejected.faithfulness.violations] == ["misattributed"]


def test_delivered_markdown_carries_its_provenance(tmp_path: Path) -> None:
    case = crack_case()
    live = generate_report(case, FakeClient([faithful_reply(case)], model="m"), FREE)
    assert "Testausgabe von m" in live.delivered.to_markdown()
    template = generate_template(case).delivered.to_markdown()
    assert "Herkunft: Vorlagenbericht ohne Sprachmodell" in template


def test_replay_lookup_uses_the_full_input_identity(tmp_path: Path) -> None:
    case = crack_case()
    recorder = RecordingClient(FakeClient([slot_reply(case)], model="m"))
    generate_report(case, recorder, CONFIG)
    write_records(replay_path(tmp_path, "m"), recorder.records)
    replay = ReplayClient.from_directory(tmp_path, "m")
    assert generate_report(case, replay, CONFIG).status == "accepted"
    assert case.image is not None
    changed = case.model_copy(update={"image": case.image.model_copy(update={"sha256": "a" * 64})})
    # Same messages (the hash is not in the prompt), but a different input: a miss.
    assert initial_messages(build_facts(changed), load_prompt("report_slots_v2")) == (
        initial_messages(build_facts(case), load_prompt("report_slots_v2"))
    )
    assert generate_report(changed, replay, CONFIG).status == "replay_miss"


def test_crack_finding_must_be_stated_explicitly() -> None:
    case = crack_case()
    sheet = build_facts(case)
    draft = template_draft(sheet).model_copy(
        update={
            "rissbefund": "Risswahrscheinlichkeit: 93,1 %. Entscheidungsschwelle: 50 %. "
            "Einzelner Bildausschnitt."
        }
    )
    report = InspectionReport(case_id=case.case_id, draft=draft, generator="fake")
    result = check_completeness(report, check_sections(draft.sections(), sheet), sheet)
    assert "crack_label" in result.missing and result.score < 1.0


def test_slot_expansion_beyond_the_section_limit_falls_back() -> None:
    """A long element name repeated through {bauteil} must not crash generate_report."""
    image = crack_case().image
    assert image is not None
    long_image = image.model_copy(update={"element": "Stütze " + "a" * 110})
    case = InspectionInput(case_id="PB-040", image=long_image, crack=crack_case().crack)
    long_summary = "; ".join(["{bauteil}"] * 18)
    reply = slot_reply(case, zusammenfassung=long_summary)
    result = generate_report(case, FakeClient([reply, reply]), CONFIG)
    assert result.status == "invalid" and result.fallback_used
    assert result.attempts[0].schema_error is not None
    assert "Platzhalter" in result.attempts[0].schema_error
