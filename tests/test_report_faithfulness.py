"""Adversarial and regression tests of the numeric faithfulness checker.

Facts of ``both_case()``: strength 61,9 MPa with interval 54,2 to 69,6 MPa at 90 %,
cement 540 kg/m³, water 162 kg/m³, superplasticizer 2,5 kg/m³, coarse aggregate
1040 kg/m³, w/z 0,30, age 28 days, contributions +5,2 (cement) and \u22123,1 MPa (age), crack
probability 93,1 % at a threshold of 50 %, image 227 \u00d7 227 pixels captured on 14.03.2026 at
10:15, case PB-022, element "Decke über EG, Feld C2".
"""

from datetime import datetime

import pytest
from report_helpers import IMAGE, MIX, both_case, crack, crack_case, strength, strength_case

from baulens.report.facts import build_facts, decision_preserving_percent
from baulens.report.faithfulness import (
    check_sections,
    check_slot_sections,
    check_text,
    describe_violations,
    render_slots,
)
from baulens.report.template import template_draft
from baulens.schemas import ImageMetadata, InspectionInput

BOTH = build_facts(both_case())


def verdicts(text: str, case: InspectionInput | None = None) -> list[tuple[str, str]]:
    sheet = build_facts(case) if case is not None else BOTH
    return [(check.text, check.status) for check in check_text(text, sheet).checks]


def statuses(text: str, case: InspectionInput | None = None) -> list[str]:
    return [status for _, status in verdicts(text, case)]


@pytest.mark.parametrize(
    "text",
    [
        "Die prognostizierte Druckfestigkeit beträgt 61,9 MPa.",
        "Die Druckfestigkeit beträgt 61,9 N/mm².",
        "Das Prognoseintervall reicht von 54,2 MPa bis 69,6 MPa.",
        "Das Prognoseintervall reicht von 54,2 bis 69,6 MPa.",
        "Das Prognoseintervall: 54,2\u201369,6 MPa.",
        "Die Druckfestigkeit liegt zwischen 54,2 und 69,6 MPa.",
        "Die untere Grenze liegt bei 54,2 MPa, die obere Grenze bei 69,6 MPa.",
        "Die obere Grenze der Druckfestigkeit beträgt 69,6 MPa.",
        "Das 90-%-Prognoseintervall reicht von 54,2 bis 69,6 MPa.",
        "Das Prognoseintervall (90 % nominelle Überdeckung) ist breit.",
        "Die nominelle Überdeckung beträgt 90 %.",
        "Zement 540 kg/m³, Wasser 162 kg/m3, Fließmittel 2,5 kg/m³.",
        "Für die Rezeptur mit 540 kg/m³ Zement, 162 kg/m³ Wasser und 1040 kg/m³ Kies.",
        "Die grobe Gesteinskörnung beträgt 1.040 kg/m³.",
        "Die grobe Gesteinskörnung beträgt 1\u202f040 kg/m³.",
        "Der w/z-Wert beträgt 0,30.",
        "Das Betonalter beträgt 28 Tage; es handelt sich um die 28-Tage-Festigkeit.",
        "Geprüft nach achtundzwanzig Tagen.",
        "Die Risswahrscheinlichkeit beträgt 93,1 %, die Entscheidungsschwelle 50 %.",
        "Risswahrscheinlichkeit: 93,1 %.",
        "Riss erkannt (Risswahrscheinlichkeit: 93,1 %).",
        "Die Risswahrscheinlichkeit von 93,1 % liegt über der Entscheidungsschwelle von 50 %.",
        "Der Beitrag des Betonalters beträgt \u22123,1 MPa.",
        "Der Beitrag des Betonalters beträgt -3,1 MPa.",
        "Der Zement erhöht die Festigkeit um +5,2 MPa.",
        "Das Bild hat 227 \u00d7 227 Pixel und wurde am 14.03.2026 um 10:15 Uhr aufgenommen.",
        "Prüfung nach DIN EN 12390-3 oder an Bohrkernen nach DIN EN 12504-1.",
        "Fall PB-022, Bauteil Decke über EG, Feld C2, Bild IMG_0423.jpg, Modell resnet18.",
        "1. Zusammenfassung folgt.",
    ],
)
def test_accepts_exact_bound_numbers(text: str) -> None:
    result = check_text(text, BOTH)
    assert result.faithful, describe_violations(result)


# Every example from the review of PR #11 (blockers 1-3), with the expected verdicts.
REVIEW_EXAMPLES = [
    ("Die Druckfestigkeit beträgt 69,6 MPa.", [("69,6 MPa", "misattributed")]),
    (
        "Das Intervall reicht von 69,6 bis 54,2 MPa.",
        [("69,6", "misattributed"), ("54,2 MPa", "misattributed")],
    ),
    ("Risswahrscheinlichkeit: 90 %.", [("90 %", "misattributed")]),
    ("Die Zementmasse beträgt 540 t/m³.", [("540 t/m³", "misattributed")]),
    ("Aufnahme um 10:15:59 Uhr.", [("10:15:59", "invented")]),
    ("Die grobe Gesteinskörnung beträgt 1,040 kg/m³.", [("1,040 kg/m³", "invented")]),
    ("Geprüft wurden dreihundert Bohrkerne.", [("dreihundert", "invented")]),
]


@pytest.mark.parametrize(("text", "expected"), REVIEW_EXAMPLES)
def test_review_examples_are_caught(text: str, expected: list[tuple[str, str]]) -> None:
    assert verdicts(text) == expected


def test_review_rounding_against_threshold() -> None:
    case = crack_case(probability=0.4999, threshold=0.5)
    sheet = build_facts(case)
    probability = sheet.fact("crack.crack_probability")
    assert probability is not None and probability.display == "49,99 %"
    # 50 % is the threshold's value, so calling it the probability is a misattribution.
    assert statuses("Die Risswahrscheinlichkeit beträgt 50 %.", case) == ["misattributed"]
    assert statuses("Die Risswahrscheinlichkeit beträgt 49,9 %.", case) == ["inexact"]
    assert statuses("Die Risswahrscheinlichkeit beträgt 49,99 %.", case) == ["ok"]
    assert "49,99 %" in template_draft(sheet).rissbefund  # type: ignore[operator]


def test_review_no_rounding_freedom_for_a_tie() -> None:
    case = strength_case(point_mpa=61.5, lower_mpa=54.0, upper_mpa=69.0)
    assert statuses("Die Druckfestigkeit beträgt 61,5 MPa.", case) == ["ok"]
    assert statuses("Die Druckfestigkeit beträgt 61 MPa.", case) == ["inexact"]
    assert statuses("Die Druckfestigkeit beträgt 62 MPa.", case) == ["inexact"]


def test_review_numeric_case_ids_mask_only_at_labelled_positions() -> None:
    amount = InspectionInput(case_id="999kg", mix=MIX, strength=strength())
    assert statuses("Die Zementmasse beträgt 999kg.", amount) == ["invented"]
    assert check_text("Fall 999kg: Bericht.", build_facts(amount)).faithful
    numeric = InspectionInput(case_id="123456", mix=MIX, strength=strength())
    sheet = build_facts(numeric)
    assert check_text("Für den Fall 123456 liegt ein Ergebnis vor.", sheet).faithful
    assert check_sections(template_draft(sheet).sections(), sheet).faithful
    assert statuses("Die Druckfestigkeit beträgt 123456 MPa.", numeric) == ["invented"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # invented values and codes
        ("Die Druckfestigkeit beträgt 65,3 MPa.", [("65,3 MPa", "invented")]),
        ("Festigkeitsklasse C50/60", [("50", "invented"), ("60", "invented")]),
        ("Expositionsklasse XC4", [("4", "invented")]),
        ("Es sind 3 Bohrkerne zu entnehmen.", [("3", "invented")]),
        ("Es sind drei Bohrkerne zu entnehmen.", [("drei", "invented")]),
        (
            "Prüfung nach dreihundertfünfundsechzig Tagen.",
            [("dreihundertfünfundsechzig Tagen", "invented")],
        ),
        ("Wiederholung nach zweitausend Zyklen.", [("zweitausend", "invented")]),
        ("CEM I 42,5 R", [("42,5", "invented")]),
        ("im Jahr 2026", [("2026", "invented")]),
        # rounding and format: no freedom at all
        ("Die Druckfestigkeit beträgt 62 MPa.", [("62 MPa", "inexact")]),
        ("Die Druckfestigkeit beträgt 61,90 MPa.", [("61,90 MPa", "inexact")]),
        ("Die Druckfestigkeit beträgt 61,87 MPa.", [("61,87 MPa", "inexact")]),
        ("Die Druckfestigkeit beträgt 61.9 MPa.", [("61.9 MPa", "inexact")]),
        ("Die Risswahrscheinlichkeit beträgt 93 %.", [("93 %", "inexact")]),
        ("Die Risswahrscheinlichkeit beträgt 0,931.", [("0,931", "invented")]),
        # units
        ("Die Druckfestigkeit beträgt 61,9.", [("61,9", "misattributed")]),
        ("Die Druckfestigkeit beträgt 28 MPa.", [("28 MPa", "misattributed")]),
        ("Das Betonalter beträgt 28 Jahre.", [("28 Jahre", "misattributed")]),
        ("Der Zement beträgt 540 kg.", [("540 kg", "misattributed")]),
        ("Der Zement beträgt 540 MPa.", [("540 MPa", "misattributed")]),
        ("Die Risswahrscheinlichkeit beträgt 93,1 MPa.", [("93,1 MPa", "misattributed")]),
        # roles
        ("Die untere Grenze liegt bei 69,6 MPa.", [("69,6 MPa", "misattributed")]),
        ("Die Entscheidungsschwelle liegt bei 93,1 %.", [("93,1 %", "misattributed")]),
        ("Die Überdeckung beträgt 50 %.", [("50 %", "misattributed")]),
        ("Der Wassergehalt beträgt 540 kg/m³.", [("540 kg/m³", "misattributed")]),
        ("Der Beitrag beträgt 61,9 MPa.", [("61,9 MPa", "misattributed")]),
        ("Ein Wert von 61,9 MPa wurde ermittelt.", [("61,9 MPa", "unbound")]),
        ("Ergebnis: 93,1 %.", [("93,1 %", "unbound")]),
        # signs
        ("Der Beitrag des Betonalters beträgt +3,1 MPa.", [("+3,1 MPa", "misattributed")]),
        ("Der Beitrag des Betonalters beträgt 3,1 MPa.", [("3,1 MPa", "misattributed")]),
        ("Die Druckfestigkeit beträgt \u221261,9 MPa.", [("\u221261,9 MPa", "misattributed")]),
        ("Die Abweichung beträgt -4 %.", [("-4 %", "invented")]),
        # numbers hidden in ranges
        ("Das Intervall reicht von 54,2 bis 71,0 MPa.", [("54,2", "ok"), ("71,0 MPa", "invented")]),
        ("Das Intervall reicht von 48,0 bis 69,6 MPa.", [("48,0", "invented"), ("69,6 MPa", "ok")]),
        (
            "Das Intervall reicht von 54,2 bis 61,9 MPa.",
            [("54,2", "ok"), ("61,9 MPa", "misattributed")],
        ),
        # dates, times, references
        ("Aufgenommen am 15.03.2026.", [("15.03.2026", "invented")]),
        ("Aufgenommen am 2026-03-14.", [("2026-03-14", "invented")]),
        ("Aufgenommen um 11:30 Uhr.", [("11:30", "invented")]),
        ("Prüfung nach DIN EN 12390-3:2019", [("DIN EN 12390-3:2019", "invented")]),
        ("Prüfung nach DIN 1048-5", [("DIN 1048-5", "invented")]),
        # identifiers are masked only as whole tokens
        ("Fall PB-022 und Fall PB-023", [("023", "invented")]),
    ],
)
def test_flags_violations(text: str, expected: list[tuple[str, str]]) -> None:
    assert verdicts(text) == expected


def test_duplicates_are_checked_one_by_one() -> None:
    text = "Die Druckfestigkeit beträgt 61,9 MPa, erneut 61,9 MPa; das Alter 61,9 Tage."
    assert statuses(text) == ["ok", "unbound", "misattributed"]


def test_negative_lower_bound_needs_its_sign() -> None:
    young = strength_case(point_mpa=10.2, lower_mpa=-1.9, upper_mpa=22.4)
    assert statuses("Das Intervall reicht von \u22121,9 MPa bis 22,4 MPa.", young) == ["ok", "ok"]
    assert statuses("Das Intervall reicht von -1,9 bis 22,4 MPa.", young) == ["ok", "ok"]
    assert statuses("Die untere Grenze liegt bei 1,9 MPa.", young) == ["misattributed"]


def test_range_dash_is_not_a_sign() -> None:
    assert statuses("Prognoseintervall 54,2\u201369,6 MPa") == ["ok", "ok"]
    assert statuses("Prognoseintervall 54,2 -69,6 MPa") == ["ok", "ok"]


def test_rendering_preserves_the_crack_decision() -> None:
    assert decision_preserving_percent(0.4999, 0.5) == (2, 0)
    assert decision_preserving_percent(0.50004, 0.5) == (3, 0)
    assert decision_preserving_percent(0.931, 0.5) == (1, 0)
    assert decision_preserving_percent(0.93, 0.5) == (0, 0)
    assert decision_preserving_percent(0.5, 0.5) == (0, 0)
    assert decision_preserving_percent(0.4261, 0.42615) == (1, 3)


def test_seconds_are_part_of_the_time() -> None:
    image = ImageMetadata(
        filename="IMG_0423.jpg",
        width_px=227,
        height_px=227,
        captured_at=datetime(2026, 3, 14, 10, 15, 59),
    )
    case = InspectionInput(case_id="PB-030", image=image, crack=crack())
    assert statuses("Aufgenommen um 10:15:59 Uhr.", case) == ["ok"]
    assert statuses("Aufgenommen um 10:15 Uhr.", case) == ["invented"]


def test_structured_result_records_positions_and_fields() -> None:
    text = "Die Druckfestigkeit beträgt 61,9 MPa"
    (check,) = check_text(text, BOTH, section="druckfestigkeit").checks
    assert text[check.start : check.end] == "61,9 MPa"
    assert check.matched_fields == ["strength.point_mpa"]
    assert check.unit == "MPa" and check.section == "druckfestigkeit"


def test_check_sections_merges_and_describes() -> None:
    result = check_sections(
        {
            "zusammenfassung": "Die Druckfestigkeit beträgt 61,9 MPa.",
            "empfehlung": "Prüfung in 14 Tagen.",
        },
        BOTH,
    )
    assert result.violation_counts() == {
        "invented": 1,
        "misattributed": 0,
        "inexact": 0,
        "unbound": 0,
    }
    (line,) = describe_violations(result)
    assert "14 Tagen" in line and "empfehlung" in line


def test_slot_mode_renders_numbers_and_rejects_literals() -> None:
    raw = {
        "zusammenfassung": "Die prognostizierte Druckfestigkeit beträgt {druckfestigkeit} MPa.",
        "rissbefund": "Die Risswahrscheinlichkeit beträgt {risswahrscheinlichkeit} %. "
        "Riss erkannt.",
    }
    assert render_slots(raw["zusammenfassung"], BOTH).endswith("61,9 MPa.")
    assert check_slot_sections(raw, BOTH).faithful  # type: ignore[arg-type]
    literal = {"zusammenfassung": "Die Druckfestigkeit beträgt 61,9 MPa und drei Proben."}
    statuses_ = [check.status for check in check_slot_sections(literal, BOTH).violations]  # type: ignore[arg-type]
    assert statuses_.count("invented") == 2


@pytest.mark.parametrize(
    ("text", "status"),
    [
        ("Die Druckfestigkeit beträgt {obere_grenze} MPa.", "misattributed"),
        ("Die Druckfestigkeit beträgt {druckfestigkeit} kg/m³.", "misattributed"),
        ("Die Druckfestigkeit beträgt {druckfestigkeit}.", "misattributed"),
        ("Ein Wert von {druckfestigkeit} MPa.", "unbound"),
        ("Die Druckfestigkeit beträgt {festigkeitsklasse}.", "invented"),
        ("Aufgenommen am 14.03.2026.", "invented"),
    ],
)
def test_slot_mode_role_checks_placeholders(text: str, status: str) -> None:
    result = check_slot_sections({"zusammenfassung": text}, BOTH)  # type: ignore[dict-item]
    assert [check.status for check in result.violations] == [status]


def test_template_uses_exact_forms_on_helper_cases() -> None:
    for case in (strength_case(), crack_case(0.2), both_case(in_training_domain=False)):
        sheet = build_facts(case)
        assert check_sections(template_draft(sheet).sections(), sheet).faithful


def test_identifier_parts_are_masked_only_as_tokens() -> None:
    assert check_text("Feld C2 der Decke über EG", BOTH).faithful
    image = IMAGE.model_copy(update={"element": "Stütze 12"})
    case = InspectionInput(case_id="PB-031", image=image, crack=crack())
    assert check_text("Bauteil Stütze 12", build_facts(case)).faithful
    assert statuses("Es wurden 12 Risse gefunden.", case) == ["invented"]
