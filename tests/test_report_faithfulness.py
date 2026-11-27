"""Free-text mode: adversarial and regression tests of the numeric faithfulness checker.

Facts of ``both_case()``: strength 61,9 MPa with interval 54,2 to 69,6 MPa at 90 %,
cement 540 kg/m³, water 162 kg/m³, superplasticizer 2,5 kg/m³, coarse aggregate
1040 kg/m³, w/z 0,30, age 28 days, contributions +5,2 (cement) and \u22123,1 MPa (age), crack
probability 93,1 % at a threshold of 50 %, image 227 \u00d7 227 pixels captured on 14.03.2026 at
10:15, case PB-022, element "Decke über EG, Feld C2".
"""

from datetime import datetime
from decimal import Decimal

import pytest
from report_helpers import IMAGE, MIX, both_case, crack, crack_case, strength, strength_case

from baulens.report.facts import build_facts, decision_preserving_percent
from baulens.report.faithfulness import check_sections, check_text, describe_violations
from baulens.report.template import template_draft
from baulens.schemas import CrackPrediction, ImageMetadata, InspectionInput

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
        "Das Betonalter beträgt 28 Tage.",
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
        "Das Bild hat eine Breite von 227 Pixeln und eine Höhe von 227 Pixeln.",
        "Werte: 61,9 MPa (Prognosewert), 90 % (nominelle Überdeckung), 0,30 (w/z-Wert).",
        "Der Beitrag des Betonalters von 28 Tagen beträgt \u22123,1 MPa.",
        "Durch das Betonalter ergibt sich ein Rückgang von \u22123,1 MPa.",
        "Es wurde ein Riss erkannt; eine Sichtprüfung wird empfohlen.",
    ],
)
def test_accepts_exact_bound_numbers(text: str) -> None:
    result = check_text(text, BOTH)
    assert result.faithful, describe_violations(result)


# Every free-text example from the first review of PR #11 (blockers 1-3).
FIRST_REVIEW = [
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
# Free-text equivalents of the second review's examples (blockers 2-3, should-fixes).
SECOND_REVIEW = [
    ("Der Beitrag des Betonalters beträgt +5,2 MPa.", [("+5,2 MPa", "misattributed")]),
    ("Die Frist zur Behebung beträgt 28 Tage.", [("28 Tage", "unbound")]),
    ("Die Frist zur Behebung beträgt 28 Tage Betonalter.", [("28 Tage", "misattributed")]),
    ("Prüfdeadline ist der 14.03.2026.", [("14.03.2026", "misattributed")]),
    ("Die Risswahrscheinlichkeit beträgt 93,1 %-Punkte.", [("93,1 %-Punkte", "misattributed")]),
    ("Geprüft wurden Drei Bohrkerne.", [("Drei", "invented")]),
    ("Geprüft wurden ⁹ Bohrkerne.", [("⁹", "invented")]),
    ("Geprüft wurden ⑨ Bohrkerne.", [("⑨", "invented")]),
    ("9. Bohrkerne wurden entnommen.", [("9", "invented")]),
    ("Die Risswahrscheinlichkeit beträgt ein Drittel.", [("Drittel", "invented")]),
    ("Die Risswahrscheinlichkeit ist doppelt so hoch.", [("doppelt", "invented")]),
    ("Geprüft wurde ein Dutzend Bohrkerne.", [("Dutzend", "invented")]),
    ("Geprüft wurden XII Bohrkerne.", [("XII", "invented")]),
    ("Geprüft wurden Ⅻ Bohrkerne.", [("Ⅻ", "invented")]),
    ("Die Risswahrscheinlichkeit beträgt ein halbes Prozent.", [("halbes", "invented")]),
    ("Die Risswahrscheinlichkeit beträgt neunzig Prozent.", [("neunzig", "invented")]),
    ("Geprüft wurden ٩ Bohrkerne.", [("٩", "invented")]),
    ("Geprüft wurden \uff19 Bohrkerne.", [("\uff19", "invented")]),
    ("Die Festigkeit liegt über C30/37.", [("30", "invented"), ("37", "invented")]),
]


@pytest.mark.parametrize(("text", "expected"), FIRST_REVIEW + SECOND_REVIEW)
def test_review_examples_are_caught(text: str, expected: list[tuple[str, str]]) -> None:
    assert verdicts(text) == expected


def test_strength_class_in_an_element_name_excuses_only_that_name() -> None:
    image = IMAGE.model_copy(update={"element": "Wand C30/37"})
    case = InspectionInput(case_id="PB-033", image=image, crack=crack())
    sheet = build_facts(case)
    assert check_text("Das Bauteil Wand C30/37 wurde fotografiert.", sheet).faithful
    assert statuses("Die Festigkeit liegt über C30/37.", case) == ["invented", "invented"]


def test_review_rounding_against_threshold() -> None:
    case = crack_case(probability=0.4999, threshold=0.5)
    sheet = build_facts(case)
    probability = sheet.fact("crack.crack_probability")
    assert probability is not None and probability.display == "49,99 %"
    assert statuses("Die Risswahrscheinlichkeit beträgt 50 %.", case) == ["misattributed"]
    assert statuses("Die Risswahrscheinlichkeit beträgt 49,9 %.", case) == ["inexact"]
    assert statuses("Die Risswahrscheinlichkeit beträgt 49,99 %.", case) == ["ok"]
    assert "49,99 %" in template_draft(sheet).rissbefund  # type: ignore[operator]


def test_no_precision_cap_for_the_crack_decision() -> None:
    case = crack_case(probability=0.4999999999, threshold=0.5)
    sheet = build_facts(case)
    probability = sheet.fact("crack.crack_probability")
    assert probability is not None and probability.display == "49,99999999 %"
    assert "49,99999999 %" in template_draft(sheet).rissbefund  # type: ignore[operator]


@pytest.mark.parametrize("seed", range(40))
def test_rendered_comparison_equals_raw_comparison(seed: int) -> None:
    """Property-style: random probabilities around random thresholds keep their comparison."""
    import random

    rng = random.Random(seed)
    threshold = round(rng.uniform(0.05, 0.95), rng.randint(1, 8))
    probability = round(threshold + rng.choice([-1, 1]) * 10 ** -rng.randint(2, 12), 12)
    probability = min(max(probability, 0.0), 1.0)
    label = "crack" if probability >= threshold else "no_crack"
    prediction = CrackPrediction(
        crack_probability=probability, threshold=threshold, label=label,
        model_name="m", model_version="v1",
    )  # fmt: skip
    p_decimals, t_decimals = decision_preserving_percent(probability, threshold)
    sheet = build_facts(InspectionInput(case_id="PB-P", crack=prediction))
    shown_p = sheet.fact("crack.crack_probability")
    shown_t = sheet.fact("crack.threshold")
    assert shown_p is not None and shown_t is not None
    assert (shown_p.value >= shown_t.value) == (probability >= threshold)
    assert (shown_p.value == shown_t.value) == (
        Decimal(repr(probability)) == Decimal(repr(threshold))
    )
    assert shown_t.value == Decimal(repr(threshold)) * 100
    assert p_decimals >= 0 and t_decimals >= 0


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
        ("Die Druckfestigkeit beträgt 65,3 MPa.", [("65,3 MPa", "invented")]),
        ("Expositionsklasse XC4", [("4", "invented")]),
        ("Es sind 3 Bohrkerne zu entnehmen.", [("3", "invented")]),
        ("Es sind drei Bohrkerne zu entnehmen.", [("drei", "invented")]),
        ("Das Betonalter beträgt achtundzwanzig Tage.", [("achtundzwanzig", "invented")]),
        ("Die zweite Prüfung folgt.", [("zweite", "invented")]),
        ("CEM I 42,5 R", [("I", "invented"), ("42,5", "invented")]),
        ("im Jahr 2026", [("2026", "invented")]),
        ("1. Zusammenfassung folgt.", [("1", "invented")]),
        ("Die Druckfestigkeit beträgt 62 MPa.", [("62 MPa", "inexact")]),
        ("Die Druckfestigkeit beträgt 61,90 MPa.", [("61,90 MPa", "inexact")]),
        ("Die Druckfestigkeit beträgt 61.9 MPa.", [("61.9 MPa", "inexact")]),
        ("Die Risswahrscheinlichkeit beträgt 93 %.", [("93 %", "inexact")]),
        ("Die Druckfestigkeit beträgt 61,9.", [("61,9", "misattributed")]),
        ("Die Druckfestigkeit beträgt 28 MPa.", [("28 MPa", "misattributed")]),
        ("Das Betonalter beträgt 28 Jahre.", [("28 Jahre", "misattributed")]),
        ("Der Zement beträgt 540 kg.", [("540 kg", "misattributed")]),
        ("Die untere Grenze liegt bei 69,6 MPa.", [("69,6 MPa", "misattributed")]),
        ("Die Entscheidungsschwelle liegt bei 93,1 %.", [("93,1 %", "misattributed")]),
        ("Der Wassergehalt beträgt 540 kg/m³.", [("540 kg/m³", "misattributed")]),
        ("Ein Wert von 61,9 MPa wurde ermittelt.", [("61,9 MPa", "unbound")]),
        ("Der Beitrag des Betonalters beträgt +3,1 MPa.", [("+3,1 MPa", "misattributed")]),
        ("Die Druckfestigkeit beträgt \u221261,9 MPa.", [("\u221261,9 MPa", "misattributed")]),
        ("Das Intervall reicht von 54,2 bis 71,0 MPa.", [("54,2", "ok"), ("71,0 MPa", "invented")]),
        ("Aufgenommen am 15.03.2026.", [("15.03.2026", "invented")]),
        ("Prüfung nach DIN EN 12390-3:2019", [("DIN EN 12390-3:2019", "invented")]),
        ("Fall PB-022 und Fall PB-023", [("023", "invented")]),
        ("Feld C2 der Decke", [("2", "invented")]),
    ],
)
def test_flags_violations(text: str, expected: list[tuple[str, str]]) -> None:
    assert verdicts(text) == expected


def test_negative_lower_bound_needs_its_sign() -> None:
    young = strength_case(point_mpa=10.2, lower_mpa=-1.9, upper_mpa=22.4)
    assert statuses("Das Intervall reicht von \u22121,9 MPa bis 22,4 MPa.", young) == ["ok", "ok"]
    assert statuses("Die untere Grenze liegt bei 1,9 MPa.", young) == ["misattributed"]


def test_seconds_are_part_of_the_time() -> None:
    image = ImageMetadata(
        filename="IMG_0423.jpg", width_px=227, height_px=227,
        captured_at=datetime(2026, 3, 14, 10, 15, 59),
    )  # fmt: skip
    case = InspectionInput(case_id="PB-030", image=image, crack=crack())
    assert statuses("Aufgenommen um 10:15:59 Uhr.", case) == ["ok"]
    assert statuses("Aufgenommen um 10:15 Uhr.", case) == ["invented"]


def test_structured_result_records_positions_and_fields() -> None:
    text = "Die Druckfestigkeit beträgt 61,9 MPa"
    (check,) = check_text(text, BOTH, section="druckfestigkeit").checks
    assert text[check.start : check.end] == "61,9 MPa"
    assert check.matched_fields == ["strength.point_mpa"]


def test_check_sections_merges_and_describes() -> None:
    result = check_sections(
        {"zusammenfassung": "Die Druckfestigkeit beträgt 61,9 MPa.", "empfehlung": "In 14 Tagen."},
        BOTH,
    )
    assert result.violation_counts()["invented"] == 1
    (line,) = describe_violations(result)
    assert "14 Tagen" in line and "empfehlung" in line


def test_template_is_faithful_on_helper_cases() -> None:
    for case in (strength_case(), crack_case(0.2), both_case(in_training_domain=False)):
        sheet = build_facts(case)
        assert check_sections(template_draft(sheet).sections(), sheet).faithful
