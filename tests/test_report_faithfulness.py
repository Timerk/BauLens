"""Adversarial tests of the numeric faithfulness checker.

Facts of ``both_case()``: strength 61,9 MPa (54,2 to 69,6 MPa, 90 %), cement 540 kg/m³,
water 162 kg/m³, superplasticizer 2,5 kg/m³, coarse aggregate 1040 kg/m³, w/z 0,30, age
28 days, contributions +5,2 and -3,1 MPa, crack probability 93,1 % at a threshold of 50 %,
image 227 x 227 pixels captured on 14.03.2026 at 10:15.
"""

import pytest
from report_helpers import both_case, crack_case, strength_case

from baulens.report.facts import build_facts
from baulens.report.faithfulness import check_sections, check_text, describe_violations
from baulens.schemas import InspectionInput

BOTH = build_facts(both_case())


def verdicts(text: str, case: InspectionInput | None = None) -> list[tuple[str, str]]:
    sheet = build_facts(case) if case is not None else BOTH
    return [(check.text, check.status) for check in check_text(text, sheet).checks]


def statuses(text: str, case: InspectionInput | None = None) -> list[str]:
    return [status for _, status in verdicts(text, case)]


@pytest.mark.parametrize(
    "text",
    [
        "Die Druckfestigkeit beträgt 61,9 MPa.",
        "Die Druckfestigkeit beträgt 61.9 MPa.",
        "Die Druckfestigkeit beträgt rund 62 MPa.",
        "Die Druckfestigkeit beträgt 61,9 N/mm².",
        "Das Prognoseintervall reicht von 54,2 bis 69,6 MPa.",
        "Das Prognoseintervall reicht von 54,2\u201369,6 MPa.",
        "Das Prognoseintervall reicht von 54,2 - 69,6 MPa.",
        "zwischen 54 und 70 MPa",
        "Das 90-%-Prognoseintervall",
        "eine nominelle Überdeckung von 90 %",
        "eine nominelle Überdeckung von 0,90",
        "Zement 540 kg/m³, Wasser 162 kg/m3, Fließmittel 2,5 kg",
        "grobe Gesteinskörnung 1040 kg/m³",
        "grobe Gesteinskörnung 1.040 kg/m³",
        "grobe Gesteinskörnung 1 040 kg/m³",
        "grobe Gesteinskörnung 1\u00a0040 kg/m³",
        "w/z-Wert 0,30 bzw. 0,3",
        "Betonalter 28 Tage, also die 28-Tage-Festigkeit nach 28 d (28d)",
        "nach achtundzwanzig Tagen",
        "Die Risswahrscheinlichkeit beträgt 93,1 % (0,931), gerundet 93 %.",
        "Die Entscheidungsschwelle liegt bei 50 % bzw. 0,5.",
        "Beitrag des Betonalters: \u22123,1 MPa; es verringert die Festigkeit um 3,1 MPa.",
        "Der Zement erhöht die Festigkeit um +5,2 MPa.",
        "Bild 227 \u00d7 227 Pixel, aufgenommen am 14.03.2026 um 10:15 Uhr.",
        "aufgenommen am 14. März 2026 (2026-03-14)",
        "Prüfung nach DIN EN 12390-3 oder an Bohrkernen nach DIN EN 12504-1.",
        "Fall PB-022, Bauteil Decke über EG, Feld C2, Bild IMG_0423.jpg, Modell resnet18.",
        "Feld C2 der Decke",
        "1. Zusammenfassung folgt.",
    ],
)
def test_accepts_permitted_renderings(text: str) -> None:
    result = check_text(text, BOTH)
    assert result.faithful, describe_violations(result)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # hallucinated values
        ("Die Druckfestigkeit beträgt 65,3 MPa.", [("65,3 MPa", "invented")]),
        ("Festigkeitsklasse C50/60", [("50", "invented"), ("60", "invented")]),
        ("Es sind 3 Bohrkerne zu entnehmen.", [("3", "invented")]),
        ("Es sind drei Bohrkerne zu entnehmen.", [("drei", "invented")]),
        ("Wiederholung in 6 Monaten", [("6 Monaten", "invented")]),
        ("CEM I 42,5 R", [("42,5", "invented")]),
        # rounding tricks
        ("Die Druckfestigkeit beträgt 61 MPa.", [("61 MPa", "bad_rounding")]),
        ("Die Druckfestigkeit beträgt 61,87 MPa.", [("61,87 MPa", "bad_rounding")]),
        ("Die Druckfestigkeit beträgt 62,0 MPa.", [("62,0 MPa", "bad_rounding")]),
        ("Die Druckfestigkeit beträgt etwa 60 MPa.", [("60 MPa", "invented")]),
        ("Die Risswahrscheinlichkeit beträgt 0,9.", [("0,9", "bad_rounding")]),
        # wrong unit or quantity
        ("Die Druckfestigkeit beträgt 28 MPa.", [("28 MPa", "misattributed")]),
        ("Das Betonalter beträgt 61,9 Tage.", [("61,9 Tage", "misattributed")]),
        ("Zementgehalt 540 MPa", [("540 MPa", "misattributed")]),
        ("Risswahrscheinlichkeit 0,931 %", [("0,931 %", "misattributed")]),
        ("Die Risswahrscheinlichkeit beträgt 90 %.", [("90 %", "misattributed")]),
        ("Die Entscheidungsschwelle liegt bei 93,1 %.", [("93,1 %", "misattributed")]),
        # wrong sign
        ("Das Alter wirkt mit +3,1 MPa.", [("+3,1 MPa", "misattributed")]),
        ("Der Zement wirkt mit \u22125,2 MPa.", [("\u22125,2 MPa", "misattributed")]),
        # numbers hidden in ranges
        ("Intervall 54,2\u201371,0 MPa", [("54,2", "ok"), ("71,0 MPa", "invented")]),
        ("Intervall 48,0 bis 69,6 MPa", [("48,0", "invented"), ("69,6 MPa", "ok")]),
        ("Intervall 50,0 bis 69,6 MPa", [("50,0", "misattributed"), ("69,6 MPa", "ok")]),
        ("Expositionsklasse XC4", [("4", "invented")]),
        ("Intervall 28\u201369,6 MPa", [("28", "misattributed"), ("69,6 MPa", "ok")]),
        # thousands separators that do not resolve to a fact
        ("grobe Gesteinskörnung 1.050 kg/m³", [("1.050 kg/m³", "invented")]),
        ("1,040,000 kg/m³", [("1,040,000 kg/m³", "invented")]),
        # negative numbers
        ("Die Druckfestigkeit beträgt \u221261,9 MPa.", [("\u221261,9 MPa", "misattributed")]),
        ("eine Abweichung von -4 %", [("-4 %", "invented")]),
        # dates, times and references
        ("aufgenommen am 15.03.2026", [("15.03.2026", "invented")]),
        ("aufgenommen am 31.02.2026", [("31.02.2026", "invented")]),
        ("um 11:30 Uhr", [("11:30", "invented")]),
        ("Prüfung nach DIN EN 12390-3:2019", [("DIN EN 12390-3:2019", "invented")]),
        ("Prüfung nach DIN 1048-5", [("DIN 1048-5", "invented")]),
        ("im Jahr 2026", [("2026", "invented")]),
    ],
)
def test_flags_violations(text: str, expected: list[tuple[str, str]]) -> None:
    assert verdicts(text) == expected


def test_duplicates_are_checked_one_by_one() -> None:
    text = "61,9 MPa, erneut 61,9 MPa und noch einmal 61,9 Tage."
    assert statuses(text) == ["ok", "ok", "misattributed"]


def test_rounding_must_not_cross_the_threshold() -> None:
    borderline = crack_case(probability=0.497, threshold=0.5)
    assert statuses("Risswahrscheinlichkeit 49,7 %", borderline) == ["ok"]
    result = check_text("Risswahrscheinlichkeit 50 %", build_facts(borderline))
    assert [check.status for check in result.checks] == ["bad_rounding"]
    assert "Entscheidungsschwelle" in result.checks[0].reason
    above = crack_case(probability=0.504, threshold=0.5)
    assert statuses("Risswahrscheinlichkeit 50 %", above) == ["ok"]


def test_negative_lower_bound_requires_its_sign() -> None:
    young = strength_case(point_mpa=10.2, lower_mpa=-1.9, upper_mpa=22.4)
    assert statuses("Intervall von \u22121,9 MPa bis 22,4 MPa", young) == ["ok", "ok"]
    assert statuses("Intervall von -1,9 bis 22,4 MPa", young) == ["ok", "ok"]
    assert statuses("untere Grenze 1,9 MPa", young) == ["misattributed"]


def test_range_dash_is_not_a_sign() -> None:
    assert statuses("54,2\u201369,6 MPa") == ["ok", "ok"]
    assert statuses("54,2 -69,6 MPa") == ["ok", "ok"]


def test_bare_digits_only_match_exactly() -> None:
    # 2,5 kg/m³ rounds to 3 kg/m³, but a bare 3 must not pass as that rounding.
    assert statuses("3 kg/m³ Fließmittel") == ["ok"]
    assert statuses("3 Proben") == ["invented"]


def test_context_attribution_only_restricts_percent_quantities() -> None:
    case = strength_case()
    # No crack facts: "Risswahrscheinlichkeit" cannot misattribute anything here.
    assert statuses("Risswahrscheinlichkeit 90 %", case) == ["ok"]
    assert statuses("Schwellenwert 50 %, Risswahrscheinlichkeit 93,1 %") == ["ok", "ok"]


def test_identifiers_are_masked_only_verbatim() -> None:
    result = check_text("Fall PB-022 und Fall PB-023", BOTH)
    assert [check.text for check in result.violations] == ["023"]
    assert [span.text for span in result.masked] == ["PB-022"]


def test_letterless_versions_are_masked_only_after_version() -> None:
    case = InspectionInput.model_validate(
        {
            **crack_case().model_dump(),
            "crack": {**crack_case().crack.model_dump(), "model_version": "1.2"},
        }  # type: ignore[union-attr]
    )
    sheet = build_facts(case)
    assert check_text("Modell resnet18, Version 1.2", sheet).faithful
    assert not check_text("ein Anteil von 1.2", sheet).faithful


def test_structured_result_records_positions_and_fields() -> None:
    text = "Druckfestigkeit 61,9 MPa"
    (check,) = check_text(text, BOTH, section="druckfestigkeit").checks
    assert text[check.start : check.end] == "61,9 MPa"
    assert check.matched_fields == ["strength.point_mpa"]
    assert check.unit == "MPa"
    assert check.section == "druckfestigkeit"


def test_check_sections_merges_and_describes() -> None:
    result = check_sections(
        {"zusammenfassung": "61,9 MPa", "empfehlung": "Prüfung in 14 Tagen"}, BOTH
    )
    assert result.violation_counts() == {"invented": 1, "misattributed": 0, "bad_rounding": 0}
    (line,) = describe_violations(result)
    assert "14 Tagen" in line and "empfehlung" in line
