"""Strict slot mode: grammar, code-owned quantities, zero numerals, role binding, gates."""

import pytest
from report_helpers import IMAGE, both_case, crack, crack_case, strength_case

from baulens.report.facts import build_facts
from baulens.report.numerals import find_numerals, is_number_word
from baulens.report.slots import check_slot_sections, check_slot_text, render_slots
from baulens.schemas import InspectionInput

BOTH = build_facts(both_case())


def slot_statuses(text: str, case: InspectionInput | None = None) -> list[str]:
    sheet = build_facts(case) if case is not None else BOTH
    return [check.status for check in check_slot_text(text, sheet).violations]


def test_slots_render_complete_quantities() -> None:
    slots = BOTH.slots()
    assert slots["druckfestigkeit"] == "61,9 MPa"
    assert slots["beitrag_betonalter"] == "\u22123,1 MPa"
    assert slots["beitrag_zement"] == "+5,2 MPa"
    assert slots["risswahrscheinlichkeit"] == "93,1 %"
    assert slots["betonalter"] == "28 Tage"
    assert slots["aufnahmezeit"] == "10:15 Uhr"
    assert slots["norm_druckfestigkeit"] == "DIN EN 12390-3"
    rendered = render_slots("Die Druckfestigkeit beträgt **{druckfestigkeit}**.", BOTH)
    assert rendered == "Die Druckfestigkeit beträgt 61,9 MPa."


@pytest.mark.parametrize(
    "text",
    [
        "Die prognostizierte Druckfestigkeit beträgt {druckfestigkeit}.",
        "Das Prognoseintervall reicht von {untere_grenze} bis {obere_grenze}.",
        "Die Druckfestigkeit liegt zwischen {untere_grenze} und {obere_grenze}.",
        "Die untere Grenze liegt bei {untere_grenze}, die obere Grenze bei {obere_grenze}.",
        "Die nominelle Überdeckung des Prognoseintervalls beträgt {ueberdeckung}.",
        "Die Risswahrscheinlichkeit von {risswahrscheinlichkeit} liegt über der "
        "Entscheidungsschwelle von {entscheidungsschwelle}.",
        "Das Betonalter beträgt {betonalter}.",
        "Die Rezeptur enthält {zement} Zement und {wasser} Wasser.",
        "Der w/z-Wert beträgt {wz_wert}.",
        "Der Beitrag des Zements beträgt {beitrag_zement}, der des Betonalters "
        "{beitrag_betonalter}.",
        "Das Bild {bild} wurde am {aufnahmedatum} um {aufnahmezeit} aufgenommen.",
        "Die Prognose des Modells {festigkeitsmodell} ersetzt keine Prüfung nach "
        "{norm_druckfestigkeit}.",
        "Für den Fall {fall_id} am Bauteil {bauteil} wurde ein Riss erkannt.",
        "Die prognostizierte Druckfestigkeit beträgt **{druckfestigkeit}**.",
        "Die prognostizierte Druckfestigkeit beträgt {druckfestigkeit}; sie ist eine Schätzung.",
        "Die Prognose ist eine Modellschätzung und ersetzt eine Prüfung nicht.",
        "Massen sind je m³ Beton angegeben.",
    ],
)
def test_accepts_well_formed_slot_text(text: str) -> None:
    result = check_slot_text(text, BOTH)
    assert result.faithful, [(check.text, check.reason) for check in result.violations]


# Every slot example from the second review of PR #11, plus the grammar cases.
SECOND_REVIEW = [
    ("Die Druckfestigkeit beträgt -{druckfestigkeit} MPa.", "misattributed"),
    ("Die Druckfestigkeit beträgt +{druckfestigkeit} kg.", "misattributed"),
    ("Die Druckfestigkeit beträgt {druckfestigkeit} MPa.", "misattributed"),
    ("Der Beitrag des Betonalters beträgt {beitrag_zement}.", "misattributed"),
    ("Die Frist zur Behebung beträgt {betonalter}.", "misattributed"),
    ("Prüfdeadline ist der {aufnahmedatum}.", "misattributed"),
    ("Die Risswahrscheinlichkeit beträgt {risswahrscheinlichkeit} %-Punkte.", "misattributed"),
    ("Die Risswahrscheinlichkeit beträgt {risswahrscheinlichkeit}-Punkte.", "misattributed"),
    ("Die Druckfestigkeit beträgt nicht {druckfestigkeit}.", "misattributed"),
    ("Geprüft wurden Drei Bohrkerne.", "invented"),
    ("Geprüft wurden ⁹ Bohrkerne.", "invented"),
    ("Geprüft wurden ⑨ Bohrkerne.", "invented"),
    ("9. Bohrkerne wurden entnommen.", "invented"),
    ("Die Festigkeit liegt über C30/37.", "invented"),
    ("Die Risswahrscheinlichkeit beträgt ein Drittel.", "invented"),
    ("Die Risswahrscheinlichkeit ist doppelt so hoch.", "invented"),
    ("Geprüft wurde ein Dutzend Bohrkerne.", "invented"),
    ("Geprüft wurden XII Bohrkerne.", "invented"),
    ("Geprüft wurden Ⅻ Bohrkerne.", "invented"),
    ("Die Risswahrscheinlichkeit beträgt ein halbes Prozent.", "invented"),
    ("Die Prüfung erfolgt nach DIN EN 12390-3.", "invented"),
    ("Die Druckfestigkeit beträgt {druckfestigkeit_mpa}.", "invented"),
    ("Die Druckfestigkeit beträgt {Druckfestigkeit}.", "invented"),
    ("Die Druckfestigkeit beträgt {druckfestigkeit:wert}.", "invented"),
    ("Die Druckfestigkeit beträgt \\{druckfestigkeit\\}.", "invented"),
    ("Die Druckfestigkeit beträgt \\{druckfestigkeit}.", "invented"),
    ("Die Druckfestigkeit beträgt {obere_grenze}.", "misattributed"),
    ("Die Risswahrscheinlichkeit beträgt {entscheidungsschwelle}.", "misattributed"),
    ("Das Intervall reicht von {obere_grenze} bis {untere_grenze}.", "misattributed"),
    ("Der Wert beträgt {druckfestigkeit}.", "unbound"),
]


@pytest.mark.parametrize(("text", "status"), SECOND_REVIEW)
def test_review_slot_examples_are_rejected(text: str, status: str) -> None:
    found = slot_statuses(text)
    assert found and found[0] == status, found


def test_double_braces_are_rejected() -> None:
    assert "invented" in slot_statuses("Die Druckfestigkeit beträgt {{druckfestigkeit}}.")


def test_repeated_slots_are_allowed_across_sections() -> None:
    raw = {
        "zusammenfassung": "Die prognostizierte Druckfestigkeit beträgt {druckfestigkeit}.",
        "druckfestigkeit": "Die Druckfestigkeit beträgt {druckfestigkeit}; das Prognoseintervall "
        "reicht von {untere_grenze} bis {obere_grenze}.",
    }
    assert check_slot_sections(raw, build_facts(strength_case())).faithful  # type: ignore[arg-type]


def test_mandatory_facts_gate_slot_delivery() -> None:
    sheet = build_facts(strength_case())
    raw = {"druckfestigkeit": "Die Druckfestigkeit wird im Labor geprüft."}
    missing = [c.text for c in check_slot_sections(raw, sheet).violations]  # type: ignore[arg-type]
    assert missing == ["{druckfestigkeit}", "{untere_grenze}", "{obere_grenze}"]
    crack_sheet = build_facts(crack_case())
    raw = {"rissbefund": "Die Risswahrscheinlichkeit beträgt {risswahrscheinlichkeit}."}
    result = check_slot_sections(raw, crack_sheet)  # type: ignore[arg-type]
    assert [c.status for c in result.violations] == ["missing"]


def test_identifiers_with_strength_classes_only_via_their_slot() -> None:
    image = IMAGE.model_copy(update={"element": "Wand C30/37"})
    case = InspectionInput(case_id="PB-033", image=image, crack=crack())
    assert check_slot_text("Das Bauteil {bauteil} wurde fotografiert.", build_facts(case)).faithful
    assert slot_statuses("Die Festigkeit liegt über C30/37.", case) == ["invented", "invented"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("drei Drei DREI", ["drei", "Drei", "DREI"]),
        (
            "die erste, zweiten, achtundzwanzigste und zweitens",
            ["erste", "zweiten", "achtundzwanzigste", "zweitens"],
        ),
        ("halb, Hälfte, Drittel, anderthalb", ["halb", "Hälfte", "Drittel", "anderthalb"]),
        ("doppelt, dreifach, zweimal", ["doppelt", "dreifach", "zweimal"]),
        ("ein Dutzend, ein Paar, Hunderte", ["Dutzend", "Paar", "Hunderte"]),
        ("ein, eine, einem, einmal, mehrfach, ein paar", []),
        (
            "II, XII, CEM I, Ⅻ, ⑨, ⁹, ½, ٩, \uff19",
            ["II", "XII", "I", "Ⅻ", "⑨", "⁹", "½", "٩", "\uff19"],
        ),
        ("kg/m³ und mm²", []),
    ],
)
def test_numeral_detector(text: str, expected: list[str]) -> None:
    assert [numeral.text for numeral in find_numerals(text)] == expected


def test_articles_are_not_numerals_but_eins_is() -> None:
    assert not is_number_word("eine") and is_number_word("eins")
