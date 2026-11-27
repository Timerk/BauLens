"""Completeness: does a report state the facts and caveats that a reviewer needs?

Required items depend on the case. A fact counts as stated when the faithfulness check
matched at least one token to it; caveats are found with keyword patterns. The patterns are
deliberately simple and listed here so that a reviewer can judge what "mentioned" means.
"""

import re

from baulens.report.facts import FactSheet
from baulens.report.schemas import (
    DISCLAIMER,
    CompletenessItem,
    CompletenessResult,
    FaithfulnessResult,
    InspectionReport,
)

_OUT_OF_DOMAIN = re.compile(
    r"au(ß|ss)erhalb\s+(des|der)\s+(\w+\s+){0,3}(Trainings|Wertebereich|Datenbereich|Daten)"
    r"|Trainingsbereich\w*\s+(\w+\s+){0,3}(verlassen|überschritten)|extrapol",
    re.IGNORECASE,
)
_HEDGE = re.compile(
    r"\b(wenn|falls|sofern|soweit|möglicherweise|könnte|könnten|kann|können|nicht)\b",
    re.IGNORECASE,
)
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_ESTIMATE = re.compile(r"Schätzung|geschätzt|Prognose|prognostiziert|Modellwert", re.IGNORECASE)
_TESTING = re.compile(
    r"12390-3|Druckfestigkeitsprüfung|Probekörper|Prüfkörper|Bohrkern", re.IGNORECASE
)
_PATCH = re.compile(r"Bildausschnitt|Ausschnitt|Einzelbild|Patch|Bildbereich", re.IGNORECASE)
_VISUAL = re.compile(
    r"Sichtprüfung|Inaugenscheinnahme|Begehung|vor Ort|handnah|Bauwerksprüfung", re.IGNORECASE
)
_NO_CRACK = re.compile(
    r"\bkein(e|en|em)?\s+(\w+\s+){0,2}Riss|nicht\s+als\s+Riss|rissfrei|ohne\s+(\w+\s+)?Riss"
    r"|keine\s+Rissbildung|kein\w*\s+Hinweis\w*\s+auf\s+(\w+\s+){0,2}Riss",
    re.IGNORECASE,
)
_CRACK = re.compile(
    r"Riss\w*\s+(wurde\s+|wird\s+)?(erkannt|festgestellt|detektiert|nachgewiesen|identifiziert"
    r"|vorhanden|angezeigt)|als\s+Riss\s+klassifiziert|als\s+rissbehaftet"
    r"|weist\s+(\w+\s+){0,4}auf\s+(einen\s+)?Riss|Ergebnis\w*\s*(der\s+Klassifikation\s*)?:?\s*Riss",
    re.IGNORECASE,
)
"""An explicit positive finding; a crack section that only lists numbers does not state one."""


def crack_finding_consistent(finding: str, label: str) -> bool:
    """The crack section states the finding explicitly and in line with the label."""
    negated = bool(_NO_CRACK.search(finding))
    if label == "no_crack":
        return negated
    return bool(_CRACK.search(finding)) and not negated


def _item(name: str, satisfied: bool, detail: str) -> CompletenessItem:
    return CompletenessItem(name=name, satisfied=satisfied, detail=detail)


def check_completeness(
    report: InspectionReport, faithfulness: FaithfulnessResult, sheet: FactSheet
) -> CompletenessResult:
    """Score which required facts and caveats a report contains."""
    case = sheet.case
    sections = report.draft.sections()
    text = "\n".join(sections.values())
    stated = faithfulness.matched_fields()
    items: list[CompletenessItem] = []

    def fact_item(name: str, *fields: str) -> None:
        missing = [field for field in fields if field not in stated]
        detail = "genannt" if not missing else "fehlt: " + ", ".join(missing)
        items.append(_item(name, not missing, detail))

    if case.strength is not None:
        fact_item("strength_point", "strength.point_mpa")
        fact_item("interval_bounds", "strength.lower_mpa", "strength.upper_mpa")
        fact_item("interval_coverage", "strength.interval_coverage")
        fact_item("age", "mix.age_days")
        items.append(
            _item("estimate_caveat", bool(_ESTIMATE.search(text)), "Hinweis auf Modellschätzung")
        )
        items.append(_item("testing_reference", bool(_TESTING.search(text)), "Verweis auf Prüfung"))
        if not case.strength.in_training_domain:
            items.append(
                _item(
                    "out_of_domain_warning",
                    bool(_OUT_OF_DOMAIN.search(text)),
                    "Warnung: außerhalb des Trainingsbereichs",
                )
            )
        else:
            # Consistency, not presence: an in-domain case must not be declared out of domain.
            # Conditional or negated sentences ("wenn Eingaben außerhalb ... liegen") pass.
            false_claims = [
                sentence
                for sentence in _SENTENCE.split(text)
                if _OUT_OF_DOMAIN.search(sentence) and not _HEDGE.search(sentence)
            ]
            items.append(
                _item(
                    "no_false_domain_warning",
                    not false_claims,
                    false_claims[0] if false_claims else "keine falsche Bereichswarnung",
                )
            )
    if case.crack is not None:
        fact_item("crack_probability", "crack.crack_probability")
        fact_item("crack_threshold", "crack.threshold")
        consistent = crack_finding_consistent(sections.get("rissbefund", ""), case.crack.label)
        items.append(_item("crack_label", consistent, f"Klassifikation {case.crack.label}"))
        items.append(_item("patch_caveat", bool(_PATCH.search(text)), "nur Bildausschnitt"))
        items.append(_item("visual_inspection", bool(_VISUAL.search(text)), "Sichtprüfung vor Ort"))
    items.append(_item("disclaimer", report.disclaimer == DISCLAIMER, "Prüfvermerk eingefügt"))
    return CompletenessResult(items=items)
