"""Deterministic report text without a language model: the safe default and the baseline.

The template only inserts ``Fact.display`` strings, so it passes the faithfulness check by
construction; the tests assert that for every evaluation case.
"""

from baulens.report.facts import Fact, FactSheet
from baulens.report.schemas import InspectionReport, ReportDraft

TEMPLATE_GENERATOR = "template"
TEMPLATE_PROVENANCE = "Vorlagenbericht ohne Sprachmodell"
_BORDERLINE = 0.05


def _shown(sheet: FactSheet, field: str) -> str:
    fact: Fact | None = sheet.fact(field)
    if fact is None:
        raise KeyError(field)
    return fact.display


def _subject(sheet: FactSheet) -> str:
    case = sheet.case
    if case.image is not None and case.image.element:
        return f"das Bauteil „{case.image.element}“ (Fall {case.case_id})"
    return f"den Fall {case.case_id}"


def template_draft(sheet: FactSheet) -> ReportDraft:
    case = sheet.case
    strength, crack = case.strength, case.crack
    summary: list[str] = []
    strength_text: str | None = None
    crack_text: str | None = None
    limits: list[str] = []
    advice: list[str] = []

    if strength is not None:
        point = _shown(sheet, "strength.point_mpa")
        days = case.mix.age_days if case.mix is not None else 0
        age = f"{days} Tag" if days == 1 else f"{days} Tagen"
        lower = _shown(sheet, "strength.lower_mpa")
        upper = _shown(sheet, "strength.upper_mpa")
        coverage = _shown(sheet, "strength.interval_coverage")
        summary.append(
            f"Für {_subject(sheet)} schätzt das Modell bei einem Betonalter von {age} eine "
            f"Druckfestigkeit von {point}."
        )
        domain = (
            "Die Eingaben liegen innerhalb des Wertebereichs der Trainingsdaten."
            if strength.in_training_domain
            else "Mindestens eine Eingabe liegt außerhalb des Wertebereichs der Trainingsdaten; "
            "die Prognose ist daher besonders unsicher."
        )
        strength_text = (
            f"Für die angegebene Betonrezeptur mit {_shown(sheet, 'mix.cement')} Zement, "
            f"{_shown(sheet, 'mix.water')} Wasser (w/z-Wert "
            f"{_shown(sheet, 'mix.water_cement_ratio')}) und einem Betonalter von {age} "
            f"prognostiziert das Modell {strength.model_name} eine Druckfestigkeit von {point}. "
            f"Das Prognoseintervall mit einer nominellen Überdeckung von {coverage} reicht von "
            f"{lower} bis {upper}. {domain}"
        )
        limits.append(
            "Die Druckfestigkeit ist eine Modellschätzung und keine Messung; der tatsächliche "
            "Wert kann auch außerhalb des Prognoseintervalls liegen."
        )
        advice.append(
            "Die Druckfestigkeit sollte durch eine Prüfung an Probekörpern nach "
            "DIN EN 12390-3 oder an Bohrkernen nach DIN EN 12504-1 bestätigt werden."
        )
        if not strength.in_training_domain:
            limits.append(
                "Da die Eingaben außerhalb des Trainingsbereichs liegen, extrapoliert das Modell."
            )

    if crack is not None:
        probability = _shown(sheet, "crack.crack_probability")
        threshold = _shown(sheet, "crack.threshold")
        found = crack.label == "crack"
        verdict = "Riss erkannt" if found else "kein Riss erkannt"
        summary.append(
            f"Die automatische Bildauswertung ergibt bei einer Risswahrscheinlichkeit von "
            f"{probability}: {verdict}."
            if strength is not None
            else f"Für {_subject(sheet)} ergibt die automatische Bildauswertung bei einer "
            f"Risswahrscheinlichkeit von {probability}: {verdict}."
        )
        image = f"Das Bild {case.image.filename}" if case.image is not None else "Das Bild"
        crack_text = (
            f"{image} wurde vom Modell {crack.model_name} mit einer Risswahrscheinlichkeit von "
            f"{probability} bewertet; die Entscheidungsschwelle liegt bei {threshold}. "
            f"Ergebnis der Klassifikation: {verdict}."
        )
        if abs(crack.crack_probability - crack.threshold) < _BORDERLINE:
            crack_text += (
                " Die Risswahrscheinlichkeit liegt nahe an der Entscheidungsschwelle, das "
                "Ergebnis ist daher unsicher."
            )
        limits.append(
            "Die Rissklassifikation bewertet nur den abgebildeten Bildausschnitt und erlaubt "
            "keine Aussage über Rissbreite, Risstiefe oder Rissursache."
        )
        advice.append(
            "Der Befund sollte bei einer handnahen Sichtprüfung vor Ort durch eine sachkundige "
            "Person bestätigt und bewertet werden."
            if found
            else "Eine Sichtprüfung vor Ort wird empfohlen, da ein einzelner Bildausschnitt "
            "Risse an anderen Stellen nicht erfasst."
        )

    replaces = []
    if strength is not None:
        replaces.append("eine Druckfestigkeitsprüfung nach DIN EN 12390-3")
    if crack is not None:
        replaces.append("eine Sichtprüfung durch eine sachkundige Person vor Ort")
    if len(replaces) == 2:
        limits.append(f"Dieser Entwurf ersetzt weder {replaces[0]} noch {replaces[1]}.")
    else:
        limits.append(f"Dieser Entwurf ersetzt keine {replaces[0].removeprefix('eine ')}.")
    return ReportDraft(
        zusammenfassung=" ".join(summary),
        druckfestigkeit=strength_text,
        rissbefund=crack_text,
        einschraenkungen=" ".join(limits),
        empfehlung=" ".join(advice),
    )


def template_report(sheet: FactSheet) -> InspectionReport:
    return InspectionReport(
        case_id=sheet.case.case_id,
        draft=template_draft(sheet),
        generator=TEMPLATE_GENERATOR,
        provenance=TEMPLATE_PROVENANCE,
    )
