"""Citable facts of an inspection case, rendered once for the prompt, the slots and the checker.

Every number a report may contain comes from a :class:`Fact` and is accepted only in exactly
the form shown to the model (``Fact.number``, German format, ROUND_HALF_UP). There is no
rounding freedom. The only equivalent spellings are typographic: ``\u2212`` or ``-`` as minus and
``1040``, ``1.040`` or ``1 040`` for the same digits.

Precision is chosen so that rendering cannot change a decision the input encodes: the crack
probability gets as many decimals as needed to stay on the same side of the threshold and to
differ from it (0.4999 against 0.5 renders as ``49,99 %`` against ``50 %``).

Each fact also carries a *role* (what the number is, e.g. ``point`` or ``lower``) used by the
checker to bind a number to the words around it, and a *slot* name used by the slot mode, in
which the model writes ``{druckfestigkeit}`` and code inserts ``61,9 MPa``.
"""

import re
from dataclasses import dataclass, replace
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal
from pathlib import PurePath
from typing import Literal

from baulens.schemas import (
    ConcreteMix,
    CrackPrediction,
    ImageMetadata,
    InspectionInput,
    StrengthPrediction,
)

UnitKind = Literal["MPa", "kg/m³", "d", "%", "px"]

MINUS = "\u2212"
STRENGTH_DECIMALS = 1
MAX_INGREDIENT_DECIMALS = 2
RATIO_DECIMALS = 2

MIX_LABELS: dict[str, str] = {
    "cement": "Zement",
    "blast_furnace_slag": "Hüttensand",
    "fly_ash": "Flugasche",
    "water": "Wasser",
    "superplasticizer": "Fließmittel",
    "coarse_aggregate": "Grobe Gesteinskörnung",
    "fine_aggregate": "Feine Gesteinskörnung",
}
MIX_SLOTS: dict[str, str] = {
    "cement": "zement",
    "blast_furnace_slag": "huettensand",
    "fly_ash": "flugasche",
    "water": "wasser",
    "superplasticizer": "fliessmittel",
    "coarse_aggregate": "grobe_gesteinskoernung",
    "fine_aggregate": "feine_gesteinskoernung",
}
_FEATURE_LABELS: dict[str, str] = {
    **MIX_LABELS,
    "age_days": "Betonalter",
    "age": "Betonalter",
    "water_cement_ratio": "w/z-Wert",
    "w_c_ratio": "w/z-Wert",
    "water_binder_ratio": "w/b-Wert",
    "w_b_ratio": "w/b-Wert",
}
_RATIO_FEATURES = {"water_cement_ratio", "w_c_ratio"}
FEATURE_TERMS: dict[str, str] = {
    "cement": r"zement(?!wert)",
    "blast_furnace_slag": r"hüttensand|schlacke",
    "fly_ash": r"flugasche",
    "water": r"wasser(?!zement|-zement)",
    "superplasticizer": r"fließmittel|fliessmittel",
    "coarse_aggregate": r"grob|kies",
    "fine_aggregate": r"fein|sand",
    "age_days": r"alter",
    "age": r"alter",
    "water_cement_ratio": r"w/z|wasserzement|wasser-zement",
    "w_c_ratio": r"w/z|wasserzement|wasser-zement",
    "water_binder_ratio": r"w/b|wasserbindemittel|wasser-bindemittel",
    "w_b_ratio": r"w/b|wasserbindemittel|wasser-bindemittel",
}
_AGE_FEATURES = {"age", "age_days"}


@dataclass(frozen=True)
class Fact:
    """A citable number, addressed by its dotted field path in the ``InspectionInput``."""

    field: str
    label: str
    role: str
    value: Decimal
    decimals: int
    unit: UnitKind | None
    number: str
    """The number exactly as shown, e.g. ``61,9``, ``\u22123,1``, ``+5,2``."""
    display: str
    """Number with unit, e.g. ``61,9 MPa``."""
    slot: str
    signed: bool = False
    terms: str | None = None
    """Regex naming the ingredient or feature (``zement``): a contribution or amount must be
    stated in a clause that names it."""


@dataclass(frozen=True)
class Identifier:
    """A verbatim input string that may contain digits (case id, file, element, model).

    ``labelled`` identifiers look like numbers (``123456``, ``999kg``) and are masked only
    right after a label such as "Fall" or "Version"; all others are masked as whole tokens.
    """

    text: str
    labelled: bool


@dataclass(frozen=True)
class TextSlot:
    """A non-numeric slot (identifier, date or time) with its rendered text."""

    slot: str
    label: str
    text: str


@dataclass(frozen=True)
class FactGroup:
    title: str
    lines: tuple[str, ...]


@dataclass(frozen=True)
class FactSheet:
    """Everything the generator may state about one case."""

    case: InspectionInput
    facts: tuple[Fact, ...]
    groups: tuple[FactGroup, ...]
    identifiers: tuple[Identifier, ...]
    text_slots: tuple[TextSlot, ...]
    dates: tuple[str, ...]
    times: tuple[str, ...]

    def fact(self, field: str) -> Fact | None:
        return next((fact for fact in self.facts if fact.field == field), None)

    def roles(self) -> set[str]:
        return {fact.role for fact in self.facts}

    def slots(self) -> dict[str, str]:
        """Slot name to inserted text. A number slot inserts the complete quantity, with sign
        and unit (``61,9 MPa``, a signed ``-3,1 MPa``, ``93,1 %``); the model writes neither."""
        rendered = {fact.slot: fact.display for fact in self.facts}
        rendered.update({slot.slot: slot.text for slot in self.text_slots})
        return rendered

    def render(self) -> str:
        """The fact list exactly as it is shown to the model in free-text mode."""
        return "\n\n".join(
            f"{group.title}:\n" + "\n".join(f"- {line}" for line in group.lines)
            for group in self.groups
        )


def to_decimal(value: float) -> Decimal:
    """Convert without binary noise: ``61.9`` becomes ``Decimal('61.9')``."""
    return Decimal(repr(value)) if isinstance(value, float) else Decimal(value)


def quantize(value: Decimal, decimals: int) -> Decimal:
    return value.quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_UP)


def shown_decimals(value: Decimal, max_decimals: int) -> int:
    """Fewest decimals (at most ``max_decimals``) that show ``value`` rounded to the maximum."""
    rounded = quantize(value, max_decimals)
    for decimals in range(max_decimals + 1):
        if quantize(rounded, decimals) == rounded:
            return decimals
    return max_decimals


def format_de(value: Decimal, decimals: int, *, signed: bool = False) -> str:
    """German number format: decimal comma, typographic minus, no thousands separator."""
    rounded = quantize(value, decimals)
    digits = f"{abs(rounded):.{decimals}f}".replace(".", ",")
    if rounded < 0:
        return f"{MINUS}{digits}"
    if signed and rounded > 0:
        return f"+{digits}"
    return digits


_UNIT_SUFFIX: dict[UnitKind, str] = {"MPa": " MPa", "kg/m³": " kg/m³", "%": " %", "px": " Pixel"}


def _fact(
    field: str,
    label: str,
    role: str,
    value: Decimal,
    decimals: int,
    unit: UnitKind | None,
    slot: str,
    *,
    signed: bool = False,
) -> Fact:
    shown = quantize(value, decimals)
    number = format_de(shown, decimals, signed=signed)
    if unit == "d":
        suffix = " Tag" if shown == 1 else " Tage"
    else:
        suffix = _UNIT_SUFFIX.get(unit, "") if unit is not None else ""
    return Fact(field, label, role, shown, decimals, unit, number, number + suffix, slot, signed)


def exact_decimals(value: Decimal) -> int:
    """Decimals needed to write ``value`` exactly (``Decimal('49.99999999')`` needs 8)."""
    exponent = value.normalize().as_tuple().exponent
    return max(0, -exponent) if isinstance(exponent, int) else 0


def decision_preserving_percent(probability: float, threshold: float) -> tuple[int, int]:
    """Decimals (in percent) for probability and threshold that keep the raw comparison.

    There is no precision cap. The threshold is shown exactly. The probability gets at least
    one decimal and as many more as needed until the shown values compare like the raw ones:
    same side of the threshold, and different unless the raw values are equal. At the latest
    this is reached when the probability is shown exactly (0.4999999999 renders as
    ``49,99999999 %`` against ``50 %``).
    """
    raw_p, raw_t = to_decimal(probability) * 100, to_decimal(threshold) * 100
    t_decimals = exact_decimals(raw_t)
    limit = max(exact_decimals(raw_p), 1)
    for p_decimals in range(1, limit + 1):
        shown_p = quantize(raw_p, p_decimals)
        same_side = (shown_p >= raw_t) == (raw_p >= raw_t)
        distinct = (shown_p != raw_t) or (raw_p == raw_t)
        if same_side and distinct:
            return shown_decimals(shown_p, p_decimals), t_decimals
    return limit, t_decimals


def _mix_facts(mix: ConcreteMix) -> tuple[list[Fact], FactGroup]:
    facts: list[Fact] = []
    lines: list[str] = []
    for name, label in MIX_LABELS.items():
        value = to_decimal(getattr(mix, name))
        decimals = shown_decimals(value, MAX_INGREDIENT_DECIMALS)
        fact = replace(
            _fact(
                f"mix.{name}",
                label,
                f"ingredient:{name}",
                value,
                decimals,
                "kg/m³",
                MIX_SLOTS[name],
            ),
            terms=FEATURE_TERMS[name],
        )
        facts.append(fact)
        lines.append(f"{label}: {fact.display}")
    ratio = _fact(
        "mix.water_cement_ratio",
        "w/z-Wert (Wasser/Zement, berechnet)",
        "ratio",
        to_decimal(mix.water) / to_decimal(mix.cement),
        RATIO_DECIMALS,
        None,
        "wz_wert",
    )
    ratio = replace(ratio, terms=FEATURE_TERMS["water_cement_ratio"])
    age = _fact(
        "mix.age_days",
        "Betonalter bei Prüfung",
        "age",
        Decimal(mix.age_days),
        0,
        "d",
        "betonalter",
    )
    facts += [ratio, age]
    lines += [f"{ratio.label}: {ratio.display}", f"{age.label}: {age.display}"]
    return facts, FactGroup("Betonrezeptur (Massen je m³ Beton)", tuple(lines))


def _slug(text: str) -> str:
    for umlaut, replacement in (("ä", "ae"), ("ö", "oe"), ("ü", "ue"), ("ß", "ss")):
        text = text.lower().replace(umlaut, replacement)
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _feature_facts(
    strength: StrengthPrediction, existing: list[Fact]
) -> tuple[list[Fact], list[str]]:
    facts: list[Fact] = []
    lines: list[str] = []
    for index, item in enumerate(strength.top_features):
        label = _FEATURE_LABELS.get(item.feature, item.feature)
        slug = _slug(label)
        terms = FEATURE_TERMS.get(item.feature, re.escape(label.lower()))
        raw = to_decimal(item.value)
        if item.feature in MIX_LABELS:
            value = _fact(
                f"strength.top_features[{index}].value",
                f"Wert von {label}",
                f"ingredient:{item.feature}",
                raw,
                shown_decimals(raw, MAX_INGREDIENT_DECIMALS),
                "kg/m³",
                f"wert_{slug}",
            )
        elif item.feature in _AGE_FEATURES:
            value = _fact(
                f"strength.top_features[{index}].value",
                f"Wert von {label}",
                "age",
                raw,
                0,
                "d",
                f"wert_{slug}",
            )
        else:
            role = "ratio" if item.feature in _RATIO_FEATURES else f"feature:{item.feature}"
            value = _fact(
                f"strength.top_features[{index}].value",
                f"Wert von {label}",
                role,
                raw,
                RATIO_DECIMALS if role == "ratio" else shown_decimals(raw, RATIO_DECIMALS),
                None,
                f"wert_{slug}",
            )
        contribution = _fact(
            f"strength.top_features[{index}].contribution_mpa",
            f"Beitrag von {label}",
            "contribution",
            to_decimal(item.contribution_mpa),
            STRENGTH_DECIMALS,
            "MPa",
            f"beitrag_{slug}",
            signed=True,
        )
        value = replace(value, terms=terms)
        contribution = replace(contribution, terms=terms)
        # A feature value that renders like an existing fact of the same role adds nothing.
        duplicate = any(
            other.role == value.role and other.number == value.number for other in existing
        )
        if not duplicate:
            facts.append(value)
        facts.append(contribution)
        lines.append(
            f"Einflussgröße {label} (Wert {value.display}): Beitrag {contribution.display}"
        )
    return facts, lines


def _strength_facts(
    strength: StrengthPrediction, existing: list[Fact]
) -> tuple[list[Fact], FactGroup]:
    def mpa(field: str, label: str, role: str, value: float, slot: str) -> Fact:
        return _fact(
            f"strength.{field}", label, role, to_decimal(value), STRENGTH_DECIMALS, "MPa", slot
        )

    point = mpa(
        "point_mpa",
        "Prognostizierte Druckfestigkeit",
        "point",
        strength.point_mpa,
        "druckfestigkeit",
    )
    lower = mpa(
        "lower_mpa",
        "Untere Grenze des Prognoseintervalls",
        "lower",
        strength.lower_mpa,
        "untere_grenze",
    )
    upper = mpa(
        "upper_mpa",
        "Obere Grenze des Prognoseintervalls",
        "upper",
        strength.upper_mpa,
        "obere_grenze",
    )
    raw_coverage = to_decimal(strength.interval_coverage) * 100
    coverage = _fact(
        "strength.interval_coverage",
        "Nominelle Überdeckung des Prognoseintervalls",
        "coverage",
        raw_coverage,
        exact_decimals(raw_coverage),
        "%",
        "ueberdeckung",
    )
    facts = [point, lower, upper, coverage]
    domain = (
        "ja"
        if strength.in_training_domain
        else "NEIN \u2013 mindestens eine Eingabe liegt außerhalb des Trainingsbereichs"
    )
    lines = [
        f"Modell: {strength.model_name} (Version {strength.model_version})",
        f"{point.label}: {point.display}",
        f"{lower.label}: {lower.display}",
        f"{upper.label}: {upper.display}",
        f"{coverage.label}: {coverage.display}",
        f"Methode des Prognoseintervalls: {strength.interval_method}",
        f"Eingaben im Wertebereich der Trainingsdaten: {domain}",
    ]
    feature_facts, feature_lines = _feature_facts(strength, existing + facts)
    return facts + feature_facts, FactGroup(
        "Druckfestigkeitsprognose (Modellschätzung)", tuple(lines + feature_lines)
    )


def _crack_facts(
    crack: CrackPrediction, image: ImageMetadata | None
) -> tuple[list[Fact], FactGroup]:
    p_decimals, t_decimals = decision_preserving_percent(crack.crack_probability, crack.threshold)
    probability = _fact(
        "crack.crack_probability",
        "Risswahrscheinlichkeit",
        "probability",
        to_decimal(crack.crack_probability) * 100,
        p_decimals,
        "%",
        "risswahrscheinlichkeit",
    )
    threshold = _fact(
        "crack.threshold",
        "Entscheidungsschwelle",
        "threshold",
        to_decimal(crack.threshold) * 100,
        t_decimals,
        "%",
        "entscheidungsschwelle",
    )
    facts = [probability, threshold]
    lines = [f"Modell: {crack.model_name} (Version {crack.model_version})"]
    if image is not None:
        width = _fact(
            "image.width_px", "Bildbreite", "pixels", Decimal(image.width_px), 0, "px", "bildbreite"
        )
        height = _fact(
            "image.height_px", "Bildhöhe", "pixels", Decimal(image.height_px), 0, "px", "bildhoehe"
        )
        facts += [width, height]
        lines.append(f"Bild: {image.filename}, {width.number} \u00d7 {height.display}")
        if image.captured_at is not None:
            lines.append(
                f"Aufnahmezeitpunkt: {render_date(image.captured_at)}, "
                f"{render_time(image.captured_at)} Uhr"
            )
    result = "Riss erkannt" if crack.label == "crack" else "kein Riss erkannt"
    lines += [
        f"{probability.label}: {probability.display}",
        f"{threshold.label}: {threshold.display}",
        f"Ergebnis der Klassifikation: {result}",
        "Bewertet wird ein einzelner Bildausschnitt, nicht das gesamte Bauteil.",
    ]
    return facts, FactGroup("Rissklassifikation (Bildausschnitt)", tuple(lines))


def render_date(moment: datetime) -> str:
    return f"{moment:%d.%m.%Y}"


def render_time(moment: datetime) -> str:
    return f"{moment:%H:%M}" if moment.second == 0 else f"{moment:%H:%M:%S}"


def _looks_numeric(text: str) -> bool:
    """Identifiers that start with a digit could hide an amount ("999kg") when masked."""
    return bool(re.match(r"[\d+\-\u2212.,]", text))


def _identifiers(case: InspectionInput) -> list[Identifier]:
    values: list[str | None] = [case.case_id]
    if case.image is not None:
        values += [
            case.image.filename,
            PurePath(case.image.filename).stem,
            case.image.element,
            case.image.sha256,
        ]
    if case.strength is not None:
        values += [
            case.strength.model_name,
            case.strength.model_version,
            case.strength.interval_method,
        ]
        values += [item.feature for item in case.strength.top_features]
    if case.crack is not None:
        values += [case.crack.model_name, case.crack.model_version]
    # Only complete identifier strings: a part such as "C30/37" of "Wand C30/37" is never
    # masked on its own, so it cannot excuse the same characters elsewhere in the text.
    strings = [value for value in values if value and any(char.isdigit() for char in value)]
    unique = list(dict.fromkeys(strings))
    return [
        Identifier(text, labelled=_looks_numeric(text) or not any(c.isalpha() for c in text))
        for text in unique
    ]


def build_facts(case: InspectionInput) -> FactSheet:
    """Collect the citable facts, identifiers, dates and slots of one inspection case."""
    facts: list[Fact] = []
    general = [f"Fall-ID: {case.case_id}"]
    text_slots = [TextSlot("fall_id", "Fall-ID", case.case_id)]
    if case.image is not None and case.image.element:
        general.append(f"Bauteil: {case.image.element}")
        text_slots.append(TextSlot("bauteil", "Bauteil", case.image.element))
    groups = [FactGroup("Allgemein", tuple(general))]
    if case.mix is not None:
        mix_facts, group = _mix_facts(case.mix)
        facts += mix_facts
        groups.append(group)
    if case.strength is not None:
        strength_facts, group = _strength_facts(case.strength, facts)
        facts += strength_facts
        groups.append(group)
    dates: list[str] = []
    times: list[str] = []
    if case.crack is not None:
        crack_facts, group = _crack_facts(case.crack, case.image)
        facts += crack_facts
        groups.append(group)
    if case.image is not None:
        text_slots.append(TextSlot("bild", "Bilddatei", case.image.filename))
        if case.image.captured_at is not None:
            dates.append(render_date(case.image.captured_at))
            times.append(render_time(case.image.captured_at))
            text_slots += [
                TextSlot("aufnahmedatum", "Aufnahmedatum", dates[0]),
                TextSlot("aufnahmezeit", "Aufnahmezeit", f"{times[0]} Uhr"),
            ]
    if case.strength is not None:
        text_slots += [
            TextSlot(
                "festigkeitsmodell",
                "Festigkeitsmodell",
                f"{case.strength.model_name} (Version {case.strength.model_version})",
            ),
            TextSlot("norm_druckfestigkeit", "Norm Druckfestigkeitsprüfung", "DIN EN 12390-3"),
            TextSlot("norm_bohrkern", "Norm Bohrkernprüfung", "DIN EN 12504-1"),
        ]
    if case.crack is not None:
        text_slots.append(
            TextSlot(
                "rissmodell",
                "Rissmodell",
                f"{case.crack.model_name} (Version {case.crack.model_version})",
            )
        )
    return FactSheet(
        case=case,
        facts=tuple(facts),
        groups=tuple(groups),
        identifiers=tuple(_identifiers(case)),
        text_slots=tuple(text_slots),
        dates=tuple(dates),
        times=tuple(times),
    )
