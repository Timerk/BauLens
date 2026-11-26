"""Citable facts of an inspection case, rendered once for the prompt and for the checker.

Every number a report may contain comes from a :class:`Fact`. The prompt shows
``Fact.display`` and the faithfulness checker accepts exactly the values listed in
``Fact.renderings``, so what the model is shown and what the checker accepts cannot drift apart.

Permitted renderings (documented in ``docs/results/report.md``):

* every value at the precision shown in the prompt, or numerically equal with trailing zeros
  (``0,30`` for ``0,3``);
* rounded to fewer decimals, down to ``Rendering.min_decimals``;
* probabilities, the interval coverage and the decision threshold either as percent
  (``93,1 %``) or as a fraction (``0,931``);
* a rounding of the crack probability must not move it across the decision threshold;
* signed feature contributions with their sign or as an absolute value without a sign
  (German prose often carries the sign in the verb, "verringert um 3,1 MPa").
"""

import re
from dataclasses import dataclass
from datetime import date, time
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
Quantity = Literal[
    "strength",
    "coverage",
    "probability",
    "threshold",
    "ingredient",
    "age",
    "ratio",
    "contribution",
    "feature",
    "pixels",
]

MINUS = "\u2212"
MAX_INGREDIENT_DECIMALS = 1
STRENGTH_DECIMALS = 1
PERCENT_DECIMALS = 1
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
_FEATURE_LABELS: dict[str, str] = {
    **MIX_LABELS,
    "age_days": "Betonalter",
    "age": "Betonalter",
    "water_cement_ratio": "w/z-Wert",
    "w_c_ratio": "w/z-Wert",
    "water_binder_ratio": "w/b-Wert",
    "w_b_ratio": "w/b-Wert",
}


@dataclass(frozen=True)
class Rendering:
    """One accepted way to write a fact.

    ``value`` is the value exactly as shown to the model with ``decimals`` digits after the
    decimal separator. ``min_decimals`` is the coarsest rounding the checker accepts and
    ``guard`` a boundary (the decision threshold) that a rounding must not cross.
    """

    value: Decimal
    decimals: int
    min_decimals: int
    unit: UnitKind | None
    guard: Decimal | None = None


@dataclass(frozen=True)
class Fact:
    """A citable number from the input, addressed by its dotted field path."""

    field: str
    label: str
    quantity: Quantity
    display: str
    renderings: tuple[Rendering, ...]
    allow_unsigned: bool = False


@dataclass(frozen=True)
class FactGroup:
    """A titled block of fact lines in the prompt."""

    title: str
    lines: tuple[str, ...]


@dataclass(frozen=True)
class FactSheet:
    """Everything the generator may state about one case."""

    case: InspectionInput
    facts: tuple[Fact, ...]
    groups: tuple[FactGroup, ...]
    identifiers: tuple[str, ...]
    versions: tuple[str, ...]
    dates: frozenset[date]
    times: frozenset[time]

    def fact(self, field: str) -> Fact | None:
        return next((fact for fact in self.facts if fact.field == field), None)

    def has_quantity(self, quantity: Quantity) -> bool:
        return any(fact.quantity == quantity for fact in self.facts)

    def render(self) -> str:
        """Render the fact list exactly as it is shown to the model."""
        blocks = []
        for group in self.groups:
            body = "\n".join(f"- {line}" for line in group.lines)
            blocks.append(f"{group.title}:\n{body}")
        return "\n\n".join(blocks)


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
    """German number format with decimal comma, typographic minus and no thousands separator."""
    rounded = quantize(value, decimals)
    digits = f"{abs(rounded):.{decimals}f}".replace(".", ",")
    if rounded < 0:
        return f"{MINUS}{digits}"
    if signed and rounded > 0:
        return f"+{digits}"
    return digits


def _scalar(
    field: str,
    label: str,
    quantity: Quantity,
    value: Decimal,
    *,
    decimals: int,
    min_decimals: int,
    unit: UnitKind | None,
    suffix: str,
    signed: bool = False,
) -> Fact:
    shown = quantize(value, decimals)
    display = format_de(shown, decimals, signed=signed) + suffix
    rendering = Rendering(shown, decimals, min(min_decimals, decimals), unit)
    return Fact(field, label, quantity, display, (rendering,), allow_unsigned=signed)


def _fraction(
    field: str, label: str, quantity: Quantity, value: float, guard: float | None = None
) -> Fact:
    """A fraction in [0, 1] offered as percent (shown) and as a decimal fraction."""
    raw = to_decimal(value)
    percent_decimals = shown_decimals(raw * 100, PERCENT_DECIMALS)
    percent = quantize(raw * 100, percent_decimals)
    fraction = quantize(raw, percent_decimals + 2)
    guard_fraction = to_decimal(guard) if guard is not None else None
    guard_percent = guard_fraction * 100 if guard_fraction is not None else None
    renderings = (
        Rendering(percent, percent_decimals, 0, "%", guard_percent),
        Rendering(
            fraction, percent_decimals + 2, min(2, percent_decimals + 2), None, guard_fraction
        ),
    )
    display = f"{format_de(percent, percent_decimals)} %"
    return Fact(field, label, quantity, display, renderings)


def _mix_facts(mix: ConcreteMix) -> tuple[list[Fact], FactGroup]:
    facts: list[Fact] = []
    lines: list[str] = []
    for name, label in MIX_LABELS.items():
        value = to_decimal(getattr(mix, name))
        decimals = shown_decimals(value, MAX_INGREDIENT_DECIMALS)
        fact = _scalar(
            f"mix.{name}",
            label,
            "ingredient",
            value,
            decimals=decimals,
            min_decimals=0,
            unit="kg/m³",
            suffix=" kg/m³",
        )
        facts.append(fact)
        lines.append(f"{label}: {fact.display}")
    ratio = _scalar(
        "mix.water_cement_ratio",
        "w/z-Wert (Wasser/Zement, berechnet)",
        "ratio",
        to_decimal(mix.water) / to_decimal(mix.cement),
        decimals=RATIO_DECIMALS,
        min_decimals=RATIO_DECIMALS,
        unit=None,
        suffix="",
    )
    age = _scalar(
        "mix.age_days",
        "Betonalter bei Prüfung",
        "age",
        Decimal(mix.age_days),
        decimals=0,
        min_decimals=0,
        unit="d",
        suffix=" Tag" if mix.age_days == 1 else " Tage",
    )
    facts += [ratio, age]
    lines += [f"{ratio.label}: {ratio.display}", f"{age.label}: {age.display}"]
    return facts, FactGroup("Betonrezeptur (Massen je m³ Beton)", tuple(lines))


def _feature_unit(feature: str, value: float) -> tuple[UnitKind | None, str]:
    if feature in MIX_LABELS:
        return "kg/m³", " kg/m³"
    if feature in {"age", "age_days"}:
        return "d", " Tag" if value == 1 else " Tage"
    return None, ""


def _strength_facts(strength: StrengthPrediction) -> tuple[list[Fact], FactGroup]:
    def mpa(field: str, label: str, value: float) -> Fact:
        return _scalar(
            f"strength.{field}",
            label,
            "strength",
            to_decimal(value),
            decimals=STRENGTH_DECIMALS,
            min_decimals=0,
            unit="MPa",
            suffix=" MPa",
        )

    point = mpa("point_mpa", "Prognostizierte Druckfestigkeit", strength.point_mpa)
    lower = mpa("lower_mpa", "Untere Grenze des Prognoseintervalls", strength.lower_mpa)
    upper = mpa("upper_mpa", "Obere Grenze des Prognoseintervalls", strength.upper_mpa)
    coverage = _fraction(
        "strength.interval_coverage",
        "Nominelle Überdeckung",
        "coverage",
        strength.interval_coverage,
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
        f"Prognoseintervall ({coverage.display} nominelle Überdeckung): "
        f"{lower.display} bis {upper.display}",
        f"Methode des Prognoseintervalls: {strength.interval_method}",
        f"Eingaben im Wertebereich der Trainingsdaten: {domain}",
    ]
    for index, item in enumerate(strength.top_features):
        label = _FEATURE_LABELS.get(item.feature, item.feature)
        unit, suffix = _feature_unit(item.feature, item.value)
        raw = to_decimal(item.value)
        max_decimals = MAX_INGREDIENT_DECIMALS if unit is not None else RATIO_DECIMALS
        value = _scalar(
            f"strength.top_features[{index}].value",
            f"Wert von {label}",
            "feature",
            raw,
            decimals=shown_decimals(raw, max_decimals),
            min_decimals=0 if unit is not None else max_decimals,
            unit=unit,
            suffix=suffix,
        )
        contribution = _scalar(
            f"strength.top_features[{index}].contribution_mpa",
            f"Beitrag von {label}",
            "contribution",
            to_decimal(item.contribution_mpa),
            decimals=STRENGTH_DECIMALS,
            min_decimals=0,
            unit="MPa",
            suffix=" MPa",
            signed=True,
        )
        facts += [value, contribution]
        lines.append(
            f"Einflussgröße {label} (Wert {value.display}): Beitrag {contribution.display}"
        )
    return facts, FactGroup("Druckfestigkeitsprognose (Modellschätzung)", tuple(lines))


def _crack_facts(
    crack: CrackPrediction, image: ImageMetadata | None
) -> tuple[list[Fact], FactGroup]:
    probability = _fraction(
        "crack.crack_probability",
        "Risswahrscheinlichkeit",
        "probability",
        crack.crack_probability,
        guard=crack.threshold,
    )
    threshold = _fraction("crack.threshold", "Entscheidungsschwelle", "threshold", crack.threshold)
    facts = [probability, threshold]
    lines = [f"Modell: {crack.model_name} (Version {crack.model_version})"]
    if image is not None:
        width = _scalar(
            "image.width_px",
            "Bildbreite",
            "pixels",
            Decimal(image.width_px),
            decimals=0,
            min_decimals=0,
            unit="px",
            suffix=" Pixel",
        )
        height = _scalar(
            "image.height_px",
            "Bildhöhe",
            "pixels",
            Decimal(image.height_px),
            decimals=0,
            min_decimals=0,
            unit="px",
            suffix=" Pixel",
        )
        facts += [width, height]
        lines.append(f"Bild: {image.filename} ({image.width_px} \u00d7 {image.height_px} Pixel)")
        if image.captured_at is not None:
            moment = image.captured_at
            lines.append(f"Aufnahmezeitpunkt: {moment:%d.%m.%Y}, {moment:%H:%M} Uhr")
    result = "Riss erkannt" if crack.label == "crack" else "kein Riss erkannt"
    lines += [
        f"{probability.label}: {probability.display}",
        f"{threshold.label}: {threshold.display}",
        f"Ergebnis der Klassifikation: {result}",
        "Bewertet wird ein einzelner Bildausschnitt, nicht das gesamte Bauteil.",
    ]
    return facts, FactGroup("Rissklassifikation (Bildausschnitt)", tuple(lines))


def _identifier_strings(case: InspectionInput) -> list[str]:
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
    strings = [value for value in values if value and any(char.isdigit() for char in value)]
    # Reports often quote only part of a free-text identifier ("Feld C2" from
    # "Decke über EG, Feld C2"): also mask its comma-separated parts and its words that mix
    # letters and digits. Pure digit words ("4") are never masked on their own.
    for value in list(strings):
        parts = [part.strip() for part in re.split(r"[,;()]", value)]
        words = [word.strip(".,;()") for word in value.split()]
        for piece in parts + words:
            if any(c.isdigit() for c in piece) and any(c.isalpha() for c in piece):
                strings.append(piece)
    return list(dict.fromkeys(strings))


def build_facts(case: InspectionInput) -> FactSheet:
    """Collect the citable facts, identifiers and dates of one inspection case."""
    facts: list[Fact] = []
    general = [f"Fall-ID: {case.case_id}"]
    if case.image is not None and case.image.element:
        general.append(f"Bauteil: {case.image.element}")
    groups = [FactGroup("Allgemein", tuple(general))]
    if case.mix is not None:
        mix_facts, group = _mix_facts(case.mix)
        facts += mix_facts
        groups.append(group)
    if case.strength is not None:
        strength_facts, group = _strength_facts(case.strength)
        facts += strength_facts
        groups.append(group)
    if case.crack is not None:
        crack_facts, group = _crack_facts(case.crack, case.image)
        facts += crack_facts
        groups.append(group)

    identifiers = _identifier_strings(case)
    dates: set[date] = set()
    times: set[time] = set()
    if case.image is not None and case.image.captured_at is not None:
        dates.add(case.image.captured_at.date())
        times.add(case.image.captured_at.time().replace(second=0, microsecond=0))
    return FactSheet(
        case=case,
        facts=tuple(facts),
        groups=tuple(groups),
        identifiers=tuple(value for value in identifiers if any(c.isalpha() for c in value)),
        versions=tuple(value for value in identifiers if not any(c.isalpha() for c in value)),
        dates=frozenset(dates),
        times=frozenset(times),
    )
