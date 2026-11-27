"""The fixed evaluation cases: loading, saving and their seeded synthetic generator.

The cases are synthetic but plausible. Mixes are drawn from the ranges of the UCI Concrete
Compressive Strength data (cement 102-540 kg/m³, water 122-247 kg/m³, age 1-365 days) and
strengths from a simple Abrams-type water/binder relation with an ACI-style age factor, so
point predictions, interval widths and SHAP-like contributions look like the output of a
real model. They are fixtures for the report generator, not training data.
"""

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

import numpy as np

from baulens.schemas import (
    ConcreteMix,
    CrackPrediction,
    FeatureContribution,
    ImageMetadata,
    InspectionInput,
    StrengthPrediction,
)

CASES_SEED = 20261126
STRENGTH_MODEL = ("lightgbm_conformal", "m2-preview")
CRACK_MODEL = ("resnet18", "m3-preview")
ELEMENTS = (
    "Stütze Achse B/3",
    "Unterzug Ebene 2, Feld Süd",
    "Brüstung Treppenhaus Nord",
    "Wand Tiefgarage, Achse 4",
    "Fundamentbalken Pfeiler Ost",
    "Decke über EG, Feld C2",
    "Kragplatte Balkon Westseite",
    "Widerlager Brücke Mühlbach",
    "Schachtwand Pumpwerk Süd",
    "Attika Flachdach Gebäude Ä",
)

Interval = Literal["narrow", "normal", "wide"]
CrackKind = Literal["crack", "no_crack", "below", "above"]


@dataclass(frozen=True)
class Scenario:
    """What a case must cover; numbers are drawn from the seeded generator."""

    age: int | None = None
    interval: Interval = "normal"
    coverage: float = 0.9
    out_of_domain: Literal["cement", "water", "superplasticizer", "age"] | None = None
    negative_lower: bool = False
    crack: CrackKind | None = None
    threshold: float = 0.5
    captured: bool = True


SCENARIOS: tuple[Scenario, ...] = (
    # strength only
    Scenario(age=28),
    Scenario(age=28, coverage=0.95),
    Scenario(age=28, interval="narrow"),
    Scenario(age=7, interval="narrow"),
    Scenario(age=56),
    Scenario(age=1, negative_lower=True),
    Scenario(age=3),
    Scenario(age=365),
    Scenario(age=365, coverage=0.8, interval="wide"),
    Scenario(age=28, interval="wide", out_of_domain="cement"),
    Scenario(age=28, interval="wide", out_of_domain="water"),
    Scenario(age=730, interval="wide", out_of_domain="age"),
    # crack only
    Scenario(crack="crack"),
    Scenario(crack="crack", threshold=0.42),
    Scenario(crack="crack", captured=False),
    Scenario(crack="no_crack"),
    Scenario(crack="no_crack", threshold=0.61),
    Scenario(crack="no_crack", captured=False),
    Scenario(crack="below"),
    Scenario(crack="above"),
    Scenario(crack="above", threshold=0.42),
    # strength and crack
    Scenario(age=28, crack="crack"),
    Scenario(age=28, crack="no_crack"),
    Scenario(age=2, crack="above"),
    Scenario(age=28, interval="wide", out_of_domain="superplasticizer", crack="crack"),
    Scenario(age=365, crack="below"),
    Scenario(age=7, interval="narrow", crack="no_crack"),
    Scenario(age=90, interval="wide", coverage=0.95, crack="crack"),
)


def _uniform(rng: np.random.Generator, low: float, high: float, decimals: int = 1) -> float:
    return round(float(rng.uniform(low, high)), decimals)


def _mix(rng: np.random.Generator, scenario: Scenario) -> ConcreteMix:
    assert scenario.age is not None
    values = {
        "cement": _uniform(rng, 220, 480),
        "blast_furnace_slag": _uniform(rng, 40, 180) if rng.random() < 0.5 else 0.0,
        "fly_ash": _uniform(rng, 60, 140) if rng.random() < 0.35 else 0.0,
        "water": _uniform(rng, 150, 195),
        "superplasticizer": _uniform(rng, 2, 11) if rng.random() < 0.6 else 0.0,
        "coarse_aggregate": _uniform(rng, 850, 1100),
        "fine_aggregate": _uniform(rng, 640, 860),
    }
    match scenario.out_of_domain:
        case "cement":
            values["cement"] = _uniform(rng, 570, 620)
        case "water":
            values["water"] = _uniform(rng, 252, 270)
        case "superplasticizer":
            values["superplasticizer"] = _uniform(rng, 36, 45)
        case _:
            pass
    return ConcreteMix(**values, age_days=scenario.age)


def _strength(rng: np.random.Generator, mix: ConcreteMix, scenario: Scenario) -> StrengthPrediction:
    binder = mix.cement + 0.7 * mix.blast_furnace_slag + 0.4 * mix.fly_ash
    water_binder = mix.water / binder
    strength_28 = 108 * math.exp(-2.3 * water_binder) + 0.25 * mix.superplasticizer

    def age_factor(days: int) -> float:
        return days / (4 + 0.85 * days)

    age = min(mix.age_days, 365)
    point = strength_28 * age_factor(age) / age_factor(28) + float(rng.normal(0, 1.5))
    point = round(max(point, 2.5), 2)
    if scenario.negative_lower:
        half_width = round(point + float(rng.uniform(1.5, 3.0)), 2)
    else:
        low, high = {"narrow": (4.2, 5.4), "normal": (6.8, 8.6), "wide": (12.0, 16.5)}[
            scenario.interval
        ]
        half_width = _uniform(rng, low, high, 2)
    age_contribution = {1: -24.0, 2: -19.0, 3: -16.0, 7: -9.0, 28: 0.8, 56: 3.5, 90: 5.0}
    contributions = [
        FeatureContribution(
            feature="age_days",
            value=float(mix.age_days),
            contribution_mpa=round(
                age_contribution.get(mix.age_days, 8.0) + float(rng.normal(0, 0.8)), 2
            ),
        ),
        FeatureContribution(
            feature="cement",
            value=mix.cement,
            contribution_mpa=round((mix.cement - 280) * 0.05 + float(rng.normal(0, 0.5)), 2),
        ),
        FeatureContribution(
            feature="water_cement_ratio",
            value=round(mix.water / mix.cement, 3),
            contribution_mpa=round(-(mix.water / mix.cement - 0.55) * 20, 2),
        ),
    ]
    contributions.sort(key=lambda item: abs(item.contribution_mpa), reverse=True)
    return StrengthPrediction(
        point_mpa=point,
        lower_mpa=round(point - half_width, 2),
        upper_mpa=round(point + half_width, 2),
        interval_coverage=scenario.coverage,
        interval_method="split conformal (absolute residuals)",
        in_training_domain=scenario.out_of_domain is None,
        top_features=contributions,
        model_name=STRENGTH_MODEL[0],
        model_version=STRENGTH_MODEL[1],
    )


def _crack(rng: np.random.Generator, scenario: Scenario) -> CrackPrediction:
    threshold = scenario.threshold
    match scenario.crack:
        case "crack":
            probability = _uniform(rng, 0.9, 0.999, 4)
        case "no_crack":
            probability = _uniform(rng, 0.001, 0.15, 4)
        case "below":
            probability = round(threshold - _uniform(rng, 0.003, 0.03, 4), 4)
        case _:
            probability = round(threshold + _uniform(rng, 0.003, 0.03, 4), 4)
    return CrackPrediction(
        crack_probability=probability,
        threshold=threshold,
        label="crack" if probability >= threshold else "no_crack",
        model_name=CRACK_MODEL[0],
        model_version=CRACK_MODEL[1],
    )


def generate_cases(seed: int = CASES_SEED) -> list[InspectionInput]:
    """Build the evaluation cases deterministically from ``seed``."""
    rng = np.random.default_rng(seed)
    start = datetime(2026, 3, 2, 8, 0)
    cases = []
    for number, scenario in enumerate(SCENARIOS, start=1):
        mix = strength = image = crack = None
        if scenario.age is not None:
            mix = _mix(rng, scenario)
            strength = _strength(rng, mix, scenario)
        if scenario.crack is not None:
            crack = _crack(rng, scenario)
            captured = start + timedelta(
                days=int(rng.integers(0, 200)), minutes=int(rng.integers(0, 600))
            )
            image = ImageMetadata(
                filename=f"IMG_{int(rng.integers(1000, 9999)):04d}.jpg",
                width_px=227,
                height_px=227,
                sha256="".join(f"{value:02x}" for value in rng.integers(0, 256, size=32)),
                captured_at=captured if scenario.captured else None,
                element=ELEMENTS[number % len(ELEMENTS)],
            )
        cases.append(
            InspectionInput(
                case_id=f"PB-{number:03d}", mix=mix, strength=strength, image=image, crack=crack
            )
        )
    return cases


def load_cases(path: Path) -> list[InspectionInput]:
    with path.open(encoding="utf-8") as handle:
        return [InspectionInput.model_validate_json(line) for line in handle if line.strip()]


def save_cases(path: Path, cases: list[InspectionInput]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for case in cases:
            handle.write(case.model_dump_json(exclude_none=True) + "\n")
