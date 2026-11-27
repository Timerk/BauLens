"""Inspection cases shared by the report tests."""

from datetime import datetime

from baulens.schemas import (
    ConcreteMix,
    CrackPrediction,
    FeatureContribution,
    ImageMetadata,
    InspectionInput,
    StrengthPrediction,
)

MIX = ConcreteMix(
    cement=540.0,
    blast_furnace_slag=0.0,
    fly_ash=0.0,
    water=162.0,
    superplasticizer=2.5,
    coarse_aggregate=1040.0,
    fine_aggregate=676.0,
    age_days=28,
)


def strength(**overrides: object) -> StrengthPrediction:
    values: dict[str, object] = {
        "point_mpa": 61.87,
        "lower_mpa": 54.2,
        "upper_mpa": 69.6,
        "interval_coverage": 0.9,
        "interval_method": "split conformal",
        "in_training_domain": True,
        "top_features": [
            FeatureContribution(feature="cement", value=540.0, contribution_mpa=5.21),
            FeatureContribution(feature="age_days", value=28.0, contribution_mpa=-3.14),
        ],
        "model_name": "lightgbm_conformal",
        "model_version": "m2-test",
    }
    values.update(overrides)
    return StrengthPrediction.model_validate(values)


def crack(probability: float = 0.931, threshold: float = 0.5) -> CrackPrediction:
    return CrackPrediction(
        crack_probability=probability,
        threshold=threshold,
        label="crack" if probability >= threshold else "no_crack",
        model_name="resnet18",
        model_version="m3-test",
    )


IMAGE = ImageMetadata(
    filename="IMG_0423.jpg",
    width_px=227,
    height_px=227,
    captured_at=datetime(2026, 3, 14, 10, 15),
    element="Decke über EG, Feld C2",
)


def strength_case(**overrides: object) -> InspectionInput:
    return InspectionInput(case_id="PB-007", mix=MIX, strength=strength(**overrides))


def crack_case(probability: float = 0.931, threshold: float = 0.5) -> InspectionInput:
    return InspectionInput(case_id="PB-013", image=IMAGE, crack=crack(probability, threshold))


def both_case(**overrides: object) -> InspectionInput:
    return InspectionInput(
        case_id="PB-022", mix=MIX, strength=strength(**overrides), image=IMAGE, crack=crack()
    )


def template_reply(case: InspectionInput, **changes: str) -> str:
    """A free-text model reply built from the template text."""
    import json

    from baulens.report.facts import build_facts
    from baulens.report.template import template_draft

    draft = template_draft(build_facts(case)).model_dump(exclude_none=True)
    draft.pop("bewertung", None)
    draft.update(changes)
    return json.dumps(draft, ensure_ascii=False)


NARRATIVE = {
    "bewertung": "Die Befunde sind als Modellschätzung einzuordnen und fachlich zu prüfen.",
    "einschraenkungen": "Der Entwurf beruht auf Modellen und ersetzt keine Prüfung vor Ort.",
    "empfehlung": "Eine Sichtprüfung vor Ort und eine Prüfung an Bohrkernen werden empfohlen.",
}


def narrative_reply(**changes: str) -> str:
    """A narrative-mode reply without numbers or findings."""
    import json

    return json.dumps({**NARRATIVE, **changes}, ensure_ascii=False)
