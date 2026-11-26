import pytest
from pydantic import ValidationError

from baulens.schemas import (
    ConcreteMix,
    CrackPrediction,
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


def _strength(**overrides: object) -> StrengthPrediction:
    values: dict[str, object] = {
        "point_mpa": 61.9,
        "lower_mpa": 54.2,
        "upper_mpa": 69.6,
        "interval_coverage": 0.9,
        "interval_method": "split conformal",
        "in_training_domain": True,
        "model_name": "hist_gradient_boosting",
        "model_version": "test",
    }
    values.update(overrides)
    return StrengthPrediction.model_validate(values)


def test_valid_inspection_input_round_trips() -> None:
    crack = CrackPrediction(
        crack_probability=0.93,
        threshold=0.5,
        label="crack",
        model_name="resnet18",
        model_version="test",
    )
    image = ImageMetadata(filename="00001.jpg", width_px=227, height_px=227)
    case = InspectionInput(
        case_id="case-01", mix=MIX, strength=_strength(), image=image, crack=crack
    )
    assert InspectionInput.model_validate_json(case.model_dump_json()) == case


def test_interval_must_contain_point() -> None:
    with pytest.raises(ValidationError, match="lower_mpa <= point_mpa"):
        _strength(lower_mpa=65.0)


def test_crack_label_must_match_threshold() -> None:
    with pytest.raises(ValidationError, match="contradicts"):
        CrackPrediction(
            crack_probability=0.2,
            threshold=0.5,
            label="crack",
            model_name="resnet18",
            model_version="test",
        )


def test_inspection_input_requires_a_result_and_mix() -> None:
    with pytest.raises(ValidationError, match="strength or a crack"):
        InspectionInput(case_id="empty")
    with pytest.raises(ValidationError, match="requires the mix"):
        InspectionInput(case_id="no-mix", strength=_strength())


def test_mix_rejects_unknown_fields_and_negative_values() -> None:
    with pytest.raises(ValidationError):
        ConcreteMix.model_validate({**MIX.model_dump(), "slump": 10})
    with pytest.raises(ValidationError):
        ConcreteMix.model_validate({**MIX.model_dump(), "fly_ash": -1})
