"""Shared data contracts between the analysis modules, the report generator and the API.

These models are the only interface between ``strength``, ``cracks``, ``report`` and ``api``.
Change them deliberately: the report's numeric faithfulness check treats every number in an
``InspectionInput`` as citable, so adding a numeric field changes what a report may state.

Units follow the UCI Concrete Compressive Strength dataset: ingredient masses in kg per m³ of
concrete, age in days, strength in MPa.
"""

from datetime import datetime
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Physically plausible input bounds; deliberately wider than the training data, whose ranges
# are reported separately through ``StrengthPrediction.in_training_domain``.
_KG = "kg/m³"


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ConcreteMix(_Frozen):
    """One concrete mix design at a given testing age."""

    cement: float = Field(gt=0, le=1000, description=f"Portland cement in {_KG}")
    blast_furnace_slag: float = Field(ge=0, le=600, description=f"Blast furnace slag in {_KG}")
    fly_ash: float = Field(ge=0, le=400, description=f"Fly ash in {_KG}")
    water: float = Field(gt=0, le=400, description=f"Water in {_KG}")
    superplasticizer: float = Field(ge=0, le=60, description=f"Superplasticizer in {_KG}")
    coarse_aggregate: float = Field(gt=0, le=1600, description=f"Coarse aggregate in {_KG}")
    fine_aggregate: float = Field(gt=0, le=1300, description=f"Fine aggregate in {_KG}")
    age_days: int = Field(ge=1, le=3650, description="Age at testing in days")


class FeatureContribution(_Frozen):
    """Contribution of one (engineered) feature to a single strength prediction."""

    feature: str
    value: float = Field(description="Feature value for this mix")
    contribution_mpa: float = Field(description="Signed contribution in MPa (e.g. SHAP value)")


class StrengthPrediction(_Frozen):
    """Predicted compressive strength with a prediction interval."""

    point_mpa: float
    lower_mpa: float
    upper_mpa: float
    interval_coverage: float = Field(gt=0, lt=1, description="Nominal coverage, e.g. 0.9")
    interval_method: str = Field(description="e.g. 'split conformal (absolute residuals)'")
    in_training_domain: bool = Field(
        description="False if any input lies outside the ranges seen during training"
    )
    top_features: list[FeatureContribution] = Field(default_factory=list, max_length=10)
    model_name: str
    model_version: str

    @model_validator(mode="after")
    def _ordered_interval(self) -> Self:
        if not self.lower_mpa <= self.point_mpa <= self.upper_mpa:
            raise ValueError("expected lower_mpa <= point_mpa <= upper_mpa")
        return self


class CrackPrediction(_Frozen):
    """Binary crack classification of one surface image."""

    crack_probability: float = Field(ge=0, le=1)
    threshold: float = Field(gt=0, lt=1, description="Decision threshold chosen on validation")
    label: Literal["crack", "no_crack"]
    model_name: str
    model_version: str

    @model_validator(mode="after")
    def _label_matches_threshold(self) -> Self:
        expected = "crack" if self.crack_probability >= self.threshold else "no_crack"
        if self.label != expected:
            raise ValueError(f"label {self.label!r} contradicts probability and threshold")
        return self


class ImageMetadata(_Frozen):
    """Metadata of an inspected image. Free-text fields must not carry measurements."""

    filename: str = Field(min_length=1, max_length=255)
    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    captured_at: datetime | None = None
    element: str | None = Field(
        default=None, max_length=120, description="Inspected component, e.g. 'Stütze B'"
    )


class InspectionInput(_Frozen):
    """Everything the report generator may use. At least one analysis result is required."""

    case_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    mix: ConcreteMix | None = None
    strength: StrengthPrediction | None = None
    image: ImageMetadata | None = None
    crack: CrackPrediction | None = None

    @model_validator(mode="after")
    def _has_result(self) -> Self:
        if self.strength is None and self.crack is None:
            raise ValueError("expected a strength or a crack result")
        if self.strength is not None and self.mix is None:
            raise ValueError("a strength result requires the mix it was predicted for")
        return self
