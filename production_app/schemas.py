"""Pydantic request/response schemas for the prediction API."""

from typing import Annotated, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

# Exact feature names and order the saved pipeline was fitted on (Task 01).
# The 6 "concave points_*" columns are not valid identifiers, so those fields
# keep a pythonic name and expose the dataset name through an alias.
FEATURES = (
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",
    "radius_se",
    "texture_se",
    "perimeter_se",
    "area_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "concave points_se",
    "symmetry_se",
    "fractal_dimension_se",
    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave points_worst",
    "symmetry_worst",
    "fractal_dimension_worst",
)

LABELS = {0: "benign", 1: "malignant"}

# Every measurement is a non-negative, finite number: NaN and infinity must
# never reach the pipeline.
FeatureValue = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class PredictRequest(BaseModel):
    """One observation: the 30 cytological measurements of a single tumor."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    radius_mean: FeatureValue
    texture_mean: FeatureValue
    perimeter_mean: FeatureValue
    area_mean: FeatureValue
    smoothness_mean: FeatureValue
    compactness_mean: FeatureValue
    concavity_mean: FeatureValue
    concave_points_mean: FeatureValue = Field(alias="concave points_mean")
    symmetry_mean: FeatureValue
    fractal_dimension_mean: FeatureValue
    radius_se: FeatureValue
    texture_se: FeatureValue
    perimeter_se: FeatureValue
    area_se: FeatureValue
    smoothness_se: FeatureValue
    compactness_se: FeatureValue
    concavity_se: FeatureValue
    concave_points_se: FeatureValue = Field(alias="concave points_se")
    symmetry_se: FeatureValue
    fractal_dimension_se: FeatureValue
    radius_worst: FeatureValue
    texture_worst: FeatureValue
    perimeter_worst: FeatureValue
    area_worst: FeatureValue
    smoothness_worst: FeatureValue
    compactness_worst: FeatureValue
    concavity_worst: FeatureValue
    concave_points_worst: FeatureValue = Field(alias="concave points_worst")
    symmetry_worst: FeatureValue
    fractal_dimension_worst: FeatureValue

    def to_frame(self) -> pd.DataFrame:
        """One-row frame in the exact column order the pipeline expects."""
        return pd.DataFrame([self.model_dump(by_alias=True)], columns=list(FEATURES))


class PredictResponse(BaseModel):
    """Predicted class plus a human-readable label."""

    prediction: int = Field(..., description="0 = benign, 1 = malignant")
    label: str
    probability: float | None = Field(
        None, description="Malignant probability, when the model supports scores"
    )


class HealthResponse(BaseModel):
    status: Literal["ok", "unhealthy"]
    model_loaded: bool
    detail: str | None = None
