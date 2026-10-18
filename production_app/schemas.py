"""Pydantic request/response schemas for the prediction API."""

from typing import Literal

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


class PredictRequest(BaseModel):
    """One observation: the 30 cytological measurements of a single tumor."""

    model_config = ConfigDict(populate_by_name=True)

    radius_mean: float = Field(..., ge=0)
    texture_mean: float = Field(..., ge=0)
    perimeter_mean: float = Field(..., ge=0)
    area_mean: float = Field(..., ge=0)
    smoothness_mean: float = Field(..., ge=0)
    compactness_mean: float = Field(..., ge=0)
    concavity_mean: float = Field(..., ge=0)
    concave_points_mean: float = Field(..., ge=0, alias="concave points_mean")
    symmetry_mean: float = Field(..., ge=0)
    fractal_dimension_mean: float = Field(..., ge=0)
    radius_se: float = Field(..., ge=0)
    texture_se: float = Field(..., ge=0)
    perimeter_se: float = Field(..., ge=0)
    area_se: float = Field(..., ge=0)
    smoothness_se: float = Field(..., ge=0)
    compactness_se: float = Field(..., ge=0)
    concavity_se: float = Field(..., ge=0)
    concave_points_se: float = Field(..., ge=0, alias="concave points_se")
    symmetry_se: float = Field(..., ge=0)
    fractal_dimension_se: float = Field(..., ge=0)
    radius_worst: float = Field(..., ge=0)
    texture_worst: float = Field(..., ge=0)
    perimeter_worst: float = Field(..., ge=0)
    area_worst: float = Field(..., ge=0)
    smoothness_worst: float = Field(..., ge=0)
    compactness_worst: float = Field(..., ge=0)
    concavity_worst: float = Field(..., ge=0)
    concave_points_worst: float = Field(..., ge=0, alias="concave points_worst")
    symmetry_worst: float = Field(..., ge=0)
    fractal_dimension_worst: float = Field(..., ge=0)

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
