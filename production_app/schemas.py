"""Pydantic request/response schemas for the prediction API."""

from typing import Annotated, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

# Selected in Task 03 (embedded L1 + stability filter, see
# notebooks/02_feature_selection.ipynb). Dataset order, which is also the order
# the saved pipeline was fitted on. The 2 "concave points_*" columns are not
# valid identifiers, so those fields keep a pythonic name and expose the
# dataset name through an alias.
FEATURES = (
    "concave points_mean",
    "radius_se",
    "radius_worst",
    "texture_worst",
    "smoothness_worst",
    "concavity_worst",
    "concave points_worst",
    "symmetry_worst",
)

LABELS = {0: "benign", 1: "malignant"}

# Every measurement is a non-negative, finite number: NaN and infinity must
# never reach the pipeline.
FeatureValue = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class PredictRequest(BaseModel):
    """One observation: the selected cytological measurements of a single tumor."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    concave_points_mean: FeatureValue = Field(alias="concave points_mean")
    radius_se: FeatureValue
    radius_worst: FeatureValue
    texture_worst: FeatureValue
    smoothness_worst: FeatureValue
    concavity_worst: FeatureValue
    concave_points_worst: FeatureValue = Field(alias="concave points_worst")
    symmetry_worst: FeatureValue

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
