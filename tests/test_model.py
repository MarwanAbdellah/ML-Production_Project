"""The saved artifact, the feature list, and the API schema must agree."""

import pandas as pd
import pytest

from production_app.config import settings
from production_app.model_service import load_features, load_model, predict
from production_app.schemas import FEATURES, PredictRequest


def test_artifact_matches_declared_features():
    assert list(load_model().feature_names_in_) == list(FEATURES)
    assert load_features() == list(FEATURES)


def test_request_frame_uses_the_same_order():
    row = pd.read_csv(settings.data_path).iloc[0]
    payload = {name: float(row[name]) for name in FEATURES}
    assert PredictRequest(**payload).to_frame().columns.tolist() == list(FEATURES)


def test_predict_without_retraining():
    rows = pd.read_csv(settings.data_path).head(3)[list(FEATURES)]
    assert len(predict(load_model(), rows)) == 3


@pytest.mark.parametrize(("label", "expected"), [("B", 0), ("M", 1)])
def test_known_rows_predict_their_class(label, expected):
    df = pd.read_csv(settings.data_path)
    row = df[df["diagnosis"] == label].head(1)[list(FEATURES)]
    assert predict(load_model(), row)[0] == expected
