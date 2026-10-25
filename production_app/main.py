from pathlib import Path

from production_app.data import load_dataset, prepare_features, split_data
from production_app.evaluation import evaluate
from production_app.feature_selection import select_features
from production_app.logging_config import get_logger
from production_app.model_service import (
    load_features,
    load_model,
    predict,
    save_features,
    save_model,
)
from production_app.preprocessing import build_pipeline
from production_app.schemas import FEATURES
from production_app.training import train

logger = get_logger(__name__)


def run_training() -> dict[str, float]:
    df = load_dataset()
    X_train, X_test, y_train, y_test = split_data(df)
    features = select_features(X_train, y_train)
    # self-check: the API schema must describe exactly what the pipeline sees
    assert features == list(
        FEATURES
    ), f"schema drift: selected {features} but the API declares {list(FEATURES)}"
    save_features(features)
    pipeline = train(build_pipeline(), X_train[features], y_train)
    metrics = evaluate(y_test, predict(pipeline, X_test[features]))
    save_model(pipeline)
    # self-check: the reloaded artifact must reproduce the in-memory predictions
    assert (
        load_model().predict(X_test[features]) == predict(pipeline, X_test[features])
    ).all()
    logger.info("Save/reload roundtrip verified")
    return metrics


def run_prediction(input_path: Path | None = None) -> None:
    pipeline = load_model()
    features = load_features()
    X = prepare_features(load_dataset(input_path))
    missing = [name for name in features if name not in X.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    X = X[features].head(5)
    predictions = predict(pipeline, X)
    labels = ["malignant" if p else "benign" for p in predictions]
    logger.info("Predictions for %d rows: %s", len(X), labels)
