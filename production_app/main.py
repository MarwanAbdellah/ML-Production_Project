from pathlib import Path

from production_app.data import load_dataset, prepare_features, split_data
from production_app.evaluation import evaluate
from production_app.logging_config import get_logger
from production_app.model_service import load_model, predict, save_model
from production_app.preprocessing import build_pipeline
from production_app.training import train

logger = get_logger(__name__)


def run_training() -> dict[str, float]:
    df = load_dataset()
    X_train, X_test, y_train, y_test = split_data(df)
    pipeline = train(build_pipeline(), X_train, y_train)
    metrics = evaluate(y_test, predict(pipeline, X_test))
    save_model(pipeline)
    # self-check: the reloaded artifact must reproduce the in-memory predictions
    assert (load_model().predict(X_test) == predict(pipeline, X_test)).all()
    logger.info("Save/reload roundtrip verified")
    return metrics


def run_prediction(input_path: Path | None = None) -> None:
    pipeline = load_model()
    X = prepare_features(load_dataset(input_path)).head(5)
    predictions = predict(pipeline, X)
    labels = ["malignant" if p else "benign" for p in predictions]
    logger.info("Predictions for %d rows: %s", len(X), labels)
