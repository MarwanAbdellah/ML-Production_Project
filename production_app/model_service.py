import joblib
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline

from production_app import config
from production_app.logging_config import get_logger

logger = get_logger(__name__)


def save_model(pipeline: Pipeline) -> None:
    config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, config.MODEL_PATH)
    logger.info("Saved pipeline to %s", config.MODEL_PATH)


def load_model() -> Pipeline:
    if not config.MODEL_PATH.exists():
        logger.error("Model artifact not found: %s", config.MODEL_PATH)
        raise FileNotFoundError(config.MODEL_PATH)
    logger.info("Loaded pipeline from %s", config.MODEL_PATH)
    return joblib.load(config.MODEL_PATH)


def predict(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    return pipeline.predict(X)
