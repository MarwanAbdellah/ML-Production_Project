from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline

from production_app.config import settings
from production_app.logging_config import get_logger

logger = get_logger(__name__)


def save_model(pipeline: Pipeline) -> None:
    settings.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, settings.model_path)
    logger.info("Saved pipeline to %s", settings.model_path)


@lru_cache(maxsize=1)
def load_model() -> Pipeline:
    """Load the trusted fitted artifact once and reuse it (never refits)."""
    if not settings.model_path.exists():
        logger.error("Model artifact not found: %s", settings.model_path)
        raise FileNotFoundError(settings.model_path)
    logger.info("Loaded pipeline from %s", settings.model_path)
    return joblib.load(settings.model_path)


def predict(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    return pipeline.predict(X)
