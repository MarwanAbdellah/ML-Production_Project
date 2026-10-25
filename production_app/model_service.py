import json
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


def save_features(features: list[str]) -> None:
    settings.selected_features_path.parent.mkdir(parents=True, exist_ok=True)
    settings.selected_features_path.write_text(
        json.dumps(features, indent=2) + "\n", encoding="utf-8"
    )
    logger.info("Saved %d selected features", len(features))


@lru_cache(maxsize=1)
def load_features() -> list[str]:
    """The exact selected-feature names and order the pipeline was fitted on."""
    if not settings.selected_features_path.exists():
        logger.error("Feature list not found: %s", settings.selected_features_path)
        raise FileNotFoundError(settings.selected_features_path)
    return json.loads(settings.selected_features_path.read_text(encoding="utf-8"))
