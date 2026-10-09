import time

import pandas as pd
from imblearn.pipeline import Pipeline

from production_app.logging_config import get_logger

logger = get_logger(__name__)


def train(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    logger.info("Training started on %d samples", len(X_train))
    start = time.perf_counter()
    pipeline.fit(X_train, y_train)
    logger.info("Training finished in %.2f seconds", time.perf_counter() - start)
    return pipeline
