import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from production_app.logging_config import get_logger

logger = get_logger(__name__)


def evaluate(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }
    logger.info("Evaluation metrics: %s", metrics)
    logger.info("Confusion matrix:\n%s", confusion_matrix(y_true, y_pred))
    logger.info("Classification report:\n%s", classification_report(y_true, y_pred))
    return metrics
