"""Embedded L1 feature selection, kept only when stable across resamples."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from production_app.logging_config import get_logger

logger = get_logger(__name__)

RESAMPLES = 5
SUBSAMPLE = 0.8
MIN_VOTES = 4
COEF_THRESHOLD = 1e-5


def select_features(X_train: pd.DataFrame, y_train: pd.Series) -> list[str]:
    """Pick features with L1 logistic regression, using the training fold only.

    A feature must survive in at least MIN_VOTES of RESAMPLES subsamples: on a
    few hundred rows, one split's non-zero coefficients are partly noise.
    """
    votes = pd.Series(0, index=X_train.columns)
    for seed in range(RESAMPLES):
        X_part, _, y_part, _ = train_test_split(
            X_train, y_train, train_size=SUBSAMPLE, stratify=y_train, random_state=seed
        )
        scaler = StandardScaler().fit(X_part)
        selector = LogisticRegressionCV(
            Cs=[0.05, 0.1, 0.2, 0.3, 0.5, 1.0],
            l1_ratios=(1.0,),  # pure L1; `penalty=` is deprecated since sklearn 1.8
            solver="liblinear",
            cv=StratifiedKFold(5, shuffle=True, random_state=seed),
            scoring="f1",
            max_iter=5000,
            random_state=seed,
            use_legacy_attributes=False,
        ).fit(scaler.transform(X_part), y_part)
        votes += np.abs(selector.coef_[0]) > COEF_THRESHOLD

    selected = [name for name in X_train.columns if votes[name] >= MIN_VOTES]
    logger.info(
        "Selected %d of %d features (kept in >=%d/%d resamples): %s",
        len(selected),
        X_train.shape[1],
        MIN_VOTES,
        RESAMPLES,
        selected,
    )
    return selected
