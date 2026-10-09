from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from production_app import config
from production_app.logging_config import get_logger

logger = get_logger(__name__)

REQUIRED_COLUMNS = {"diagnosis", "radius_mean"}


def load_dataset(path: Path | None = None) -> pd.DataFrame:
    path = path or config.DATA_PATH
    if not path.exists():
        logger.error("Dataset not found: %s", path)
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        logger.error("Dataset %s missing required columns: %s", path, sorted(missing))
        raise ValueError(f"missing required columns: {sorted(missing)}")
    logger.info("Loaded dataset %s with shape %s", path, df.shape)
    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in ("id", "diagnosis") if c in df.columns])


def split_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = prepare_features(df)
    y = (df["diagnosis"] == "M").astype(int)
    logger.info(
        "Splitting data: test_size=%s random_state=%s",
        config.TEST_SIZE,
        config.RANDOM_STATE,
    )
    return train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
