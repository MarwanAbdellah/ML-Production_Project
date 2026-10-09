from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from production_app import config


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("sampler", SMOTETomek(random_state=config.RANDOM_STATE)),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression()),
        ]
    )
