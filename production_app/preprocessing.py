from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from production_app.config import settings


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("sampler", SMOTETomek(random_state=settings.random_state)),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression()),
        ]
    )
