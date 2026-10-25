import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Every runtime setting, in one object, with safe defaults."""

    data_path: Path = Path("notebooks/kaggle/breast-cancer.csv")
    model_path: Path = Path("models/model_pipeline.joblib")
    selected_features_path: Path = Path("models/selected_features.json")
    log_level: str = "INFO"
    environment: str = "development"
    test_size: float = 0.2
    random_state: int = 42

    @classmethod
    def from_env(cls) -> "Settings":
        """Build settings from environment variables (.env), else use defaults."""
        defaults = cls()
        return cls(
            data_path=Path(os.getenv("DATA_PATH", str(defaults.data_path))),
            model_path=Path(os.getenv("MODEL_PATH", str(defaults.model_path))),
            selected_features_path=Path(
                os.getenv(
                    "SELECTED_FEATURES_PATH", str(defaults.selected_features_path)
                )
            ),
            log_level=os.getenv("LOG_LEVEL", defaults.log_level),
            environment=os.getenv("ENVIRONMENT", defaults.environment),
            test_size=float(os.getenv("TEST_SIZE", defaults.test_size)),
            random_state=int(os.getenv("RANDOM_STATE", defaults.random_state)),
        )


settings = Settings.from_env()
