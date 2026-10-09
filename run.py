import argparse
import sys
from pathlib import Path

from production_app import config
from production_app.logging_config import configure_logging, get_logger
from production_app.main import run_prediction, run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Breast cancer classification app")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("train", help="Train, evaluate and save the pipeline")
    predict_parser = sub.add_parser("predict", help="Predict with the saved pipeline")
    predict_parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="CSV file to predict on (default: dataset)",
    )
    args = parser.parse_args()

    configure_logging(config.LOG_LEVEL)
    logger = get_logger(__name__)
    logger.info("Environment: %s", config.ENVIRONMENT)

    try:
        if args.command == "train":
            run_training()
        else:
            run_prediction(args.input)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
