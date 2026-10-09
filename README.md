# ML Production Project — Breast Cancer Classification

## Project goal

Turn a notebook experiment into a clean, reproducible Python application. The project
loads the Breast Cancer Wisconsin dataset, trains a classification pipeline that
predicts whether a tumor is **malignant** or **benign**, evaluates it, saves the fitted
pipeline, and serves predictions by reloading that artifact — with no retraining.

## Dataset

[Breast Cancer Wisconsin (Diagnostic)](https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset)
(`notebooks/kaggle/breast-cancer.csv`):

- 569 rows × 32 columns
- Target: `diagnosis` — `M` (malignant, 212) / `B` (benign, 357)
- 30 numeric features (mean, standard error, and worst values of tumor measurements)
- `id` is dropped (identifier, not a feature); no missing values

## ML pipeline

```
Load CSV → Validate columns → Train/test split (80/20, stratified, seeded)
         → imblearn Pipeline fit: SMOTETomek → StandardScaler → LogisticRegression
         → Evaluate on held-out test set → Save pipeline (joblib)
         → Reload artifact → Predict (no retraining)
```

- **Leakage prevention:** the entire pipeline lives in a single
  `imblearn.pipeline.Pipeline`. SMOTETomek (oversampling + Tomek-link cleaning) and
  `StandardScaler` only learn during `fit` on training data; `predict` skips the
  sampler and applies the already-fitted scaler + model.
- **Class balancing:** the dataset is imbalanced (≈63/37); SMOTETomek balances the
  training fold only, so the test set keeps its natural distribution.

## Model choice

**Logistic Regression** — the assignment asks for one straightforward model, and it is
the notebook's primary baseline. It trains in well under a second, is easy to
interpret, and reaches ~98% accuracy on this dataset. The notebook
(`notebooks/01_exploration.ipynb`) additionally compares Random Forest, XGBoost,
Decision Tree, Naive Bayes, KNN, and SVC as the experimentation record; the production
app ships only the Logistic Regression pipeline.

## Folder structure

```
ml-production-project/
|-- notebooks/
|   `-- 01_exploration.ipynb   # Stage A: exploration, EDA, model comparison
|-- production_app/
|   |-- __init__.py
|   |-- config.py              # settings from environment variables
|   |-- logging_config.py      # one shared logging setup
|   |-- data.py                # load, validate, split
|   |-- preprocessing.py       # SMOTETomek + StandardScaler + LogisticRegression pipeline
|   |-- training.py            # fit
|   |-- evaluation.py          # metrics + report
|   |-- model_service.py       # joblib save / load / predict
|   `-- main.py                # orchestration
|-- models/
|   |-- model_pipeline.joblib  # saved fitted pipeline (recreate with train)
|   `-- joblib_models/         # model pickles produced by the notebook
|-- run.py                     # entry point: train | predict
|-- .env.example
|-- .gitignore
|-- .pre-commit-config.yaml
|-- pyproject.toml
|-- uv.lock
`-- README.md
```

## Evaluation results

Held-out test set (20%, stratified, `RANDOM_STATE=42`):

| Metric   | Value  |
|----------|--------|
| Accuracy | 0.9737 |
| Precision (malignant) | 0.9756 |
| Recall (malignant)    | 0.9524 |
| F1 (malignant)        | 0.9639 |

Confusion matrix (`[[TN, FP], [FN, TP]]`):

```
[[71  1]
 [ 2 40]]
```

One false positive and two false negatives out of 114 test samples.

## Setup

Prerequisites: [uv](https://docs.astral.sh/uv/) (installs and manages Python 3.14
for you) and Git.

```bash
git clone <repository-url>
cd ML-Production_Project
uv sync                 # creates .venv, installs dependencies from uv.lock
```

Optional configuration — copy the example and edit values:

```bash
cp .env.example .env
```

The app runs with safe defaults even without a `.env` file.

## Configuration

| Variable       | Default                          | Purpose                          |
|----------------|----------------------------------|----------------------------------|
| `DATA_PATH`    | `notebooks/kaggle/breast-cancer.csv` | Input dataset location       |
| `MODEL_PATH`   | `models/model_pipeline.joblib`   | Saved pipeline artifact          |
| `LOG_LEVEL`    | `INFO`                           | Logging verbosity                |
| `ENVIRONMENT`  | `development`                    | Free-form environment label      |
| `TEST_SIZE`    | `0.2`                            | Test fraction of the split       |
| `RANDOM_STATE` | `42`                             | Seed for split + SMOTETomek      |

## Usage

Train, evaluate, and save the pipeline:

```bash
uv run run.py train
```

Predict with the saved artifact (loads the model, predicts the first 5 rows of the
dataset; no retraining happens):

```bash
uv run run.py predict
```

Predict on your own CSV with the same feature columns:

```bash
uv run run.py predict --input path/to/data.csv
```

A missing artifact or invalid input is logged as an error and exits with code 1.

## Code quality

```bash
uv run ruff check .              # lint (E, F, I rules)
uv run black .                   # format
uv run pre-commit run --all-files  # run all configured hooks
```

Pre-commit runs `ruff`, `black`, and basic file checks (trailing whitespace,
end-of-file, YAML/TOML validity) on every commit after a one-time
`uv run pre-commit install`.
