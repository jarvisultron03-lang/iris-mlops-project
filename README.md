# Iris MLOps Project

An end-to-end Iris classification project that trains a Random Forest model, tracks experiments and registers approved models with MLflow, and serves predictions through a FastAPI application.

## What it does

- Loads the Iris dataset and writes it to `data/raw/iris.csv`.
- Uses DVC to run the training pipeline and track its inputs and parameters.
- Logs training parameters and accuracy to MLflow.
- Registers each trained model as `IrisClassifier`. A model that meets the configured accuracy threshold is assigned the `champion` alias.
- Serves predictions from the champion model through a REST API.
- Runs API integration tests with pytest.

## Requirements

- Python 3.13
- [uv](https://docs.astral.sh/uv/)

## Setup

From the project root, install the project and development dependencies:

```shell
uv sync
```

The Iris CSV is DVC-managed and may not be present after a fresh clone. Generate it locally with the included preparation script:

```shell
uv run python src/prepare_data.py
```

This writes `data/raw/iris.csv` using scikit-learn's built-in Iris dataset.

## Train the model

Run the DVC training pipeline:

```shell
uv run dvc repro
```

The pipeline runs `src/train.py`. It reads `params.yaml`, splits the data into training and validation sets, trains a `RandomForestClassifier`, and logs its parameters, validation accuracy, and model to MLflow. The default experiment is `iris-classification`; the tracking database is `mlflow.db`, and model artifacts are stored under `mlruns/`.

The model is registered as `IrisClassifier`. When its validation accuracy is at least `evaluation.min_accuracy`, training assigns the new model version the `champion` alias and marks it as approved. The API requires a registered model with this alias, so a successful training run must happen before starting the API or running its tests.

## Run the API

Start the service from the project root:

```shell
uv run uvicorn app.main:app --reload
```

The API loads `models:/IrisClassifier@champion` during startup and uses the local MLflow tracking database. Visit `http://127.0.0.1:8000/docs` for the interactive Swagger UI.

### Endpoints

- `GET /` — health status and model-loading information.
- `POST /predict` — classify a flower from its four measurements in centimeters. Each value must be between 0 and 10.

Example request:

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

The prediction response includes the integer `class_id`, the `species` (`setosa`, `versicolor`, or `virginica`), and the probabilities for all three classes.

## Run tests

After setup, data preparation, and a successful training run:

```shell
uv run python -m pytest tests -v
```

The tests exercise the health endpoint, a valid prediction, and validation of out-of-range input.

## Configuration

Edit `params.yaml` to change the random seed, validation split, Random Forest settings, or minimum accuracy required for promotion to `champion`.

## Project structure

```text
app/
  main.py              FastAPI application and prediction endpoints
data/raw/
  iris.csv.dvc         DVC metadata for the Iris dataset
src/
  prepare_data.py      Generate the Iris CSV
  train.py             Train, evaluate, and register the model
tests/
  test_api.py          API integration tests
dvc.yaml               DVC training pipeline
params.yaml             Training and promotion parameters
MLproject               MLflow project entry point
```