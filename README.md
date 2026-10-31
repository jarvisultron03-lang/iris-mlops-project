# Iris MLOps Project

A learning project demonstrating a production-style machine learning workflow using:

- Python
- Scikit-Learn
- DVC
- MLflow
- MLflow Model Registry
- FastAPI
- Pytest
- Git

## Architecture

Git
    |
    +-- Code
    |
    +-- DVC metadata
           |
           v
      Iris dataset
           |
           v
        train.py
           |
           v
        MLflow
           |
           +-- Parameters
           +-- Metrics
           +-- Experiment
           +-- Registered Model
                    |
                    v
             IrisClassifier
                    |
                 champion
                    |
                    v
                 FastAPI
                    |
                 /predict

## Setup

Create a virtual environment:

python -m venv .venv

Activate it on Windows:

.venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

## Create the dataset

python src/prepare_data.py

## Add dataset to DVC

dvc add data/raw/iris.csv

## Run training

python src/train.py

The training script:

1. Loads the DVC-controlled dataset.
2. Splits the data.
3. Trains a Random Forest.
4. Logs parameters to MLflow.
5. Logs metrics to MLflow.
6. Logs the model to MLflow.
7. Registers the model as IrisClassifier.
8. Assigns the champion alias if accuracy passes the threshold.

## Start MLflow UI

mlflow ui --backend-store-uri sqlite:///mlflow.db

Then open:

http://127.0.0.1:5000

## Start FastAPI

uvicorn app.main:app --reload

Then open:

http://127.0.0.1:8000/docs

## Run tests

pytest

## Run through MLflow Projects

mlflow run . -e train

## Run through DVC

dvc repro