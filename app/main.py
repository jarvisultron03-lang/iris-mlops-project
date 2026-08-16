from contextlib import asynccontextmanager
from typing import Dict, List

import mlflow
import mlflow.sklearn

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"

MODEL_NAME = "IrisClassifier"
MODEL_ALIAS = "champion"

TARGET_NAMES = [
    "setosa",
    "versicolor",
    "virginica",
]


ml_models: Dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):

    print("Connecting to MLflow...")

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    model_uri = (
        f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    )

    print(
        f"Loading model from: {model_uri}"
    )

    try:

        model = mlflow.sklearn.load_model(
            model_uri
        )

        ml_models["iris_classifier"] = model

        print(
            "Champion model loaded successfully."
        )

    except Exception as exc:

        raise RuntimeError(
            f"Could not load MLflow model: {exc}"
        ) from exc

    yield

    ml_models.clear()

    print("Model unloaded.")


app = FastAPI(
    title="Iris Flower Classification API",
    description=(
        "FastAPI service that loads the "
        "champion Iris model from MLflow."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


class IrisFeatures(BaseModel):

    sepal_length: float = Field(
        ...,
        ge=0.0,
        le=10.0,
        description="Sepal length in cm",
    )

    sepal_width: float = Field(
        ...,
        ge=0.0,
        le=10.0,
        description="Sepal width in cm",
    )

    petal_length: float = Field(
        ...,
        ge=0.0,
        le=10.0,
        description="Petal length in cm",
    )

    petal_width: float = Field(
        ...,
        ge=0.0,
        le=10.0,
        description="Petal width in cm",
    )


class PredictionResponse(BaseModel):

    class_id: int = Field(
        ...,
        description="Predicted class ID",
    )

    species: str = Field(
        ...,
        description="Predicted Iris species",
    )

    probabilities: List[float] = Field(
        ...,
        description="Probability for each class",
    )


@app.get("/")
def health_check():

    return {
        "status": "ok",
        "model_loaded": (
            "iris_classifier" in ml_models
        ),
        "model": MODEL_NAME,
        "alias": MODEL_ALIAS,
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict_species(
    data: IrisFeatures,
):

    model = ml_models.get(
        "iris_classifier"
    )

    if model is None:

        raise HTTPException(
            status_code=500,
            detail="Model is not loaded.",
        )

    input_data = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width,
    ]]

    prediction_id = int(
        model.predict(input_data)[0]
    )

    probabilities = (
        model
        .predict_proba(input_data)[0]
        .tolist()
    )

    return PredictionResponse(
        class_id=prediction_id,
        species=TARGET_NAMES[prediction_id],
        probabilities=probabilities,
    )