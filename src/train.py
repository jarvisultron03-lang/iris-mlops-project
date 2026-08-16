from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml

from mlflow.models import infer_signature

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


DATA_PATH = Path("data/raw/iris.csv")
PARAMS_PATH = Path("params.yaml")

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MLFLOW_EXPERIMENT_NAME = "iris-classification"

REGISTERED_MODEL_NAME = "IrisClassifier"


def load_params():
    with open(PARAMS_PATH, "r") as file:
        return yaml.safe_load(file)


def load_data():
    df = pd.read_csv(DATA_PATH)

    X = df[
        [
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width",
        ]
    ]

    y = df["target"]

    return X, y


def train_model():

    # --------------------------------------------------
    # 1. Load parameters
    # --------------------------------------------------

    params = load_params()

    random_state = params["random_state"]
    test_size = params["test_size"]

    n_estimators = params["model"]["n_estimators"]
    max_depth = params["model"]["max_depth"]
    min_samples_leaf = params["model"]["min_samples_leaf"]

    min_accuracy = params["evaluation"]["min_accuracy"]

    # --------------------------------------------------
    # 2. Configure MLflow
    # --------------------------------------------------

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    mlflow.set_experiment(
        MLFLOW_EXPERIMENT_NAME
    )

    # --------------------------------------------------
    # 3. Load data
    # --------------------------------------------------

    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # --------------------------------------------------
    # 4. Start MLflow run
    # --------------------------------------------------

    with mlflow.start_run() as run:

        print(
            f"MLflow run ID: {run.info.run_id}"
        )

        # --------------------------------------------------
        # 5. Log parameters
        # --------------------------------------------------

        mlflow.log_params(
            {
                "random_state": random_state,
                "test_size": test_size,
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_leaf": min_samples_leaf,
            }
        )

        # --------------------------------------------------
        # 6. Create model
        # --------------------------------------------------

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )

        # --------------------------------------------------
        # 7. Train
        # --------------------------------------------------

        model.fit(
            X_train,
            y_train,
        )

        # --------------------------------------------------
        # 8. Evaluate
        # --------------------------------------------------

        predictions = model.predict(
            X_test
        )

        accuracy = accuracy_score(
            y_test,
            predictions,
        )

        print(
            f"Validation accuracy: {accuracy:.4f}"
        )

        mlflow.log_metric(
            "accuracy",
            accuracy,
        )

        # --------------------------------------------------
        # 9. Create model signature
        # --------------------------------------------------

        signature = infer_signature(
            X_train,
            model.predict(X_train),
        )

        # --------------------------------------------------
        # 10. Log and register model
        # --------------------------------------------------

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="iris_model",
            signature=signature,
            registered_model_name=REGISTERED_MODEL_NAME,
        )

        print(
            f"MLflow model URI: {model_info.model_uri}"
        )

        # --------------------------------------------------
        # 11. Decide whether model is deployable
        # --------------------------------------------------

        if accuracy >= min_accuracy:

            print(
                "Model passed deployment threshold."
            )

            # We need the registered version.
            client = mlflow.MlflowClient()

           

            # Find the version created by this run.
            versions = client.search_model_versions(
                f"name='{REGISTERED_MODEL_NAME}'"
            )

            current_version = None

            for version in versions:
                if version.run_id == run.info.run_id:
                    current_version = version.version
                    break

            if current_version is None:
                raise RuntimeError(
                    "Could not find registered model version."
                )

            # Assign the champion alias.
            client.set_registered_model_alias(
                name=REGISTERED_MODEL_NAME,
                alias="champion",
                version=current_version,
            )

            # Add useful metadata.
            client.set_model_version_tag(
                name=REGISTERED_MODEL_NAME,
                version=current_version,
                key="validation_status",
                value="approved",
            )

            client.set_model_version_tag(
                name=REGISTERED_MODEL_NAME,
                version=current_version,
                key="accuracy",
                value=str(accuracy),
            )

            print(
                f"Version {current_version} "
                f"assigned to champion."
            )

        else:

            print(
                "Model did NOT pass deployment threshold."
            )


if __name__ == "__main__":
    train_model()