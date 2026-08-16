from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_check():

    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"


def test_prediction():

    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert "class_id" in data
    assert "species" in data
    assert "probabilities" in data

    assert data["species"] in [
        "setosa",
        "versicolor",
        "virginica",
    ]

    assert len(data["probabilities"]) == 3


def test_invalid_input():

    payload = {
        "sepal_length": -1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422