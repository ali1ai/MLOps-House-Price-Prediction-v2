from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_predict_endpoint_success():
    payload = {
        "MSZoning": "RL",
        "LotArea": 8450,
        "OverallQual": 7,
        "OverallCond": 5,
        "YearBuilt": 2003,
        "GrLivArea": 1710
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert isinstance(response.json()["prediction"], float)


def test_predict_endpoint_missing_field():
    payload = {
        "MSZoning": "RL",
        "LotArea": 8450,
        # Missing required fields
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # FastAPI validation error
