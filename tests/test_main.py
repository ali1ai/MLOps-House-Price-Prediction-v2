import pandas as pd
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_predict_endpoint_success():
    df = pd.read_csv("data/raw/train.csv")

    row = df.iloc[0].drop(labels=["Id", "SalePrice"], errors="ignore")

    row = row.rename(
        {
            "1stFlrSF": "FirstFlrSF",
            "2ndFlrSF": "SecondFlrSF",
            "3SsnPorch": "ThreeSsnPorch",
        }
    )

    payload = {
        key: (None if pd.isna(value) else value.item() if hasattr(value, "item") else value)
        for key, value in row.items()
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200, response.text
    assert "SalePrice" in response.json()
    assert isinstance(response.json()["SalePrice"], (int, float))


def test_predict_endpoint_missing_field():
    payload = {
        "MSZoning": "RL",
        "LotArea": 8450,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
