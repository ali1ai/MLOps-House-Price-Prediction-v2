import pytest
from pydantic import ValidationError

from app.main import HouseFeatures, root


def test_root():
    response = root()
    assert response["message"] == "Ames Housing Price Prediction API is running!"


def test_api_schema_rejects_missing_fields():
    with pytest.raises(ValidationError):
        HouseFeatures(
            MSZoning="RL",
            LotArea=8450,
        )
