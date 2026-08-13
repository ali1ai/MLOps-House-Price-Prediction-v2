import pandas as pd

import app.model as model_module
from app.model import FEATURES, predict_sale_price
from src.preprocessing import preprocess_features


class FakeModel:
    def __init__(self, feature_names):
        self.feature_names_ = feature_names

    def predict(self, df):
        assert list(df.columns) == self.feature_names_
        return [123456.0]


def test_features_list_not_empty():
    assert isinstance(FEATURES, list)
    assert len(FEATURES) > 0


def test_prediction_output_type(monkeypatch):
    payload = {feature: 1 for feature in FEATURES}

    raw = pd.DataFrame([{feature: payload[feature] for feature in FEATURES}])
    raw = raw.rename(columns=model_module.API_TO_MODEL_COLUMNS)
    processed = preprocess_features(raw)

    fake_model = FakeModel(list(processed.columns))
    monkeypatch.setattr(model_module, "get_model", lambda: fake_model)

    prediction = predict_sale_price(payload)

    assert isinstance(prediction, float)
    assert prediction == 123456.0
