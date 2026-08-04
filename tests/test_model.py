from app.model import model, predict_sale_price, FEATURES

def test_model_loaded():
    # CatBoost model should be loaded once at import time
    assert model is not None

def test_features_list_not_empty():
    assert isinstance(FEATURES, list)
    assert len(FEATURES) > 0

def test_prediction_output_type():
    sample = {feature: 1 for feature in FEATURES}
    pred = predict_sale_price(sample)
    assert isinstance(pred, float)
