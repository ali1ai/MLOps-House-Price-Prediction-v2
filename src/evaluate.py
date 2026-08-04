import pandas as pd
from catboost import CatBoostRegressor
from pathlib import Path
import yaml
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

test_path = params["data"]["test"]

df_test = pd.read_csv(test_path)

X_test = df_test.drop(columns=["SalePrice"])
y_test = df_test["SalePrice"]

model_path = Path("C:/mlopsproject/models/catboost_model.cbm")
model = CatBoostRegressor()
model.load_model(str(model_path))

preds = model.predict(X_test)

mse = mean_squared_error(y_test, preds)
rmse = mse ** 0.5
mae = mean_absolute_error(y_test, preds)

metrics = {
    "rmse": rmse,
    "mae": mae
}

with open("reports/metrics.json", "w") as f:
    json.dump(metrics, f)

print("Evaluation complete.")
