import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path
import json

# -----------------------------
# 1. Load baseline + new data
# -----------------------------
BASELINE_PATH = Path("data/baseline.csv")
NEW_DATA_PATH = Path("data/new_batch.csv")
OLD_MODEL_PATH = Path("models/catboost_model.cbm")
NEW_MODEL_PATH = Path("models/catboost_model_new.cbm")
METRICS_PATH = Path("reports/retraining_metrics.json")

baseline = pd.read_csv(BASELINE_PATH)
new_data = pd.read_csv(NEW_DATA_PATH)

# -----------------------------
# 2. Split features + target
# -----------------------------
TARGET = "SalePrice"   # adjust if your target column is different

X_base = baseline.drop(columns=[TARGET])
y_base = baseline[TARGET]

X_new = new_data.drop(columns=[TARGET])
y_new = new_data[TARGET]

# -----------------------------
# 3. Load old model
# -----------------------------
old_model = CatBoostRegressor()
old_model.load_model(str(OLD_MODEL_PATH))

# Evaluate old model on new data
old_preds = old_model.predict(X_new)
old_rmse = mean_squared_error(y_new, old_preds, squared=False)
old_mae = mean_absolute_error(y_new, old_preds)
old_r2 = r2_score(y_new, old_preds)

# -----------------------------
# 4. Train new model
# -----------------------------
new_model = CatBoostRegressor(
    iterations=500,
    learning_rate=0.05,
    depth=8,
    loss_function="RMSE",
    verbose=False
)

new_model.fit(X_new, y_new)

# Evaluate new model
new_preds = new_model.predict(X_new)
new_rmse = mean_squared_error(y_new, new_preds, squared=False)
new_mae = mean_absolute_error(y_new, new_preds)
new_r2 = r2_score(y_new, new_preds)

# -----------------------------
# 5. Compare models
# -----------------------------
metrics = {
    "old_model": {
        "rmse": old_rmse,
        "mae": old_mae,
        "r2": old_r2
    },
    "new_model": {
        "rmse": new_rmse,
        "mae": new_mae,
        "r2": new_r2
    }
}

# Save metrics
METRICS_PATH.parent.mkdir(exist_ok=True)
with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=4)

print("Old model RMSE:", old_rmse)
print("New model RMSE:", new_rmse)

# -----------------------------
# 6. Save better model
# -----------------------------
if new_rmse < old_rmse:
    new_model.save_model(str(OLD_MODEL_PATH))
    print("New model is better. Model updated.")
else:
    print("Old model is better. No update performed.")
