import json
from pathlib import Path

import pandas as pd
import yaml
from catboost import CatBoostRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.preprocessing import TARGET_COLUMN


# ---------------------------------------------------------
# Project configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARAMS_PATH = PROJECT_ROOT / "params.yaml"


with open(
    PARAMS_PATH,
    "r",
    encoding="utf-8",
) as f:
    params = yaml.safe_load(f)


def project_path(path_value: str) -> Path:
    """
    Resolve a project-relative path from params.yaml.
    """
    return PROJECT_ROOT / path_value


test_path = project_path(
    params["data"]["test"]
)

model_path = project_path(
    params["paths"]["model"]
)

metrics_path = project_path(
    params["paths"]["metrics"]
)


# ---------------------------------------------------------
# Load final test data
# ---------------------------------------------------------

df_test = pd.read_csv(test_path)


if TARGET_COLUMN not in df_test.columns:
    raise ValueError(
        f"{TARGET_COLUMN} is missing from test data."
    )


X_test = df_test.drop(
    columns=[TARGET_COLUMN]
)

y_test = df_test[TARGET_COLUMN]


# ---------------------------------------------------------
# Load trained model
# ---------------------------------------------------------

if not model_path.exists():
    raise FileNotFoundError(
        f"Model was not found: {model_path}"
    )


model = CatBoostRegressor()

model.load_model(
    str(model_path)
)


# ---------------------------------------------------------
# Validate model/test feature compatibility
# ---------------------------------------------------------

model_features = list(
    model.feature_names_
)


missing_features = [
    feature
    for feature in model_features
    if feature not in X_test.columns
]


if missing_features:
    raise ValueError(
        "Test data is missing model features: "
        f"{missing_features}"
    )


unexpected_features = [
    feature
    for feature in X_test.columns
    if feature not in model_features
]


if unexpected_features:
    raise ValueError(
        "Test data contains unexpected features: "
        f"{unexpected_features}"
    )


# Match the exact trained-model feature order.
X_test = X_test[model_features]


# ---------------------------------------------------------
# Final untouched-test evaluation
# ---------------------------------------------------------

test_predictions = model.predict(
    X_test
)


test_mse = mean_squared_error(
    y_test,
    test_predictions,
)

test_rmse = test_mse ** 0.5

test_mae = mean_absolute_error(
    y_test,
    test_predictions,
)

test_r2 = r2_score(
    y_test,
    test_predictions,
)


metrics = {
    "rmse": float(test_rmse),
    "mae": float(test_mae),
    "r2": float(test_r2),
    "test_rows": int(len(X_test)),
    "feature_count": int(
        X_test.shape[1]
    ),
}


# ---------------------------------------------------------
# Save DVC metrics
# ---------------------------------------------------------

metrics_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)


with open(
    metrics_path,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        metrics,
        f,
        indent=4,
    )


# ---------------------------------------------------------
# Console summary
# ---------------------------------------------------------

print(
    "Final test evaluation complete."
)

print(
    f"Test rows:      "
    f"{len(X_test)}"
)

print(
    f"Model features: "
    f"{X_test.shape[1]}"
)

print(
    f"Test RMSE:      "
    f"{test_rmse:.4f}"
)

print(
    f"Test MAE:       "
    f"{test_mae:.4f}"
)

print(
    f"Test R2:        "
    f"{test_r2:.4f}"
)

print(
    f"Metrics saved:  "
    f"{metrics_path}"
)