from pathlib import Path
import json
import os

import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

BASELINE_PATH = Path("data/baseline.csv")
NEW_DATA_PATH = Path("data/new_batch.csv")
OLD_MODEL_PATH = Path("models/catboost_model.cbm")

DRIFT_STATUS_PATH = Path("reports/drift_status.json")
METRICS_PATH = Path("reports/retraining_metrics.json")

TARGET = "SalePrice"


# --------------------------------------------------
# 2. Check drift status / forced validation
# --------------------------------------------------

if not DRIFT_STATUS_PATH.exists():
    raise FileNotFoundError(
        "Drift status file not found. Run monitoring/drift_monitor.py first."
    )

with open(DRIFT_STATUS_PATH, "r") as file:
    drift_status = json.load(file)

drift_detected = drift_status.get("drift_detected", False)

# Manual validation option from GitHub Actions.
# Normal scheduled runs leave this false.
force_retrain = os.getenv("FORCE_RETRAIN", "false").strip().lower() == "true"

print("Drift detected:", drift_detected)
print("Forced validation run:", force_retrain)

if not drift_detected and not force_retrain:
    print("No drift detected.")
    print("Retraining skipped.")
    raise SystemExit(0)


# --------------------------------------------------
# 3. Load data
# --------------------------------------------------

if force_retrain and not drift_detected:
    print("\nControlled validation run. Starting candidate retraining.")
else:
    print("\nDrift detected. Starting retraining.")

baseline = pd.read_csv(BASELINE_PATH)
new_data = pd.read_csv(NEW_DATA_PATH)

print("Baseline shape:", baseline.shape)
print("New batch shape:", new_data.shape)


# --------------------------------------------------
# 4. Load current deployed model
# --------------------------------------------------

if not OLD_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Current model not found at {OLD_MODEL_PATH}. "
        "Restore the DVC model artifact before retraining."
    )

old_model = CatBoostRegressor()
old_model.load_model(str(OLD_MODEL_PATH))

MODEL_FEATURES = list(old_model.feature_names_)

print("Current model expects", len(MODEL_FEATURES), "features.")


# --------------------------------------------------
# 5. Prepare features
# --------------------------------------------------


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Remove columns that are not model inputs.
    df = df.drop(
        columns=["Id", "SalePrice"],
        errors="ignore",
    )

    # Create engineered missing-value indicators used
    # when the deployed CatBoost model was trained.
    df["LotFrontage_missing"] = df["LotFrontage"].isna().astype(int)
    df["MasVnrArea_missing"] = df["MasVnrArea"].isna().astype(int)

    # Fill missing categorical values.
    categorical_columns = df.select_dtypes(include=["object"]).columns

    for column in categorical_columns:
        df[column] = df[column].fillna("Missing").astype(str)

    # Check that every feature expected by the model exists.
    missing_features = [
        feature for feature in MODEL_FEATURES if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing features required by model: {missing_features}"
        )

    # Use the exact feature names and order expected
    # by the deployed model.
    df = df[MODEL_FEATURES]

    return df


X = prepare_features(baseline)
y = baseline[TARGET]


# --------------------------------------------------
# 6. Train / validation split
# --------------------------------------------------

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)


# --------------------------------------------------
# 7. Evaluate current deployed model
# --------------------------------------------------

old_preds = old_model.predict(X_val)

old_rmse = mean_squared_error(y_val, old_preds) ** 0.5
old_mae = mean_absolute_error(y_val, old_preds)
old_r2 = r2_score(y_val, old_preds)

print("\nCurrent model:")
print("RMSE:", old_rmse)
print("MAE:", old_mae)
print("R2:", old_r2)


# --------------------------------------------------
# 8. Train candidate model
# --------------------------------------------------

cat_features = X_train.select_dtypes(include=["object"]).columns.tolist()

new_model = CatBoostRegressor(
    iterations=500,
    learning_rate=0.05,
    depth=8,
    loss_function="RMSE",
    verbose=False,
    random_seed=42,
)

new_model.fit(
    X_train,
    y_train,
    cat_features=cat_features,
)


# --------------------------------------------------
# 9. Evaluate candidate model
# --------------------------------------------------

new_preds = new_model.predict(X_val)

new_rmse = mean_squared_error(y_val, new_preds) ** 0.5
new_mae = mean_absolute_error(y_val, new_preds)
new_r2 = r2_score(y_val, new_preds)

print("\nCandidate model:")
print("RMSE:", new_rmse)
print("MAE:", new_mae)
print("R2:", new_r2)


# --------------------------------------------------
# 10. Save model-comparison metrics
# --------------------------------------------------

candidate_better = new_rmse < old_rmse

metrics = {
    "drift_detected": drift_detected,
    "forced_validation": force_retrain,
    "candidate_better": candidate_better,
    "current_model": {
        "rmse": float(old_rmse),
        "mae": float(old_mae),
        "r2": float(old_r2),
    },
    "candidate_model": {
        "rmse": float(new_rmse),
        "mae": float(new_mae),
        "r2": float(new_r2),
    },
}

METRICS_PATH.parent.mkdir(exist_ok=True)

with open(METRICS_PATH, "w") as file:
    json.dump(
        metrics,
        file,
        indent=4,
    )

print("\nComparison metrics saved to:", METRICS_PATH)


# --------------------------------------------------
# 11. Promotion decision
# --------------------------------------------------

if force_retrain:
    print("\nControlled validation completed.")

    if candidate_better:
        print("Candidate model performed better than the current model.")
    else:
        print("Current model performed better than or equal to the candidate.")

    print("No model replacement performed during forced validation.")

elif candidate_better:
    new_model.save_model(str(OLD_MODEL_PATH))

    print("\nCandidate model is better.")
    print("Current deployed model has been replaced.")

else:
    print("\nCurrent model is better.")
    print("No model replacement performed.")