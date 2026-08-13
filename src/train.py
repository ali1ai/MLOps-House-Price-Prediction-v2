from pathlib import Path

import mlflow
import mlflow.catboost
import pandas as pd
import yaml
from catboost import CatBoostRegressor
from mlflow.models import infer_signature
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.preprocessing import (
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
)


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


train_path = project_path(params["data"]["train"])

validation_path = project_path(params["data"]["validation"])

model_path = project_path(params["paths"]["model"])


# ---------------------------------------------------------
# Load prepared datasets
# ---------------------------------------------------------

df_train = pd.read_csv(train_path)
df_validation = pd.read_csv(validation_path)


if TARGET_COLUMN not in df_train.columns:
    raise ValueError(f"{TARGET_COLUMN} is missing from training data.")

if TARGET_COLUMN not in df_validation.columns:
    raise ValueError(f"{TARGET_COLUMN} is missing from validation data.")


X_train = df_train.drop(columns=[TARGET_COLUMN])

y_train = df_train[TARGET_COLUMN]


X_validation = df_validation.drop(columns=[TARGET_COLUMN])

y_validation = df_validation[TARGET_COLUMN]


# ---------------------------------------------------------
# Validate feature schema
# ---------------------------------------------------------

if list(X_train.columns) != list(X_validation.columns):
    raise ValueError("Training and validation feature schemas " "do not match.")


categorical_features = [column for column in CATEGORICAL_FEATURES if column in X_train.columns]


# ---------------------------------------------------------
# MLflow configuration
# ---------------------------------------------------------

mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])

mlflow.set_experiment(params["mlflow"]["experiment_name"])


# ---------------------------------------------------------
# Train and validate
# ---------------------------------------------------------

with mlflow.start_run(run_name=params["mlflow"]["run_name"]) as run:

    model_parameters = {
        "depth": params["model"]["depth"],
        "learning_rate": (params["model"]["learning_rate"]),
        "iterations": (params["model"]["iterations"]),
        "random_seed": (params["model"]["random_seed"]),
    }

    mlflow.log_params(model_parameters)

    mlflow.log_param(
        "training_rows",
        len(X_train),
    )

    mlflow.log_param(
        "validation_rows",
        len(X_validation),
    )

    mlflow.log_param(
        "feature_count",
        X_train.shape[1],
    )

    mlflow.set_tag(
        "evaluation_dataset",
        "validation",
    )

    model = CatBoostRegressor(
        **model_parameters,
        loss_function="RMSE",
        verbose=False,
    )

    model.fit(
        X_train,
        y_train,
        cat_features=categorical_features,
    )

    # -----------------------------------------------------
    # Validation evaluation
    # -----------------------------------------------------

    validation_predictions = model.predict(X_validation)

    validation_mse = mean_squared_error(
        y_validation,
        validation_predictions,
    )

    validation_rmse = validation_mse**0.5

    validation_mae = mean_absolute_error(
        y_validation,
        validation_predictions,
    )

    validation_r2 = r2_score(
        y_validation,
        validation_predictions,
    )

    validation_metrics = {
        "validation_rmse": validation_rmse,
        "validation_mae": validation_mae,
        "validation_r2": validation_r2,
    }

    mlflow.log_metrics(validation_metrics)

    # -----------------------------------------------------
    # Save native CatBoost model
    # -----------------------------------------------------

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save_model(str(model_path))

    # -----------------------------------------------------
    # Log model to MLflow
    # -----------------------------------------------------

    signature = infer_signature(
        X_validation,
        validation_predictions,
    )

    input_example = X_validation.head(3).copy()

    mlflow.catboost.log_model(
        model,
        name="model",
        signature=signature,
        input_example=input_example,
    )

    run_id = run.info.run_id


# ---------------------------------------------------------
# Console summary
# ---------------------------------------------------------

print("Training complete.")
print(f"MLflow run ID:    {run_id}")
print(f"Training rows:    {len(X_train)}")
print(f"Validation rows:  " f"{len(X_validation)}")
print(f"Model features:   " f"{X_train.shape[1]}")
print(f"Validation RMSE:  " f"{validation_rmse:.4f}")
print(f"Validation MAE:   " f"{validation_mae:.4f}")
print(f"Validation R2:    " f"{validation_r2:.4f}")
print(f"Model saved to:   " f"{model_path}")
