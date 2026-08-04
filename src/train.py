import pandas as pd
import mlflow
import mlflow.catboost
from catboost import CatBoostRegressor
from pathlib import Path
import yaml
from sklearn.metrics import mean_squared_error, mean_absolute_error

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

train_path = params["data"]["train"]
test_path = params["data"]["test"]

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

X_train = df_train.drop(columns=["SalePrice"])
y_train = df_train["SalePrice"]

X_test = df_test.drop(columns=["SalePrice"])
y_test = df_test["SalePrice"]

cat_features = [i for i, col in enumerate(X_train.columns) if X_train[col].dtype == "object"]

mlflow.set_experiment("ames-catboost")

with mlflow.start_run():

    mlflow.log_param("depth", params["model"]["depth"])
    mlflow.log_param("learning_rate", params["model"]["learning_rate"])
    mlflow.log_param("iterations", params["model"]["iterations"])

    model = CatBoostRegressor(
        depth=params["model"]["depth"],
        learning_rate=params["model"]["learning_rate"],
        iterations=params["model"]["iterations"],
        loss_function="RMSE",
        verbose=False,
    )

    model.fit(X_train, y_train, cat_features=cat_features)

    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    rmse = mse**0.5
    mae = mean_absolute_error(y_test, preds)

    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)

    model_path = Path("C:/mlopsproject/models/catboost_model.cbm")
    model.save_model(str(model_path))

    mlflow.catboost.log_model(model, artifact_path="model")

print("Training complete.")
