from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
from catboost import CatBoostRegressor

from src.preprocessing import preprocess_features


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "catboost_model.cbm"


# ---------------------------------------------------------
# API feature schema
#
# These are API-friendly names exposed through FastAPI.
# Some Ames feature names begin with numbers and therefore
# are mapped to Python-friendly names in the API layer.
# ---------------------------------------------------------

FEATURES = [
    "MSSubClass",
    "MSZoning",
    "LotFrontage",
    "LotArea",
    "Street",
    "Alley",
    "LotShape",
    "LandContour",
    "Utilities",
    "LotConfig",
    "LandSlope",
    "Neighborhood",
    "Condition1",
    "Condition2",
    "BldgType",
    "HouseStyle",
    "OverallQual",
    "OverallCond",
    "YearBuilt",
    "YearRemodAdd",
    "RoofStyle",
    "RoofMatl",
    "Exterior1st",
    "Exterior2nd",
    "MasVnrType",
    "MasVnrArea",
    "ExterQual",
    "ExterCond",
    "Foundation",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinSF1",
    "BsmtFinType2",
    "BsmtFinSF2",
    "BsmtUnfSF",
    "TotalBsmtSF",
    "Heating",
    "HeatingQC",
    "CentralAir",
    "Electrical",
    "FirstFlrSF",
    "SecondFlrSF",
    "LowQualFinSF",
    "GrLivArea",
    "BsmtFullBath",
    "BsmtHalfBath",
    "FullBath",
    "HalfBath",
    "BedroomAbvGr",
    "KitchenAbvGr",
    "KitchenQual",
    "TotRmsAbvGrd",
    "Functional",
    "Fireplaces",
    "FireplaceQu",
    "GarageType",
    "GarageYrBlt",
    "GarageFinish",
    "GarageCars",
    "GarageArea",
    "GarageQual",
    "GarageCond",
    "PavedDrive",
    "WoodDeckSF",
    "OpenPorchSF",
    "EnclosedPorch",
    "ThreeSsnPorch",
    "ScreenPorch",
    "PoolArea",
    "PoolQC",
    "Fence",
    "MiscFeature",
    "MiscVal",
    "MoSold",
    "YrSold",
    "SaleType",
    "SaleCondition",
]


# ---------------------------------------------------------
# API name -> original Ames name
# ---------------------------------------------------------

API_TO_MODEL_COLUMNS = {
    "FirstFlrSF": "1stFlrSF",
    "SecondFlrSF": "2ndFlrSF",
    "ThreeSsnPorch": "3SsnPorch",
}


# ---------------------------------------------------------
# Model loading
#
# Lazy loading avoids failing merely by importing this
# module and makes testing/deployment behavior cleaner.
# ---------------------------------------------------------


@lru_cache(maxsize=1)
def get_model() -> CatBoostRegressor:
    """Load and cache the trained CatBoost model."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError("CatBoost model file was not found at: " f"{MODEL_PATH}")

    model = CatBoostRegressor()
    model.load_model(str(MODEL_PATH))

    return model


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------


def predict_sale_price(
    payload: Mapping[str, Any],
) -> float:
    """
    Predict the sale price of one house.

    Processing sequence:
    1. Validate required API features.
    2. Preserve the expected API feature order.
    3. Rename API-friendly fields to Ames field names.
    4. Apply the shared preprocessing pipeline.
    5. Match the exact feature schema/order of the model.
    6. Generate the prediction.
    """

    if not isinstance(payload, Mapping):
        raise TypeError("Prediction payload must be a mapping/dictionary.")

    missing_api_features = [feature for feature in FEATURES if feature not in payload]

    if missing_api_features:
        raise ValueError("Missing required API features: " f"{missing_api_features}")

    # Build one-row DataFrame using only expected API fields.
    df = pd.DataFrame([{feature: payload[feature] for feature in FEATURES}])

    # Convert API-friendly names back to original Ames names.
    df = df.rename(columns=API_TO_MODEL_COLUMNS)

    # IMPORTANT:
    # This is the exact same preprocessing function used by
    # the training/validation/test pipeline.
    df = preprocess_features(df)

    model = get_model()

    model_features = list(model.feature_names_)

    if not model_features:
        raise ValueError("The loaded CatBoost model does not contain " "feature-name metadata.")

    missing_model_features = [feature for feature in model_features if feature not in df.columns]

    if missing_model_features:
        raise ValueError(
            "Features required by the trained model are " f"missing after preprocessing: " f"{missing_model_features}"
        )

    # Match exact training feature order.
    df = df[model_features]

    prediction = model.predict(df)[0]

    return float(prediction)
