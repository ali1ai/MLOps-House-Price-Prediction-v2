from pathlib import Path

import pandas as pd
from catboost import CatBoostRegressor


MODEL_PATH = Path("models/catboost_model.cbm")

model = CatBoostRegressor()
model.load_model(str(MODEL_PATH))


# API-friendly feature names used by FastAPI and Pydantic.
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


# Convert API-friendly names to the original Ames column names used
# when the CatBoost model was trained.
API_TO_MODEL_COLUMNS = {
    "FirstFlrSF": "1stFlrSF",
    "SecondFlrSF": "2ndFlrSF",
    "ThreeSsnPorch": "3SsnPorch",
}


# Exact feature names and order expected by the saved CatBoost model.
MODEL_FEATURES = [
    API_TO_MODEL_COLUMNS.get(feature, feature) for feature in FEATURES
]


# Original categorical feature names expected by the trained model.
CAT_COLS = [
    "MSZoning",
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
    "RoofStyle",
    "RoofMatl",
    "Exterior1st",
    "Exterior2nd",
    "MasVnrType",
    "ExterQual",
    "ExterCond",
    "Foundation",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "Heating",
    "HeatingQC",
    "CentralAir",
    "Electrical",
    "KitchenQual",
    "Functional",
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "PavedDrive",
    "PoolQC",
    "Fence",
    "MiscFeature",
    "SaleType",
    "SaleCondition",
]


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values using the same types expected by CatBoost."""

    for column in CAT_COLS:
        if column in df.columns:
            df[column] = df[column].fillna("Missing").astype(str)

    for column in df.columns:
        if column not in CAT_COLS:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    return df


def predict_sale_price(payload: dict) -> float:
    """Predict a house sale price from one API request payload."""

    df = pd.DataFrame([payload])

    missing_api_features = [
        feature for feature in FEATURES if feature not in df.columns
    ]
    if missing_api_features:
        raise ValueError(
            f"Missing required features: {missing_api_features}"
        )

    # Keep only the expected API features and preserve their order.
    df = df[FEATURES]

    # Rename the API-friendly columns to the original training names.
    df = df.rename(columns=API_TO_MODEL_COLUMNS)

    # Verify that all model features are present.
    missing_model_features = [
        feature for feature in MODEL_FEATURES if feature not in df.columns
    ]
    if missing_model_features:
        raise ValueError(
            f"Missing model features after renaming: "
            f"{missing_model_features}"
        )

    # Use the exact feature order expected by CatBoost.
    df = df[MODEL_FEATURES]
    df = preprocess(df)

    prediction = model.predict(df)[0]
    return float(prediction)
