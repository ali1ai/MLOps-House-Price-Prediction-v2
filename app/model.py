from pathlib import Path

import pandas as pd
from catboost import CatBoostRegressor

# Build the model path relative to the project directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "catboost_model.cbm"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"CatBoost model file was not found: {MODEL_PATH}")

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


# Map Python-friendly API names to the original Ames column names
# used when the saved CatBoost model was trained.
API_TO_MODEL_COLUMNS = {
    "FirstFlrSF": "1stFlrSF",
    "SecondFlrSF": "2ndFlrSF",
    "ThreeSsnPorch": "3SsnPorch",
}


# Categorical columns expected by the trained CatBoost model.
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
    """Fill missing values and prepare data types for CatBoost."""

    df = df.copy()

    for column in CAT_COLS:
        if column in df.columns:
            df[column] = df[column].fillna("Missing").astype(str)

    for column in df.columns:
        if column not in CAT_COLS:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    return df


def predict_sale_price(payload: dict) -> float:
    """Predict the sale price for one house."""

    df = pd.DataFrame([payload])

    missing_api_features = [feature for feature in FEATURES if feature not in df.columns]
    if missing_api_features:
        raise ValueError(f"Missing required API features: {missing_api_features}")

    # Keep the API columns in a consistent order.
    df = df[FEATURES].copy()

    # Create the engineered missing-value indicators that were present
    # when the saved model was trained. These must be created before
    # filling the original missing values.
    df["LotFrontage_missing"] = df["LotFrontage"].isna().astype(int)
    df["MasVnrArea_missing"] = df["MasVnrArea"].isna().astype(int)

    # Rename API-friendly fields to their original Ames names.
    df = df.rename(columns=API_TO_MODEL_COLUMNS)

    # Read the exact feature names and ordering from the saved model.
    model_features = list(model.feature_names_)

    missing_model_features = [feature for feature in model_features if feature not in df.columns]
    if missing_model_features:
        raise ValueError(f"Features required by the trained model are missing: {missing_model_features}")

    df = df[model_features]
    df = preprocess(df)

    prediction = model.predict(df)[0]
    return float(prediction)
