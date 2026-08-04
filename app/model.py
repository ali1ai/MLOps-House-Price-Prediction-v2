from catboost import CatBoostRegressor
import pandas as pd

MODEL_PATH = "models/catboost_model.cbm"

# Load model once at startup
model = CatBoostRegressor()
model.load_model(MODEL_PATH)

# Full feature list (same as training)
FEATURES = [
    "MSSubClass", "MSZoning", "LotFrontage", "LotArea", "Street", "Alley",
    "LotShape", "LandContour", "Utilities", "LotConfig", "LandSlope",
    "Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle",
    "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd", "RoofStyle",
    "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType", "MasVnrArea",
    "ExterQual", "ExterCond", "Foundation", "BsmtQual", "BsmtCond",
    "BsmtExposure", "BsmtFinType1", "BsmtFinSF1", "BsmtFinType2",
    "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "Heating", "HeatingQC",
    "CentralAir", "Electrical", "1stFlrSF", "2ndFlrSF", "LowQualFinSF",
    "GrLivArea", "BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath",
    "BedroomAbvGr", "KitchenAbvGr", "KitchenQual", "TotRmsAbvGrd",
    "Functional", "Fireplaces", "FireplaceQu", "GarageType", "GarageYrBlt",
    "GarageFinish", "GarageCars", "GarageArea", "GarageQual", "GarageCond",
    "PavedDrive", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch",
    "3SsnPorch", "ScreenPorch", "PoolArea", "PoolQC", "Fence",
    "MiscFeature", "MiscVal", "MoSold", "YrSold", "SaleType",
    "SaleCondition"
]

# Identify categorical columns
CAT_COLS = [col for col in FEATURES if df[col].dtype == "object"]

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # Fill missing categorical values
    for col in CAT_COLS:
        df[col] = df[col].astype(str).fillna("Missing")

    # Fill missing numeric values
    for col in df.columns:
        if col not in CAT_COLS:
            df[col] = df[col].fillna(0)

    return df

def predict_sale_price(payload: dict) -> float:
    df = pd.DataFrame([payload])
    df = preprocess(df)
    pred = model.predict(df[FEATURES])[0]
    return float(pred)

