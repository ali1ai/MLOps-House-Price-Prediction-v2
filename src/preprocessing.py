import pandas as pd


TARGET_COLUMN = "SalePrice"
ID_COLUMN = "Id"

MISSING_CATEGORY_TOKEN = "Missing"


# ---------------------------------------------------------
# Canonical Ames categorical features
# ---------------------------------------------------------

CATEGORICAL_FEATURES = [
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


# ---------------------------------------------------------
# Missing-value indicators
# ---------------------------------------------------------

MISSING_INDICATOR_COLUMNS = [
    "LotFrontage",
    "MasVnrArea",
]


def preprocess_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply the canonical feature preprocessing used by
    training, validation, testing, inference, and retraining.

    The function does not modify the input DataFrame.
    """

    df = df.copy()

    # Remove columns that must never enter the model.
    df = df.drop(
        columns=[
            ID_COLUMN,
            TARGET_COLUMN,
        ],
        errors="ignore",
    )

    # -----------------------------------------------------
    # Missing-value indicators
    # -----------------------------------------------------

    for column in MISSING_INDICATOR_COLUMNS:
        if column in df.columns:
            df[f"{column}_missing"] = df[column].isna().astype(int)

    # -----------------------------------------------------
    # Categorical preprocessing
    # -----------------------------------------------------

    for column in CATEGORICAL_FEATURES:
        if column in df.columns:
            df[column] = df[column].fillna(MISSING_CATEGORY_TOKEN).astype(str)

    # -----------------------------------------------------
    # Numerical preprocessing
    #
    # Everything that is not a known categorical feature
    # is treated as numerical.
    # -----------------------------------------------------

    numerical_columns = [column for column in df.columns if column not in CATEGORICAL_FEATURES]

    for column in numerical_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        ).fillna(0)

    return df


def preprocess_dataset(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Preprocess a labeled Ames dataset while preserving
    the SalePrice target.
    """

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Required target column " f"'{TARGET_COLUMN}' was not found.")

    target = pd.to_numeric(
        df[TARGET_COLUMN],
        errors="raise",
    ).copy()

    features = preprocess_features(df)

    processed_df = features.copy()
    processed_df[TARGET_COLUMN] = target.values

    return processed_df
