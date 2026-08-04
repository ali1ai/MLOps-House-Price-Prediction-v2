import pandas as pd
import pytest

DATA_PATH = "data/raw/train.csv"


@pytest.fixture
def df():
    return pd.read_csv(DATA_PATH)


# ---------------------------------------------------------
# 1. Basic schema validation
# ---------------------------------------------------------

def test_required_columns_exist(df):
    required = [
        "MSSubClass", "MSZoning", "LotArea", "OverallQual", "OverallCond",
        "YearBuilt", "GrLivArea", "SalePrice"
    ]
    for col in required:
        assert col in df.columns


# ---------------------------------------------------------
# 2. Missing values
# ---------------------------------------------------------

def test_no_missing_target(df):
    assert df["SalePrice"].isnull().sum() == 0


def test_no_missing_critical_features(df):
    critical = ["MSZoning", "LotArea", "OverallQual", "GrLivArea"]
    for col in critical:
        assert df[col].isnull().sum() == 0


# ---------------------------------------------------------
# 3. Numeric ranges
# ---------------------------------------------------------

def test_lotarea_positive(df):
    assert (df["LotArea"] > 0).all()


def test_grlivarea_reasonable(df):
    # Most houses are between 300 and 6000 sqft
    assert df["GrLivArea"].between(300, 6000).all()


# ---------------------------------------------------------
# 4. Categorical allowed sets
# ---------------------------------------------------------

def test_mszoning_allowed_values(df):
    allowed = {"RL", "RM", "FV", "RH", "C (all)"}
    assert set(df["MSZoning"].dropna().unique()).issubset(allowed)


# ---------------------------------------------------------
# 5. Duplicates
# ---------------------------------------------------------

def test_no_duplicate_rows(df):
    assert df.duplicated().sum() == 0


# ---------------------------------------------------------
# 6. Outliers
# ---------------------------------------------------------

def test_saleprice_not_extreme(df):
    # Typical range: 20k to 750k
    assert df["SalePrice"].between(20000, 750000).all()


# ---------------------------------------------------------
# 7. Date formats (if present)
# ---------------------------------------------------------

def test_yearbuilt_valid(df):
    assert df["YearBuilt"].between(1800, 2026).all()
