import pandas as pd
import pytest

DATA_PATH = "tests/fixtures/ames_sample.csv"


@pytest.fixture
def df():
    return pd.read_csv(DATA_PATH)


def test_required_columns_exist(df):
    required = [
        "MSSubClass",
        "MSZoning",
        "LotArea",
        "OverallQual",
        "OverallCond",
        "YearBuilt",
        "GrLivArea",
        "SalePrice",
    ]
    for col in required:
        assert col in df.columns


def test_no_missing_target(df):
    assert df["SalePrice"].isnull().sum() == 0


def test_no_missing_critical_features(df):
    critical = ["MSZoning", "LotArea", "OverallQual", "GrLivArea"]
    for col in critical:
        assert df[col].isnull().sum() == 0


def test_lotarea_positive(df):
    assert (df["LotArea"] > 0).all()


def test_grlivarea_reasonable(df):
    assert df["GrLivArea"].between(300, 6000).all()


def test_mszoning_allowed_values(df):
    allowed = {"RL", "RM", "FV", "RH", "C (all)"}
    assert set(df["MSZoning"].dropna().unique()).issubset(allowed)


def test_no_duplicate_rows(df):
    assert df.duplicated().sum() == 0


def test_saleprice_not_extreme(df):
    assert df["SalePrice"].between(20000, 800000).all()


def test_yearbuilt_valid(df):
    assert df["YearBuilt"].between(1800, 2026).all()
