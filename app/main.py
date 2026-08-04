from fastapi import FastAPI
from pydantic import BaseModel
from app.model import predict_sale_price

app = FastAPI(
    title="Ames Housing Price Prediction API",
    description="Predict house prices using a CatBoost model",
    version="1.0"
)

# Define the input schema (must match your FEATURES list)
class HouseFeatures(BaseModel):
    MSSubClass: int
    MSZoning: str
    LotFrontage: float | None = None
    LotArea: float
    Street: str
    Alley: str | None = None
    LotShape: str
    LandContour: str
    Utilities: str
    LotConfig: str
    LandSlope: str
    Neighborhood: str
    Condition1: str
    Condition2: str
    BldgType: str
    HouseStyle: str
    OverallQual: int
    OverallCond: int
    YearBuilt: int
    YearRemodAdd: int
    RoofStyle: str
    RoofMatl: str
    Exterior1st: str
    Exterior2nd: str
    MasVnrType: str | None = None
    MasVnrArea: float | None = None
    ExterQual: str
    ExterCond: str
    Foundation: str
    BsmtQual: str | None = None
    BsmtCond: str | None = None
    BsmtExposure: str | None = None
    BsmtFinType1: str | None = None
    BsmtFinSF1: float | None = None
    BsmtFinType2: str | None = None
    BsmtFinSF2: float | None = None
    BsmtUnfSF: float | None = None
    TotalBsmtSF: float | None = None
    Heating: str
    HeatingQC: str
    CentralAir: str
    Electrical: str | None = None
    FirstFlrSF: float
    SecondFlrSF: float
    LowQualFinSF: float
    GrLivArea: float
    BsmtFullBath: float
    BsmtHalfBath: float
    FullBath: float
    HalfBath: float
    BedroomAbvGr: int
    KitchenAbvGr: int
    KitchenQual: str
    TotRmsAbvGrd: int
    Functional: str
    Fireplaces: int
    FireplaceQu: str | None = None
    GarageType: str | None = None
    GarageYrBlt: float | None = None
    GarageFinish: str | None = None
    GarageCars: float
    GarageArea: float
    GarageQual: str | None = None
    GarageCond: str | None = None
    PavedDrive: str
    WoodDeckSF: float
    OpenPorchSF: float
    EnclosedPorch: float
    ThreeSsnPorch: float
    ScreenPorch: float
    PoolArea: float
    PoolQC: str | None = None
    Fence: str | None = None
    MiscFeature: str | None = None
    MiscVal: float
    MoSold: int
    YrSold: int
    SaleType: str
    SaleCondition: str


@app.get("/")
def root():
    return {"message": "Ames Housing Price Prediction API is running!"}


@app.post("/predict")
def predict(features: HouseFeatures):
    prediction = predict_sale_price(features.dict())
    return {"SalePrice": prediction}
