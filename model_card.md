# Model Card: Ames Housing Price Prediction

## 1. Model Overview

This project implements a machine learning system for predicting residential house sale prices using the Ames Housing dataset.

The prediction model is a CatBoost regression model exposed through a FastAPI inference API. The project also includes automated testing, CI/CD, model monitoring, drift detection, and conditional model retraining.

### Model Details

- **Model type:** CatBoost Regressor
- **Task:** Regression
- **Target variable:** SalePrice
- **Dataset:** Ames Housing
- **API framework:** FastAPI
- **Model monitoring:** EvidentlyAI
- **CI/CD:** GitHub Actions
- **Model format:** CatBoost `.cbm`

---

## 2. Intended Use

The model is intended to demonstrate an end-to-end MLOps workflow for house price prediction.

Given a set of property characteristics, the model predicts the expected sale price of a house.

Example input features include:

- Overall quality
- Living area
- Year built
- Neighborhood
- Lot area
- Number of bedrooms
- Garage characteristics
- Basement characteristics
- Exterior characteristics

The project is primarily intended for educational and demonstration purposes rather than real-world financial or property valuation decisions.

---

## 3. Training Data

The model uses the Ames Housing dataset.

The training data contains residential property characteristics such as:

- Numerical features
- Categorical features
- Missing values
- Property construction information
- Property size information
- Location-related information
- Sale information

The target variable is:

`SalePrice`

CatBoost is used because it can effectively handle both numerical and categorical features.

---

## 4. Model Training

The model is trained using `CatBoostRegressor`.

The training pipeline:

1. Loads the training and test datasets.
2. Separates `SalePrice` from the input features.
3. Identifies categorical features.
4. Trains a CatBoost regression model.
5. Evaluates the model on test data.
6. Saves the trained model as:

`models/catboost_model.cbm`

The project uses regression metrics including:

- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² score

---

## 5. Model Inference

The trained model is served through a FastAPI application.

The main prediction endpoint is:

`POST /predict`

The API receives housing characteristics as JSON input and returns a predicted sale price.

Example response:

```json
{
  "SalePrice": 200000.0
}
