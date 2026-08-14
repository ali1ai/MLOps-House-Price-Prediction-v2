# House Price Prediction – End-to-End MLOps Pipeline

An end-to-end Machine Learning Operations (MLOps) project for predicting residential property prices using the Ames Housing dataset.

The project demonstrates the complete machine learning lifecycle, including reproducible data preparation, model training and evaluation, experiment tracking, data and model versioning, API serving, containerization, CI/CD, cloud deployment, drift monitoring, and conditional retraining.

---

## Overview

The objective of this project is not only to train an accurate regression model, but also to build a reproducible and operational MLOps system around it.

The pipeline includes:

- Data preparation and validation
- Shared feature preprocessing
- Train / validation / test separation
- CatBoost model training
- Model evaluation
- MLflow experiment tracking
- DVC data and model versioning
- DagsHub remote DVC storage
- FastAPI model serving
- Docker containerization
- Render cloud deployment
- GitHub Actions CI/CD
- EvidentlyAI drift monitoring
- Scheduled and manual monitoring
- Conditional candidate-model retraining
- Current-vs-candidate model comparison
- Model promotion only when performance improves

---

## Architecture

<p align="center">
  <img src="images/pipeline-diagram.png" width="900" alt="MLOps Pipeline Architecture">
</p>

The end-to-end workflow is:

```text
                    Ames Housing Dataset
                             │
                             ▼
                     Data Versioning
                           DVC
                             │
                             ▼
                    Data Preparation
                       prepare.py
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
           Train         Validation         Test
            70%             15%             15%
              │
              ▼
                  Shared Preprocessing
                  preprocessing.py
              │
              ▼
                    CatBoost Training
                        train.py
              │
        ┌─────┴───────────────┐
        ▼                     ▼
   Model Artifact         MLflow Tracking
        │                 Parameters
        │                 Metrics
        │                 Artifacts
        ▼
                     Model Evaluation
                       evaluate.py
                             │
                             ▼
                       FastAPI Service
                             │
                             ▼
                      Docker Container
                             │
                             ▼
                     Render Deployment
                             │
                             ▼
                  Production Predictions
                             │
                             ▼
                   Evidently Monitoring
                             │
                  ┌──────────┴──────────┐
                  │                     │
             No Drift              Drift / Manual
                  │                 Validation
                  │                     │
                  ▼                     ▼
              Continue          Candidate Retraining
                                      │
                                      ▼
                           Current vs Candidate
                              Model Comparison
                                      │
                                      ▼
                         Promote Only If Better

---

MLOps-House-Price-Prediction-v2/
│
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── monitor_retrain.yml
│
├── app/
│   ├── main.py
│   └── model.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── baseline.csv
│   └── new_batch.csv
│
├── images/
│
├── models/
│   └── catboost_model.cbm
│
├── monitoring/
│   └── drift_monitor.py
│
├── reports/
│   ├── metrics.json
│   ├── drift_report.html
│   ├── drift_status.json
│   └── retraining_metrics.json
│
├── retraining/
│   └── retrain.py
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── prepare.py
│   ├── train.py
│   └── evaluate.py
│
├── tests/
│
├── Dockerfile
├── params.yaml
├── dvc.yaml
├── dvc.lock
├── model_card.md
├── requirements.txt
└── README.md
---

| Component               | Technology            |
| ----------------------- | --------------------- |
| Language                | Python                |
| ML Model                | CatBoost Regressor    |
| Data Processing         | pandas / scikit-learn |
| Experiment Tracking     | MLflow                |
| Data & Model Versioning | DVC                   |
| DVC Remote Storage      | DagsHub               |
| Configuration           | YAML                  |
| API                     | FastAPI               |
| API Server              | Uvicorn               |
| Containerization        | Docker                |
| Cloud Deployment        | Render                |
| CI/CD                   | GitHub Actions        |
| Monitoring              | EvidentlyAI           |
| Source Control          | Git + GitHub          |

---

## Dataset

**Dataset**

House Prices: Advanced Regression Techniques

**Source**

https://www.kaggle.com/c/house-prices-advanced-regression-techniques

**Target Variable**

```
SalePrice
```

The dataset contains numerical and categorical attributes describing residential properties in Ames, Iowa. The objective is to predict the final sale price of each property.

---

## Pipeline

The pipeline consists of three reproducible stages.

### Data Preparation

- Load raw dataset
- Clean missing values
- Split train and test datasets
- Save processed data

### Model Training

- Read hyperparameters from `params.yaml`
- Train CatBoost model
- Save trained model
- Log experiment metadata

### Evaluation

- Generate predictions
- Calculate evaluation metrics
- Store metrics
- Log results to MLflow

---

## Running the Pipeline

Clone the repository.

```bash
git clone https://github.com/yourusername/house-price-prediction.git

cd house-price-prediction
```

Install dependencies.

```bash
pip install -r requirements.txt
```

Run the complete pipeline.

```bash
dvc repro
```

Launch MLflow.

```bash
mlflow ui
```

---

## Configuration

Pipeline parameters are managed through

```text
params.yaml
```

This enables reproducible experimentation without modifying application code.

---

## Experiment Tracking

MLflow is used to record

- Parameters
- Metrics
- Model artifacts
- Experiment history

Each pipeline execution produces a fully traceable experiment for comparison and reproducibility.

---

## Data Versioning

DVC manages

- Raw datasets
- Processed datasets
- Trained models
- Evaluation reports

Versioning data independently of Git ensures reproducible machine learning experiments while keeping the repository lightweight.

---

## Model

The project uses **CatBoost Regressor**, which provides:

- Native support for categorical features
- Minimal preprocessing requirements
- Strong regression performance
- Efficient handling of missing values
- Reduced overfitting through ordered boosting

---

## Evaluation

The model is evaluated using standard regression metrics.

- RMSE
- MAE
- R² Score

Evaluation reports are stored in

```text
reports/metrics.json
```

---

## Future Work

Potential extensions include:

Production model registry
Automated rollback
Prediction-performance monitoring when ground-truth labels become available
More advanced drift thresholds
Monitoring alerts and notifications
More extensive hyperparameter optimization
Authentication and rate limiting for the public API
Advanced observability dashboards
Automated notification channels for monitoring events
Additional model comparison strategies
More advanced deployment strategies

---

## License

This project is licensed under the MIT License.


