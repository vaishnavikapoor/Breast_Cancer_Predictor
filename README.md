# Breast Cancer Risk Prediction Platform

This project implements a complete machine learning pipeline for breast cancer malignancy prediction using a feature-optimized Logistic Regression model. The goal is to demonstrate that high diagnostic accuracy can be achieved using a minimal and interpretable feature set.

---

## Problem Statement

The objective is to determine the minimum number of tumor morphology features required to maintain near-optimal classification performance.

---
## 🚀 Live Application

- **Streamlit UI:** [https://your-streamlit-link  ](https://breastcancerpredictor-qh2pmhoyhkuxnp5c2af6uh.streamlit.app/)
- **FastAPI Backend:** https://breast-cancer-predictor-4vi0.onrender.com/docs

---
## Model Design

A Scikit-Learn pipeline is used to ensure reproducibility and numerical stability:

- Median imputation for missing values  
- Standard scaling for feature normalization  
- ANOVA F-test based feature selection (SelectKBest)  
- Logistic Regression classifier

---

## Feature Selection Results

The following features were identified as dominant predictors:

- `concave_points_worst`
- `radius_worst`
- `perimeter_worst`

Using only these features, the model achieves a mean ROC-AUC score of approximately 0.986.

---

## Evaluation

| Model | ROC-AUC |
|------|---------|
| Baseline Logistic Regression (unscaled, no feature selection) | 0.992 (numerically unstable) |
| Optimized Pipeline (k = 3 features) | 0.986 |

The optimized pipeline eliminates convergence issues observed in the baseline model while preserving diagnostic performance.

---

## Application Architecture

The system consists of a Streamlit user interface connected to a FastAPI backend that serves predictions from the trained pipeline. All predictions are logged and visualized in a monitoring dashboard.

---

## API Specification

| Method | Endpoint | Purpose |
|-------|----------|---------|
| GET | `/` | Health check |
| POST | `/predict` | Generate cancer risk prediction |
| GET | `/logs` | Return recent inference logs |

---

## Technology Stack

Python, Scikit-Learn, FastAPI, Streamlit, Render, Pandas, Joblib, Altair

---

## Key Learnings

- Effective preprocessing and feature selection can match complex models using simple linear classifiers.  
- Feature dominance analysis is essential for interpretable medical ML systems.  
- Production deployment requires robust pipelines rather than standalone models.
