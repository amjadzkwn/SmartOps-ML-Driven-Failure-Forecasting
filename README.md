# SmartOps: Machine Failure Prediction System

An end-to-end machine learning solution for predictive maintenance in industrial operations, featuring automated feature engineering, model tuning, and a real-time prediction interface.

## 📋 Overview

SmartOps is a predictive maintenance system that forecasts machine failures using sensor data. The system handles imbalanced datasets (96:3 ratio), performs automated feature engineering, and provides real-time risk assessment through an intuitive Streamlit interface.

## 🚀 Key Features

### Data Processing & Feature Engineering
- **Automated feature creation** including:
  - Temperature differential (`Temp_Diff = Process_Temp - Air_Temp`)
  - Power calculation (`Power = Torque × Angular Velocity`)
  - Failure type indicators (HDF, PWF, OSF, TWF, RNF) based on correlation analysis

### Model Training Pipeline
- **Handles class imbalance** using SMOTE (Synthetic Minority Over-sampling Technique)
- **Stratified K-Fold cross-validation** (3 folds) for stable validation
- **StandardScaler** for feature normalization
- **GridSearchCV** for hyperparameter optimization
- **F1-score optimization** for imbalanced classification

### Models Evaluated
1. **XGBoost** - Gradient boosting with regularization
2. **Random Forest** - Ensemble of decision trees
3. **Logistic Regression** - Baseline linear model

## 📊 Model Performance Metrics

| Model | F1-Score | ROC-AUC |
|-------|----------|---------|
| XGBoost | Best performing | High discrimination |
| Random Forest | Competitive | Strong performance |
| Logistic Regression | Baseline | Moderate |

## 🔧 Installation

### Prerequisites
```bash
Python 3.8+
pip install -r requirements.txt

## 🔧 Required Libraries

pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.5.0
imbalanced-learn>=0.8.0
streamlit>=1.10.0
joblib>=1.1.0
matplotlib>=3.4.0
seaborn>=0.11.0
