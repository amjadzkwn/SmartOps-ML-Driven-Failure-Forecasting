# SmartOps: Machine Failure Prediction System

An end-to-end machine learning solution for predictive maintenance in industrial operations, featuring automated feature engineering, model tuning, and a real-time prediction interface.

Dataset: https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020

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

## 📈 Input Parameters
Parameter	Description	Typical Range
Air Temperature	Ambient temperature (K)	295-310 K
Process Temperature	Operational temperature (K)	305-315 K
Rotational Speed	Machine RPM	1000-3000 rpm
Torque	Rotational force (Nm)	30-70 Nm
Tool Wear	Tool usage duration (min)	0-250 min

🎯 Risk Assessment Logic
The system implements three risk levels based on failure probability:

🔴 CRITICAL (p > 0.7): Immediate shutdown required
🟡 WARNING (0.4 < p ≤ 0.7): Schedule preventive maintenance
🟢 NORMAL (p ≤ 0.4): Continue routine monitoring

🧠 Feature Engineering Logic
The system automatically generates these derived features:

python
Temp_Diff = Process_Temp - Air_Temp
Power = Torque × (Speed × 2π/60)

## Failure type indicators
HDF = 1 if (Temp_Diff < 8.6 and Speed < 1380) else 0
PWF = 1 if (Power < 3500 or Power > 9000) else 0
OSF = 1 if (Tool_Wear × Torque > 11000) else 0
TWF = 1 if Tool_Wear > 200 else 0
RNF = 0  # Random failure (placeholder)

🔄 Data Pipeline

Raw Sensor Data → Feature Engineering → SMOTE Balancing → 
Standard Scaling → Model Prediction → Risk Assessment

⚙️ Hyperparameter Tuning Ranges
Random Forest
n_estimators: [100, 200]

max_depth: [10, 20, None]

min_samples_split: [2, 5]

XGBoost
n_estimators: [100, 200]

learning_rate: [0.01, 0.1]

max_depth: [3, 6]

Logistic Regression
C: [0.1, 1, 10]

solver: ['lbfgs']

📊 Performance Optimization
SMOTE prevents overfitting on minority class

StratifiedKFold ensures representative validation

Pipeline prevents data leakage between SMOTE and scaling

F1-score optimization balances precision and recall for imbalanced data

🔮 Future Improvements
Add more advanced feature engineering (rolling windows, time-series features)

Implement LSTM/GRU for temporal patterns

Add SHAP explanations for model interpretability

Deploy as REST API using FastAPI

Add real-time data streaming (Kafka/MQTT)

Implement A/B testing framework

Add drift detection for model monitoring

📝 License
This project is developed for educational and research purposes in predictive maintenance.

👥 Contributors
Developed as part of SmartOps Portfolio Project - Semester 8
