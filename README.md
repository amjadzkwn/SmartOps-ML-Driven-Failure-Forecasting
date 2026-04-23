# SmartOps: Machine Failure Prediction System

An end-to-end machine learning solution for predictive maintenance in industrial operations, featuring automated feature engineering, model tuning, and a real-time prediction interface.

**Dataset:** https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020

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
```

## 🔧 Required Libraries

```bash
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.5.0
imbalanced-learn>=0.8.0
streamlit>=1.10.0
joblib>=1.1.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

## 📈 Exploratory Data Analysis (EDA)

**Failure Distribution**
![failure_distribution](https://github.com/amjadzkwn/SmartOps-ML-Driven-Failure-Forecasting/blob/5de734a5093092041175196a1ee98d4a5b6b36d3/1_failure_distribution.png)

**Correlation Heatmap**
![correlation_heatmap](https://github.com/amjadzkwn/SmartOps-ML-Driven-Failure-Forecasting/blob/8bc9273921533be4e9c3e95a2352f5ba86c4f34b/2_correlation_heatmap.png)

**Sensor Analysis**
![sensor_analysis](https://github.com/amjadzkwn/SmartOps-ML-Driven-Failure-Forecasting/blob/c3cec739f73b204c77bff8a1373f78aeaf6b107c/3_sensor_analysis.png)

## 📈 Input Parameters
Parameter	Description	Typical Range:

- Air Temperature	Ambient temperature (K)	295-310 K
- Process Temperature	Operational temperature (K)	305-315 K
- Rotational Speed	Machine RPM	1000-3000 rpm
- Torque	Rotational force (Nm)	30-70 Nm
- Tool Wear	Tool usage duration (min)	0-250 min

## 🎯 Risk Assessment Logic
The system implements three risk levels based on failure probability:

```bash
🔴 CRITICAL (p > 0.7): Immediate shutdown required
🟡 WARNING (0.4 < p ≤ 0.7): Schedule preventive maintenance
🟢 NORMAL (p ≤ 0.4): Continue routine monitoring
```

## 🧠 Feature Engineering Logic
The system automatically generates these derived features:

python
```bash
Temp_Diff = Process_Temp - Air_Temp
Power = Torque × (Speed × 2π/60)

## Failure type indicators
HDF = 1 if (Temp_Diff < 8.6 and Speed < 1380) else 0
PWF = 1 if (Power < 3500 or Power > 9000) else 0
OSF = 1 if (Tool_Wear × Torque > 11000) else 0
TWF = 1 if Tool_Wear > 200 else 0
RNF = 0  # Random failure (placeholder)
```

## 🔄 Data Pipeline

Raw Sensor Data → Feature Engineering → SMOTE Balancing → 
Standard Scaling → Model Prediction → Risk Assessment

## ⚙️ Hyperparameter Tuning Ranges

| Random Forest| |
|--------------|-|
| n_estimators: | [100, 200] |
| max_depth: | [10, 20, None] |
| min_samples_split: | [2, 5] |

| XGBoost | |
|---------|-|
| n_estimators: | [100, 200] |
| learning_rate: | [0.01, 0.1] |
| max_depth: | [3, 6] |

| Logistic Regression | |
|---------------------|-|
| C: | [0.1, 1, 10] |
| solver: | ['lbfgs'] |

## 📊 Performance Optimization

- SMOTE prevents overfitting on minority class
- StratifiedKFold ensures representative validation
- Pipeline prevents data leakage between SMOTE and scaling
- F1-score optimization balances precision and recall for imbalanced data

## 📊 Model Evaluation

**Before Hyperparameter Tuning / Boosting:**
|Model|F1-Score|ROC-AUC|
|-----|--------|-------|
|**Random Forest**|0.842105|0.986177|
|**XGBoost**|0.914286|0.991300|
|**Logistic Regression**|0.488889|0.988430|

**Random Forest:**
| | precision | recall | f1-score | support |
|-|-----------|--------|----------|---------|
|0|1.00|0.99|0.99|1932|
|1|0.76|0.94|0.84|68|
|accuracy| | |0.99|2000|
|macro avg|0.88|0.97|0.92|2000|
|weighted avg|0.99|0.99|0.99|2000|

![random_forest_cm](https://github.com/amjadzkwn/SmartOps-ML-Driven-Failure-Forecasting/blob/9b8fa6f6269363a8871930d8d2b8b58e6d8458b5/RandomForest_cm.png)

**XGBoost:**
| | precision | recall | f1-score | support |
|-|-----------|--------|----------|---------|
|0|1.00|1.00|1.00|1932|
|1|0.89|0.94|0.91|68|
|accuracy| | |0.99|2000|
|macro avg|0.94|0.97|0.96|2000|
|weighted avg|0.99|0.99|0.99|2000|

![xgboost_cm](https://github.com/amjadzkwn/SmartOps-ML-Driven-Failure-Forecasting/blob/795daa1af741c2408852d867c9add80e217dac34/XGBoost_cm.png)

**Logistic Regression:**
| | precision | recall | f1-score | support |
|-|-----------|--------|----------|---------|
|0|1.00|0.93|0.96|1932|
|1|0.33|0.97|0.49|68|
|accuracy| | |0.93|2000|
|macro avg|0.66|0.95|0.73|2000|
|weighted avg|0.98|0.93|0.95|2000|

![logistic_regression_cm](https://github.com/amjadzkwn/SmartOps-ML-Driven-Failure-Forecasting/blob/730df22f6a1b0395404e44254c8cff1c87315867/LogisticRegression_cm.png)

## 🛠 Model Validation & Test Scenarios

To ensure the SmartOps predictive maintenance model performs accurately across different industrial conditions, the following test cases were conducted using the Streamlit interface.

**1. Normal Operational State**

![Output 1](https://github.com/amjadzkwn/SmartOps-ML-Driven-Failure-Forecasting/blob/ed55cf1e9c1d28339c89464807954a4e62b2d8c1/Output1.png?raw=true)

- **Objective:** Verify that the model returns a "Normal" status when all parameters stay within safe operating ranges.
- **Key Indicators:** * Temperature Difference: 10.0K (Safe)
- **Power Output:** approx 5.6 kW (Stable)
- **Result:** 0.32% Failure Probability (Status: Normal).
- **Conclusion:** The system correctly identifies optimal conditions where no intervention is required.

**2. Heat Dissipation Failure (HDF)**

![Output 2](https://github.com/amjadzkwn/SmartOps-ML-Driven-Failure-Forecasting/blob/3d30a1d9ffc57d1e899bf379ac7cb0637fbbe5fb/Output2.png)

- **Objective:** Test the model's logic for HDF, which occurs when the temperature difference between ambient air and the process is too narrow (< 8.6 K) at low speeds.
- **Key Indicators:** * Low speed ($1300\text{ RPM}$) combined with insufficient heat release.
- **Result:** 74.57% Failure Probability (Status: Failure Warning).
- **Conclusion:** The model successfully detects thermal inefficiency that could lead to engine or component overheating.

**3. Power Failure (PWF)**

![Output 3](https://github.com/amjadzkwn/SmartOps-ML-Driven-Failure-Forecasting/blob/8f8a0a23d3fad682e335536380e950ec73031847/Output3.png)

- **Objective:** Test for imbalances between torque and rotational speed that lead to power surges.
- **Key Indicators:** * High Speed ($2500\text{ RPM}$) + High Torque ($60.0\text{ Nm}$).
Calculated Power exceeds the critical threshold (usually $> 9\text{ kW}$).
- **Result:** 95.86% Failure Probability (Status: Failure Warning).
- **Conclusion:** The system identifies "Overload" conditions, triggering an immediate recommendation to stop operations to prevent permanent electrical or mechanical damage.

**4. Overstrain Failure (OSF)**

![Output 4](https://github.com/amjadzkwn/SmartOps-ML-Driven-Failure-Forecasting/blob/89beb0a1aa89d26b3a6105a617d7ed0baf484926/Output4.png)

- **Objective:** Validate the detection of failures caused by the interaction of tool wear and excessive mechanical load.
- **Key Indicators:** * Critical Tool Wear ($210\text{ min}$) multiplied by high Torque ($55.0\text{ Nm}$).
- **Result:** 70.21% Failure Probability (Status: Failure Warning).
- **Conclusion:** The logic correctly identifies that a worn-out tool under high tension is a primary cause of physical machine breakage.

**5. Edge Case: Early Warning & Marginal Data**

![Output 5](https://github.com/amjadzkwn/SmartOps-ML-Driven-Failure-Forecasting/blob/ceb73b191524f7f39afd1775b3ec163fc8134895/Output5.png)

- **Objective:** Evaluate model sensitivity to values that are above average but haven't reached the critical failure threshold.
- **Key Indicators:** * Moderate Tool Wear ($140\text{ min}$) and slightly elevated Torque ($48\text{ Nm}$).
- **Result:** 0.85% Failure Probability (Status: Normal).
- **Conclusion:** The model maintains a conservative approach to avoid "false positives," keeping the status as Normal but allowing the operator to monitor the upward trend in tool wear via the dashboard.

## 🔮 Future Improvements

- Add more advanced feature engineering (rolling windows, time-series features)
- Implement LSTM/GRU for temporal patterns
- Add SHAP explanations for model interpretability
- Deploy as REST API using FastAPI
- Add real-time data streaming (Kafka/MQTT)
- Implement A/B testing framework
- Add drift detection for model monitoring

## 📝 License
This project is developed for educational and research purposes in predictive maintenance.

## 👥 Contributors
Developed as part of SmartOps Portfolio Project - Semester 8
