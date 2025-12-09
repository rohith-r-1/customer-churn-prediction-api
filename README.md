# Customer Churn Prediction API

End-to-end machine learning project that predicts bank customer churn and exposes the model via a REST API.

## 1. Problem

Banks lose revenue when valuable customers close their accounts.  
The goal is to:

- Predict the probability that a customer will churn.
- Turn that probability into a **business decision** using an optimal threshold.
- Expose the model via an API that other services can call.

## 2. Project Highlights

- Full ML pipeline: data cleaning → feature engineering → modeling → evaluation.
- Models compared: Logistic Regression, Random Forest, XGBoost.
- Best model: XGBoost with ROC-AUC ≈ 0.86 (on held-out test set).
- Interpretability:
  - Global: feature importance, SHAP summary plots.
  - Local: SHAP for individual customers.
- Business layer:
  - Custom cost for false positives vs false negatives.
  - Decision-threshold tuning (best threshold ≈ 0.10).
- Deployment:
  - Flask API with `/predict_churn` endpoint.
  - Unified preprocessing + inference in `src/inference.py`.

## 3. Tech Stack

- Python 3.x  
- pandas, NumPy  
- scikit-learn, XGBoost  
- SHAP (interpretability)  
- Flask (API)

## 4. Repository Structure

├── app.py # Flask app exposing the churn prediction endpoint

├── notebooks/ # EDA, modeling, SHAP, threshold tuning

│ ├── 01_eda_feature_engineering.ipynb

│ ├── 02_modeling_evaluation.ipynb

│ ├── 03_interpretability_shap.ipynb

│ └── 04_business_threshold.ipynb

├── src/

│ └── inference.py # Unified preprocessing + model loading + prediction

├── data/ # (optional) data and processed artifacts

└── models/ # (optional) trained model + scaler + threshold


> Note: `data/` and `models/` may be `.gitignored` if they contain large or private files.  
> The project assumes you have `processed_data.pkl`, `best_model.pkl`, and `decision_threshold.txt` from the training notebooks.

## 5. Running the Project Locally

### 5.1. Setup

create and activate virtual environment (Windows example)
python -m venv .venv
.venv\Scripts\activate

install dependencies
pip install -r requirements.txt


(If you don’t have `requirements.txt` yet, you can generate one with `pip freeze > requirements.txt`.)

### 5.2. Start the API

From the project root:

.venv\Scripts\activate

python app.py


You should see something like:

Running on http://127.0.0.1:5000


### 5.3. Call the API

Send a POST request to:

`http://127.0.0.1:5000/predict_churn`

with JSON body:

{
"CreditScore": 650,
"Geography": "France",
"Gender": "Female",
"Age": 42,
"Tenure": 3,
"Balance": 50000,
"NumOfProducts": 2,
"HasCrCard": 1,
"IsActiveMember": 1,
"EstimatedSalary": 80000
}


Example using PowerShell:
$body = @{
CreditScore = 650
Geography = "France"
Gender = "Female"
Age = 42
Tenure = 3
Balance = 50000
NumOfProducts = 2
HasCrCard = 1
IsActiveMember = 1
EstimatedSalary = 80000
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:5000/predict_churn" -Method POST
-ContentType "application/json" `
-Body $body


Example response:
{
"churn_probability": 0.118,
"churn_prediction": 1,
"recommendation": "Target with retention offer"
}


## 6. Modeling Overview

High-level steps (implemented in the notebooks):

1. **EDA & Cleaning**
   - Handle missing values and outliers.
   - Explore churn rate across age, geography, products, activity.

2. **Feature Engineering**
   - Buckets: `AgeBucket`, `TenureBucket`.
   - Flags: `HasBalance`, `HighBalance`, `MultipleProducts`, `HighRiskProducts`.
   - Ratios: `BalancePerProduct`, `SalaryBalanceRatio`.
   - Geography: `IsGermany` and one-hot encoding.
   - Combined risk: `InactiveHighBalance`.

3. **Model Training**
   - Train Logistic Regression, Random Forest, XGBoost.
   - Evaluate with ROC-AUC, precision, recall, F1, confusion matrix.

4. **Interpretability**
   - Feature importance for tree models.
   - SHAP summary and example customer explanations.

5. **Business Threshold**
   - Define cost of FP vs FN.
   - Scan thresholds to minimize expected cost.
   - Select optimal decision threshold (≈ 0.10).

6. **Deployment**
   - Save scaler, feature names, model, decision threshold.
   - Implement `src/inference.py` to:
     - Rebuild features for a new customer.
     - Apply one-hot encoding + scaling.
     - Return churn probability and class based on threshold.
   - Wrap with Flask (`app.py`).

## 7. Model & Data Artifacts

The inference pipeline in `src/inference.py` expects the following files:

- `data/processed_data.pkl`  
  - A dictionary created by the training notebooks containing:
    - `X_train`, `X_test`, `y_train`, `y_test`
    - `feature_names` (list of all final feature columns)
    - `scaler` (fitted `StandardScaler` for numeric features)

- `models/best_model.pkl`  
  - The trained classifier (XGBoost or whatever model performed best).

- `models/decision_threshold.txt`  
  - A text file with a single floating-point number: the optimal churn probability threshold chosen from the business cost analysis.

These are produced by the notebooks in `notebooks/`:

1. Training notebook:
   - Builds features, splits data.
   - Fits the scaler and model.
   - Saves `data/processed_data.pkl` and `models/best_model.pkl`.

2. Business threshold notebook:
   - Scans decision thresholds using a custom cost function.
   - Saves the chosen value to `models/decision_threshold.txt`.

To run the API successfully, make sure these files exist in the expected paths.

