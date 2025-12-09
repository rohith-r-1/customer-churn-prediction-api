import pickle
import numpy as np
import pandas as pd

# ---------- Load all artifacts ----------

# processed_data.pkl contains: X_train, X_test, y_train, y_test, feature_names, scaler
with open('data/processed_data.pkl', 'rb') as f:
    data = pickle.load(f)

feature_names = data['feature_names']          # list of all feature columns used during training
scaler = data['scaler']                        # StandardScaler fitted on training numeric cols

# best_model.pkl is your trained XGBoost (or best model)
with open('models/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# decision_threshold.txt is the best threshold you saved
with open('models/decision_threshold.txt', 'r') as f:
    decision_threshold = float(f.read().strip())

# ---------- Define constants ----------

# numeric columns that were scaled
NUMERIC_COLS = [
    'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
    'EstimatedSalary', 'BalancePerProduct', 'SalaryBalanceRatio'
]

# original (pre-encoded) categorical columns
CATEGORICAL_COLS = ['Geography', 'Gender', 'AgeBucket', 'TenureBucket']

# ---------- Feature engineering helpers ----------

def age_bucket(age: int) -> str:
    if age < 30:
        return 'young'
    elif age < 45:
        return 'middle'
    elif age < 60:
        return 'senior'
    else:
        return 'elderly'

def tenure_bucket(tenure: int) -> str:
    if tenure <= 2:
        return 'new'
    elif tenure <= 5:
        return 'mid'
    else:
        return 'loyal'


def engineer_features(raw: dict) -> pd.DataFrame:
    """
    raw: dict with keys matching original df_clean columns before feature engineering, e.g.
         {
           "CreditScore": 650,
           "Geography": "France",
           "Gender": "Female",
           "Age": 42,
           "Tenure": 3,
           "Balance": 50000.0,
           "NumOfProducts": 2,
           "HasCrCard": 1,
           "IsActiveMember": 1,
           "EstimatedSalary": 80000.0
         }

    Returns: DataFrame with engineered features (before one-hot encoding & scaling)
    """

    df = pd.DataFrame([raw]).copy()

    # --- Derived buckets ---
    df['AgeBucket'] = df['Age'].apply(age_bucket)
    df['TenureBucket'] = df['Tenure'].apply(tenure_bucket)

    # --- Balance flags ---
    df['HasBalance'] = (df['Balance'] > 0).astype(int)

    # NOTE: we don't know the train 75th percentile exactly here, but for inference
    # we can approximate: use same rule as training (ideally you would store balance_75)
    # For simplicity, treat "high balance" as > 0.75 * max_balance ~ 0.75 * 250898 ≈ 188k
    # If you saved balance_75 during training, load it here instead.
    balance_75_approx = 188000.0
    df['HighBalance'] = (df['Balance'] > balance_75_approx).astype(int)

    # --- Product features ---
    df['MultipleProducts'] = (df['NumOfProducts'] > 1).astype(int)
    df['HighRiskProducts'] = (df['NumOfProducts'] >= 3).astype(int)

    # --- Ratios ---
    df['BalancePerProduct'] = df['Balance'] / df['NumOfProducts']
    df['SalaryBalanceRatio'] = df['EstimatedSalary'] / (df['Balance'] + 1.0)

    # --- Geography flags ---
    df['IsGermany'] = (df['Geography'] == 'Germany').astype(int)

    # --- Combined risk feature ---
    df['InactiveHighBalance'] = (
        (df['IsActiveMember'] == 0) & (df['Balance'] > balance_75_approx)
    ).astype(int)

    return df


def one_hot_and_align(df_engineered: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encodes categorical cols and aligns columns to training feature_names.
    """

    # One-hot encode using pandas.get_dummies
    df_encoded = pd.get_dummies(
        df_engineered,
        columns=CATEGORICAL_COLS,
        drop_first=True
    )

    # Ensure all features in feature_names exist; add missing with 0
    for col in feature_names:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    # Drop any extra columns not used during training
    df_encoded = df_encoded[feature_names]

    return df_encoded


def scale_numeric(df_aligned: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the trained StandardScaler to numeric columns.
    """

    df_scaled = df_aligned.copy()
    df_scaled[NUMERIC_COLS] = scaler.transform(df_scaled[NUMERIC_COLS])

    return df_scaled

# ---------- Public functions: preprocess + predict ----------

def preprocess_single(input_dict: dict) -> pd.DataFrame:
    """
    Full preprocessing pipeline for a single customer.

    input_dict keys should match the ORIGINAL cleaned columns before feature engineering:
        CreditScore, Geography, Gender, Age, Tenure,
        Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary

    Returns: 1-row DataFrame with final feature columns in the correct order.
    """
    # 1) Feature engineering
    df_engineered = engineer_features(input_dict)

    # 2) One-hot encoding + column alignment
    df_aligned = one_hot_and_align(df_engineered)

    # 3) Scaling numeric columns
    df_scaled = scale_numeric(df_aligned)

    return df_scaled   # shape (1, n_features)


def predict_churn(input_dict: dict):
    """
    Runs full pipeline: preprocess -> probability -> class prediction using decision_threshold.

    Returns:
        churn_proba: float (0-1)
        churn_pred: int  (0 or 1)
    """

    X_row = preprocess_single(input_dict)
    proba = model.predict_proba(X_row)[:, 1][0]
    pred = int(proba >= decision_threshold)

    return proba, pred


# ---------- Example usage (for manual testing) ----------
if __name__ == "__main__":
    example_customer = {
        "CreditScore": 650,
        "Geography": "France",
        "Gender": "Female",
        "Age": 42,
        "Tenure": 3,
        "Balance": 50000.0,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 80000.0
    }

    p, y = predict_churn(example_customer)
    print(f"Churn probability: {p:.3f}, prediction: {y}")
