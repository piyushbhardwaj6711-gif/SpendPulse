
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import shap

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join("data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
MODELS_DIR = "models"
RANDOM_STATE = 42

# Numeric columns that get StandardScaler treatment (rest are 0/1 dummies).
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


# ---------------------------------------------------------------------------
# 1. Load + clean
# ---------------------------------------------------------------------------
def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep='\t')

    # Drop the unique identifier — it carries no predictive signal.
    df = df.drop(columns=["customerID"])

    # TotalCharges is read as an object because a handful of rows contain
    # blank strings (new customers with 0 tenure). Coerce to numeric and
    # impute with the median.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Binary-encode the target: Yes -> 1, No -> 0
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df


# ---------------------------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------------------------
def encode_features(df: pd.DataFrame):
    y = df["Churn"]
    X = df.drop(columns=["Churn"])

    # One-hot encode every remaining categorical column. drop_first=True
    # avoids the dummy-variable trap while keeping the feature set compact.
    X = pd.get_dummies(X, drop_first=True)

    return X, y


# ---------------------------------------------------------------------------
# 3. Train / evaluate a single model
# ---------------------------------------------------------------------------
def evaluate_model(model, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "precision": round(precision_score(y_test, preds), 4),
        "recall": round(recall_score(y_test, preds), 4),
        "f1": round(f1_score(y_test, preds), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
    }


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("Loading and cleaning data...")
    df = load_and_clean(DATA_PATH)

    print("Encoding features...")
    X, y = encode_features(df)
    feature_columns = X.columns.tolist()  # saved for inference-time alignment

    print("Splitting 80/20 (stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    print("Scaling numeric columns...")
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[NUMERIC_COLS] = scaler.fit_transform(X_train[NUMERIC_COLS])
    X_test_scaled[NUMERIC_COLS] = scaler.transform(X_test[NUMERIC_COLS])

    # -----------------------------------------------------------------
    # Train baselines
    # -----------------------------------------------------------------
    print("Training Logistic Regression...")
    log_reg = LogisticRegression(max_iter=1000, class_weight="balanced")
    log_reg.fit(X_train_scaled, y_train)
    log_reg_metrics = evaluate_model(log_reg, X_test_scaled, y_test)

    print("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train_scaled, y_train)
    rf_metrics = evaluate_model(rf, X_test_scaled, y_test)

    print("\nLogistic Regression:", log_reg_metrics)
    print("Random Forest:      ", rf_metrics)

    # -----------------------------------------------------------------
    # Select best model — ROC-AUC as primary criterion, Recall as
    # tiebreaker since missing a churner is costlier than a false alarm.
    # -----------------------------------------------------------------
    candidates = [
        ("logistic_regression", log_reg, log_reg_metrics),
        ("random_forest", rf, rf_metrics),
    ]
    best_name, best_model, best_metrics = max(
        candidates, key=lambda c: (c[2]["roc_auc"], c[2]["recall"])
    )
    print(f"\nSelected best model: {best_name}")

    # -----------------------------------------------------------------
    # SHAP explainer — TreeExplainer for tree models, LinearExplainer
    # for linear models. Saved separately isn't required; app.py
    # reconstructs the explainer from the saved model + a background
    # sample, which is more portable across SHAP versions.
    # -----------------------------------------------------------------
    print("Fitting SHAP explainer for sanity-check...")
    background = X_train_scaled.sample(min(100, len(X_train_scaled)), random_state=RANDOM_STATE)
    if best_name == "random_forest":
        explainer = shap.TreeExplainer(best_model)
    else:
        explainer = shap.LinearExplainer(best_model, background)
    _ = explainer(X_test_scaled.iloc[:5])  # smoke test, not saved

    # -----------------------------------------------------------------
    # Persist artifacts
    # -----------------------------------------------------------------
    joblib.dump(best_model, os.path.join(MODELS_DIR, "model.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(feature_columns, os.path.join(MODELS_DIR, "columns.pkl"))
    joblib.dump(NUMERIC_COLS, os.path.join(MODELS_DIR, "numeric_cols.pkl"))
    joblib.dump(best_name, os.path.join(MODELS_DIR, "model_name.pkl"))
    # Small background sample needed to reconstruct LinearExplainer at inference.
    joblib.dump(background, os.path.join(MODELS_DIR, "background.pkl"))

    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump(
            {"logistic_regression": log_reg_metrics, "random_forest": rf_metrics, "selected": best_name},
            f,
            indent=2,
        )

    print(f"\nArtifacts saved to '{MODELS_DIR}/'. Best model: {best_name} | {best_metrics}")


if __name__ == "__main__":
    main()
