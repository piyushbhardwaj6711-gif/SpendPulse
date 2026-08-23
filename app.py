"""
app.py
------
Streamlit UI for the Customer Churn Prediction model.

Run:
    streamlit run app.py

Loads the artifacts produced by src/train.py (models/model.pkl, scaler.pkl,
columns.pkl, etc.), builds a single-row input matching the training feature
space, predicts churn probability, and renders a SHAP explanation for that
specific prediction.
"""

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📉",
    layout="wide",
)

MODELS_DIR = "models"


# ---------------------------------------------------------------------------
# Cached artifact loading — runs once per session, not on every rerun.
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(f"{MODELS_DIR}/model.pkl")
    scaler = joblib.load(f"{MODELS_DIR}/scaler.pkl")
    columns = joblib.load(f"{MODELS_DIR}/columns.pkl")
    numeric_cols = joblib.load(f"{MODELS_DIR}/numeric_cols.pkl")
    model_name = joblib.load(f"{MODELS_DIR}/model_name.pkl")
    background = joblib.load(f"{MODELS_DIR}/background.pkl")

    if model_name == "random_forest":
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.LinearExplainer(model, background)

    return model, scaler, columns, numeric_cols, model_name, explainer


try:
    model, scaler, feature_columns, numeric_cols, model_name, explainer = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model artifacts not found. Run `python src/train.py` first to "
        "generate the files in `models/`, then relaunch the app."
    )
    st.stop()


# ---------------------------------------------------------------------------
# Sidebar — user inputs
# ---------------------------------------------------------------------------
st.sidebar.header("Customer Profile")

st.sidebar.subheader("Account")
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 18.0, 120.0, 70.0)
total_charges = st.sidebar.slider("Total Charges ($)", 0.0, 9000.0, float(tenure * monthly_charges))
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.sidebar.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
)

st.sidebar.subheader("Services")
internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
online_security = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])

with st.sidebar.expander("More service details"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

predict_clicked = st.sidebar.button("Predict Churn", type="primary", use_container_width=True)


# ---------------------------------------------------------------------------
# Build a raw single-row DataFrame matching the ORIGINAL (pre-dummy) schema
# ---------------------------------------------------------------------------
def build_raw_input() -> pd.DataFrame:
    row = {
        "gender": gender,
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }
    return pd.DataFrame([row])


def preprocess(raw_df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode, then reindex to the exact training column order,
    filling any columns absent from this single row (because that category
    wasn't present) with 0. Finally scale the numeric columns with the
    SAME fitted scaler used in training."""
    encoded = pd.get_dummies(raw_df, drop_first=True)
    encoded = encoded.reindex(columns=feature_columns, fill_value=0)
    encoded[numeric_cols] = scaler.transform(encoded[numeric_cols])
    return encoded


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------
st.title("📉 Customer Churn Predictor")
st.caption(
    f"Model: **{model_name.replace('_', ' ').title()}** · "
    "Trained on the Telco Customer Churn dataset"
)

if not predict_clicked:
    st.info("Set the customer's attributes in the sidebar, then click **Predict Churn**.")
else:
    raw_input = build_raw_input()
    processed_input = preprocess(raw_input)

    proba = model.predict_proba(processed_input)[0, 1]
    pct = proba * 100

    if proba < 0.35:
        risk_label, color = "Low Risk", "green"
    elif proba < 0.65:
        risk_label, color = "Medium Risk", "orange"
    else:
        risk_label, color = "High Risk", "red"

    col1, col2 = st.columns([1, 2])

    with col1:
        st.metric("Churn Probability", f"{pct:.1f}%")
        st.markdown(
            f"<span style='background-color:{color}; color:white; padding:6px 14px; "
            f"border-radius:20px; font-weight:600;'>{risk_label}</span>",
            unsafe_allow_html=True,
        )
        st.progress(min(int(pct), 100))

    with col2:
        st.subheader("Why this prediction?")
        st.caption("SHAP values show how each feature pushed the prediction up (red, toward churn) or down (blue, toward retention).")

        shap_values = explainer(processed_input)

        # Binary classifiers via shap can return a 3D array (samples, features, classes)
        # for tree models — select the "churn" (class 1) slice if present.
        sv = shap_values[0]
        if hasattr(sv, "values") and sv.values.ndim > 1:
            sv.values = sv.values[:, 1]
            sv.base_values = sv.base_values[1]

        fig, ax = plt.subplots(figsize=(8, 5))
        shap.plots.waterfall(sv, max_display=10, show=False)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with st.expander("View raw input sent to the model"):
        st.dataframe(raw_input.T.rename(columns={0: "Value"}))
