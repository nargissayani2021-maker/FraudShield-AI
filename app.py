"""Thin Streamlit demo for already-engineered transaction inputs."""

import logging

import pandas as pd
import streamlit as st

from fraudshield.config import BINARY_FEATURES, CATEGORICAL_FEATURES, MODEL_FEATURES
from fraudshield.model_loader import ArtifactLoadError, load_production_model, trained_categorical_values
from fraudshield.prediction import PredictionError, predict_transactions
from fraudshield.validation import InputValidationError

FEATURE_LABELS = {
    "amt": "Transaction amount ($)",
    "category": "Transaction category",
    "gender": "Gender",
    "state": "State",
    "city_pop": "City population",
    "age": "Customer age at transaction",
    "hour": "Transaction hour",
    "day_of_week": "Day of week",
    "month": "Transaction month",
    "is_weekend": "Weekend indicator (0 or 1)",
    "cardholder_merchant_distance_km": "Cardholder–merchant distance (km)",
    "time_since_prev_txn_min": "Minutes since previous transaction",
    "customer_prev_avg_amt": "Previous average transaction amount ($)",
    "amt_ratio_to_prev_avg": "Amount-to-previous-average ratio",
    "txn_count_prev_1h": "Transactions in the previous hour",
    "no_prior_history": "No prior history indicator (0 or 1)",
}


@st.cache_resource
def load_resources():
    return load_production_model()


def main():
    st.set_page_config(page_title="FraudShield AI", page_icon="🛡️", layout="wide")
    st.title("FraudShield AI")
    st.caption("Tuned XGBoost · Credit-card fraud detection portfolio")
    st.write("Analyze an already-engineered transaction using the saved tuned XGBoost pipeline.")
    st.info("Supply verified historical inputs. This demo does not reconstruct customer history or impute missing values.")
    try:
        model, config = load_resources()
        choices = trained_categorical_values(model)
    except ArtifactLoadError as exc:
        st.error(str(exc))
        st.write("Supply the portable preprocessor, XGBoost JSON model, and xgb_config.json in models/. See README.md for setup.")
        return
    except Exception:
        logging.exception("Resource initialization failed")
        st.error("Production resources could not be initialized. See setup instructions.")
        return
    with st.form("transaction"):
        values = {}
        for feature in MODEL_FEATURES:
            label = FEATURE_LABELS[feature]
            if feature in CATEGORICAL_FEATURES:
                values[feature] = st.selectbox(label, choices[feature], index=None, key=feature)
            elif feature in BINARY_FEATURES:
                values[feature] = st.selectbox(label, (0, 1), index=None, key=feature)
            else:
                values[feature] = st.number_input(label, value=None, key=feature)
        submitted = st.form_submit_button("Analyze Transaction")
    if submitted:
        try:
            results = predict_transactions(pd.DataFrame([values]), model, config)
            result = results.iloc[0]
            score = result["Fraud Risk Score"]
            st.metric("Fraud Risk Score", "Unavailable" if pd.isna(score) else f"{score:.2%}")
            st.caption(f"Operating threshold: {config.threshold}")
            model_prediction = result["Fraud_Prediction"]
            st.write("**Binary model prediction:** " + (
                "Unavailable" if pd.isna(model_prediction) else
                f"{int(model_prediction)} — " + ("Fraud alert" if model_prediction else "No fraud alert")
            ))
            st.subheader("Application decision")
            if result["Decision"] == "MANUAL REVIEW":
                st.warning("MANUAL REVIEW REQUIRED")
                st.write("An input exceeds the notebook guardrail policy. OOD does not mean fraud; manual review takes priority over the model prediction.")
                for warning in result["OOD Warnings"]:
                    st.write(warning)
            elif result["Fraud_Prediction"]:
                st.error("FRAUD ALERT")
            else:
                st.success("NO FRAUD ALERT")
        except (InputValidationError, PredictionError) as exc:
            st.error(str(exc))
        except Exception:
            logging.exception("Transaction analysis failed")
            st.error("Transaction analysis is unavailable. No automated decision was made.")


if __name__ == "__main__":
    main()
