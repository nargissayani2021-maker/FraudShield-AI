"""Thin Streamlit demo for already-engineered transaction inputs."""

import logging

import pandas as pd
import streamlit as st

from fraudshield.config import BINARY_FEATURES, CATEGORICAL_FEATURES, MODEL_FEATURES, OOD_LIMITS
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

# Synthetic UI presets only; scores and decisions always come from inference.
LOW_RISK_EXAMPLE = dict(zip(MODEL_FEATURES, (
    100.0, "grocery_pos", "M", "TX", 100000, 40, 12, "Monday", 6, 0,
    50.0, 120.0, 75.0, 1.333, 0, 0,
)))
DEMO_EXAMPLES = {
    "Low-Risk Example": LOW_RISK_EXAMPLE,
    # Completed notebook's synthetic explainability scenario (cell 480).
    "High-Risk Example": {
        **LOW_RISK_EXAMPLE, "amt": 2000.0,
        "cardholder_merchant_distance_km": 1000.0,
        "customer_prev_avg_amt": 25.0, "amt_ratio_to_prev_avg": 80.0,
    },
    "OOD / Manual Review Example": {
        **LOW_RISK_EXAMPLE,
        "txn_count_prev_1h": OOD_LIMITS["txn_count_prev_1h"] + 1,
    },
}


def populate_example(name):
    """Populate existing widgets; do not bypass validation or calculate results."""
    st.session_state.update(DEMO_EXAMPLES[name])
    st.session_state["model_inputs_expanded"] = True
    st.session_state["selected_example"] = name


@st.cache_resource
def load_resources():
    return load_production_model()


def main():
    st.set_page_config(page_title="FraudShield AI", page_icon="🛡️", layout="wide")
    st.title("FraudShield AI")
    st.caption("Tuned XGBoost · Credit-card fraud detection portfolio")
    st.write("FraudShield scores already-engineered transaction features. "
             "In a real production system, these features would be assembled automatically "
             "from transaction data and customer/card history.")
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
    st.caption("Examples are synthetic/demo engineered inputs and do not represent real customers.")
    for column, name in zip(st.columns(3), DEMO_EXAMPLES):
        column.button(name, on_click=populate_example, args=(name,))
    selected_example = st.session_state.get("selected_example")
    if selected_example:
        st.info(f"{selected_example} loaded. Review or edit the inputs below, then select Analyze Transaction.")
    # Reserve the result's visual position before rendering the editable form.
    result_area = st.container()
    with st.form("transaction"):
        values = {}
        with st.expander("Advanced: View / Edit Model Inputs",
                         expanded=st.session_state.get("model_inputs_expanded", False)):
            transaction_column, history_column = st.columns(2)
            transaction_column.subheader("Transaction / Customer Features")
            history_column.subheader("Historical Behavior Features")
            for position, feature in enumerate(MODEL_FEATURES):
                with transaction_column if position < 11 else history_column:
                    label = FEATURE_LABELS[feature]
                    if feature in CATEGORICAL_FEATURES:
                        values[feature] = st.selectbox(label, choices[feature], index=None, key=feature)
                    elif feature in BINARY_FEATURES:
                        values[feature] = st.selectbox(label, (0, 1), index=None, key=feature)
                    else:
                        values[feature] = st.number_input(label, value=None, key=feature)
        submitted = st.form_submit_button("Analyze Transaction")
    if submitted:
        with result_area:
            render_analysis(values, model, config)


def render_analysis(values, model, config):
    """Present the existing inference result above the advanced input form."""
    st.subheader("Analysis Result")
    try:
        results = predict_transactions(pd.DataFrame([values]), model, config)
        result = results.iloc[0]
        score = result["Fraud Risk Score"]
        model_prediction = result["Fraud_Prediction"]
        score_column, prediction_column, decision_column = st.columns(3)
        score_column.metric("Fraud Risk Score", "Unavailable" if pd.isna(score) else f"{score:.2%}")
        prediction_column.metric("Model Prediction", (
            "Unavailable" if pd.isna(model_prediction) else
            ("Fraud Alert" if model_prediction else "No Fraud Alert")
        ))
        decision_column.metric("Application Decision", result["Decision"])
        st.caption(f"Operating threshold: {config.threshold}")
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
