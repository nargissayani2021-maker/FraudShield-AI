"""Presets fill the UI but use actual validation and portable model inference."""

import pandas as pd
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

from app import DEMO_EXAMPLES
from fraudshield.config import MODEL_FEATURES, OOD_LIMITS, PRODUCTION_THRESHOLD
from fraudshield.prediction import predict_transactions
from fraudshield.validation import validate_model_frame


@pytest.mark.parametrize("name", DEMO_EXAMPLES)
def test_demo_schema(name):
    assert tuple(DEMO_EXAMPLES[name]) == MODEL_FEATURES
    validate_model_frame(pd.DataFrame([DEMO_EXAMPLES[name]]))


@pytest.mark.integration
@pytest.mark.parametrize("name", DEMO_EXAMPLES)
def test_demo_real_inference(production_resources, name):
    model, config = production_resources
    result = predict_transactions(pd.DataFrame([DEMO_EXAMPLES[name]]), model, config).iloc[0]
    if name == "Low-Risk Example":
        assert result["Fraud Risk Score"] == pytest.approx(0.0003787238, rel=0, abs=5e-11)
        assert result["Fraud Risk Score"] < PRODUCTION_THRESHOLD
        assert result["Decision"] == "NO FRAUD ALERT"
    elif name == "High-Risk Example":
        assert result["Fraud Risk Score"] >= PRODUCTION_THRESHOLD
        assert result["OOD Warnings"] == ()
        assert result["Decision"] == "FRAUD ALERT"
    else:
        assert DEMO_EXAMPLES[name]["txn_count_prev_1h"] > OOD_LIMITS["txn_count_prev_1h"]
        assert len(result["OOD Warnings"]) == 1
        assert "txn_count_prev_1h" in result["OOD Warnings"][0]
        assert result["Decision"] == "MANUAL REVIEW"


@pytest.mark.integration
@pytest.mark.parametrize("name", DEMO_EXAMPLES)
def test_demo_buttons_fill_and_score(production_resources, name):
    model, config = production_resources
    expected = predict_transactions(pd.DataFrame([DEMO_EXAMPLES[name]]), model, config).iloc[0]
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run(timeout=15)
    next(button for button in app.button if button.label == name).click().run(timeout=15)
    assert not app.exception
    for widget in list(app.selectbox) + list(app.number_input):
        assert widget.value == DEMO_EXAMPLES[name][widget.key]
    assert not app.metric  # Selecting a sample does not hard-code a prediction.
    next(button for button in app.button if button.label == "Analyze Transaction").click().run(timeout=15)
    assert not app.exception
    assert app.metric[0].value == f"{expected['Fraud Risk Score']:.2%}"
    if name == "High-Risk Example":
        assert app.error[0].value == expected["Decision"]
    elif name == "Low-Risk Example":
        assert app.success[0].value == expected["Decision"]
    else:
        assert app.warning[0].value == "MANUAL REVIEW REQUIRED"
