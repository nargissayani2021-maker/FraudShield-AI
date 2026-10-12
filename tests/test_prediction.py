from unittest.mock import Mock

import numpy as np
import pytest

from fraudshield.config import ProductionConfig
from fraudshield.prediction import PredictionError, classify_score, predict_transactions
from fraudshield.validation import InputValidationError


@pytest.mark.parametrize("score,expected", [(0, 0), (0.849999, 0), (0.85, 1), (0.850001, 1), (1, 1)])
def test_threshold(score, expected):
    assert classify_score(score) == expected


@pytest.mark.parametrize("score", [None, "0.9", True, np.nan, np.inf, -np.inf, -0.1, 1.1, [], {}])
def test_malformed_scores(score):
    with pytest.raises(PredictionError):
        classify_score(score)


def test_unapproved_threshold():
    with pytest.raises(PredictionError, match="Threshold"):
        classify_score(0.9, threshold=0.5)


@pytest.mark.parametrize("score,prediction", [(0.1, 0), (0.9, 1)])
def test_review_overrides_model_decision(valid_frame, score, prediction):
    # Mock only the inference boundary, never claim trained-model validation.
    model = Mock(classes_=np.array([0, 1]))
    model.predict_proba.return_value = [[1 - score, score]]
    valid_frame["txn_count_prev_1h"] = 30
    result = predict_transactions(valid_frame, model, ProductionConfig()).iloc[0]
    assert result["Fraud_Prediction"] == prediction
    assert result["Decision"] == "MANUAL REVIEW"


@pytest.mark.parametrize("score,decision", [(0.1, "NO FRAUD ALERT"), (0.85, "FRAUD ALERT")])
def test_normal_decision_and_canonical_input(valid_frame, score, decision):
    model = Mock(classes_=np.array([0, 1]))
    model.predict_proba.return_value = [[1 - score, score]]
    result = predict_transactions(valid_frame.iloc[:, ::-1], model, ProductionConfig())
    assert result.iloc[0]["Decision"] == decision
    assert list(model.predict_proba.call_args.args[0].columns) == list(valid_frame.columns)


def test_invalid_input_never_reaches_model(valid_frame):
    model = Mock()
    valid_frame["amt"] = np.nan
    with pytest.raises(InputValidationError):
        predict_transactions(valid_frame, model, ProductionConfig())
    model.predict_proba.assert_not_called()


@pytest.mark.parametrize("output", [[[0.1, np.nan]], [[0.1, 1.2]], [[0.1]], [[0.1, 0.9], [0.1, 0.9]]])
def test_bad_model_output(valid_frame, output):
    model = Mock(classes_=np.array([0, 1]))
    model.predict_proba.return_value = output
    with pytest.raises(PredictionError):
        predict_transactions(valid_frame, model, ProductionConfig())


def test_inference_error_is_safe(valid_frame):
    model = Mock(classes_=np.array([0, 1]))
    model.predict_proba.side_effect = ValueError("internal details")
    with pytest.raises(PredictionError, match="No automated decision") as error:
        predict_transactions(valid_frame, model, ProductionConfig())
    assert "internal details" not in str(error.value)


def test_ood_failure_keeps_manual_review_without_fabricated_score(valid_frame):
    model = Mock(classes_=np.array([0, 1]))
    model.predict_proba.side_effect = ValueError("internal details")
    valid_frame["txn_count_prev_1h"] = 30
    result = predict_transactions(valid_frame, model, ProductionConfig()).iloc[0]
    assert result["Decision"] == "MANUAL REVIEW"
    assert result["Fraud Risk Score"] is None
    assert result["Fraud_Prediction"] is None


@pytest.mark.integration
def test_real_pipeline_score_and_order(production_resources, valid_frame):
    model, config = production_resources
    first = predict_transactions(valid_frame, model, config)
    second = predict_transactions(valid_frame.iloc[:, ::-1], model, config)
    assert 0 <= first.iloc[0]["Fraud Risk Score"] <= 1
    assert first.equals(second)


@pytest.mark.integration
def test_real_pipeline_unknown_category(production_resources, valid_frame):
    model, config = production_resources
    valid_frame["category"] = "brand_new_unknown_category"
    result = predict_transactions(valid_frame, model, config)
    assert 0 <= result.iloc[0]["Fraud Risk Score"] <= 1
