"""Validated inference and separate model/application decisions."""

from numbers import Real

import numpy as np
import pandas as pd

from .config import PRODUCTION_THRESHOLD, ProductionConfig
from .guardrails import check_ood_guardrails
from .validation import validate_model_frame


class PredictionError(RuntimeError):
    """A safe fraud score could not be generated."""


def classify_score(score, threshold=PRODUCTION_THRESHOLD):
    if isinstance(threshold, bool) or threshold != PRODUCTION_THRESHOLD:
        raise PredictionError("Threshold does not match the approved configuration.")
    if isinstance(score, (bool, np.bool_)) or not isinstance(score, Real):
        raise PredictionError("Fraud Risk Score must be numeric.")
    if not np.isfinite(score) or not 0 <= score <= 1:
        raise PredictionError("Fraud Risk Score must be finite and within 0–1.")
    return int(score >= threshold)


def predict_transactions(frame, model, config: ProductionConfig):
    validated = validate_model_frame(frame)
    warnings = check_ood_guardrails(validated)
    try:
        if not np.array_equal(model.classes_, [0, 1]):
            raise PredictionError("Unexpected model output classes.")
        probabilities = np.asarray(model.predict_proba(validated))
        if probabilities.shape != (len(validated), 2):
            raise PredictionError("Unexpected model score shape.")
        scores = probabilities[:, 1]
        predictions = [classify_score(score, config.threshold) for score in scores]
    except Exception as exc:
        # OOD review remains available even when a score cannot be calculated.
        # Mixed/in-range batches fail safely rather than receiving guessed labels.
        if all(warnings):
            return pd.DataFrame({
                "Fraud Risk Score": [None] * len(validated),
                "Fraud_Prediction": [None] * len(validated),
                "Decision": ["MANUAL REVIEW"] * len(validated),
                "OOD Warnings": warnings,
            }, index=validated.index)
        if isinstance(exc, PredictionError):
            raise
        raise PredictionError("Inference failed. No automated decision is available.") from exc
    decisions = ["MANUAL REVIEW" if warning else
                 "FRAUD ALERT" if prediction else "NO FRAUD ALERT"
                 for warning, prediction in zip(warnings, predictions)]
    return pd.DataFrame({
        "Fraud Risk Score": scores,
        "Fraud_Prediction": predictions,
        "Decision": decisions,
        "OOD Warnings": warnings,
    }, index=validated.index)
