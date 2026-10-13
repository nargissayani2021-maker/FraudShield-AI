"""Approved full-dataset regression; local datasets remain excluded from Git."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)

from fraudshield.config import MODEL_FEATURES, PRODUCTION_THRESHOLD
from fraudshield.validation import validate_model_frame

REGRESSION_PATH = Path(__file__).resolve().parents[1] / "data" / "fraudshield_regression_test.parquet"


@pytest.mark.integration
def test_approved_full_regression(production_resources):
    if not REGRESSION_PATH.is_file():
        pytest.skip("Approved local engineered regression parquet is unavailable.")
    model, config = production_resources
    data = pd.read_parquet(REGRESSION_PATH)
    assert len(data) == 555719, "Approved regression row count differs."
    assert data.columns.is_unique, "Duplicate regression columns."
    assert list(data.columns) == list(MODEL_FEATURES) + ["is_fraud"], "Regression schema differs."
    labels = data["is_fraud"]
    assert labels.notna().all() and labels.isin([0, 1]).all(), "Invalid regression labels."
    frame = validate_model_frame(data.loc[:, list(MODEL_FEATURES)])
    assert config.threshold == PRODUCTION_THRESHOLD
    # Validate the binary model baseline independently of OOD application routing.
    scores = model.predict_proba(frame)[:, 1]
    assert scores.shape == (len(data),)
    assert np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all()
    predictions = (scores >= config.threshold).astype(int)
    actual = confusion_matrix(labels, predictions, labels=[0, 1])
    np.testing.assert_array_equal(
        actual, [[552805, 769], [242, 1903]],
        err_msg="Approved confusion matrix differs; investigate rather than updating the baseline.",
    )
    # Approved metrics are recorded to four decimals, so compare at that precision.
    metrics = {
        "precision": (precision_score(labels, predictions), 0.7122),
        "recall": (recall_score(labels, predictions), 0.8872),
        "F1": (f1_score(labels, predictions), 0.7901),
        "ROC-AUC": (roc_auc_score(labels, scores), 0.9981),
        "PR-AUC": (average_precision_score(labels, scores), 0.9058),
    }
    for name, (value, expected) in metrics.items():
        assert value == pytest.approx(expected, rel=0, abs=5e-5), f"Approved {name} differs."
