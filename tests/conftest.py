"""Controlled input fixtures; no fake trained models."""

import pandas as pd
import pytest

from fraudshield.config import DEFAULT_CONFIG_PATH, DEFAULT_MODEL_PATH, DEFAULT_PREPROCESSOR_PATH, MODEL_FEATURES
from fraudshield.model_loader import load_production_model


@pytest.fixture
def valid_frame():
    # Notebook QA Test 2's synthetic input, not a claim of production accuracy.
    values = (100.0, "grocery_pos", "M", "TX", 100000, 40, 12, "Monday", 6, 0,
              50.0, 120.0, 75.0, 100.0 / 75.0, 0, 0)
    return pd.DataFrame([dict(zip(MODEL_FEATURES, values))])


@pytest.fixture
def portable_reference_frame(valid_frame):
    """Exact synthetic Colab validation input confirmed by the user."""
    frame = valid_frame.copy()
    frame["amt_ratio_to_prev_avg"] = 1.333
    return frame


@pytest.fixture
def production_resources():
    if not all(path.is_file() for path in (DEFAULT_MODEL_PATH, DEFAULT_CONFIG_PATH, DEFAULT_PREPROCESSOR_PATH)):
        pytest.skip("Portable production model, preprocessor and/or xgb_config.json are unavailable.")
    return load_production_model()
