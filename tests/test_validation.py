import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraudshield.config import (
    BINARY_FEATURES, CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERICAL_FEATURES,
)
from fraudshield.validation import (
    InputValidationError, validate_model_frame, validate_required_fields,
)


def test_schema_matches_notebook():
    notebook = json.loads((Path(__file__).parents[1] / "FraudShield_Final_Completed.ipynb").read_text(encoding="utf-8"))
    source = "".join(notebook["cells"][190]["source"])
    definition = ast.literal_eval(ast.parse(source).body[0].value)
    assert tuple(definition) == MODEL_FEATURES
    assert len(MODEL_FEATURES) == len(set(MODEL_FEATURES)) == 16
    assert set(CATEGORICAL_FEATURES + NUMERICAL_FEATURES + BINARY_FEATURES) == set(MODEL_FEATURES)


def test_valid_and_reordered_input(valid_frame):
    assert validate_required_fields(valid_frame.iloc[0].to_dict()) == []
    pd.testing.assert_frame_equal(validate_model_frame(valid_frame.iloc[:, ::-1]), valid_frame)


@pytest.mark.parametrize("feature", MODEL_FEATURES)
def test_missing_fields(valid_frame, feature):
    transaction = valid_frame.iloc[0].to_dict()
    del transaction[feature]
    assert validate_required_fields(transaction) == [feature]
    with pytest.raises(InputValidationError, match="Schema mismatch"):
        validate_model_frame(valid_frame.drop(columns=feature))


@pytest.mark.parametrize("value", [None, "", "  ", np.nan, pd.NA, pd.NaT])
@pytest.mark.parametrize("feature", MODEL_FEATURES)
def test_required_missing_values(valid_frame, feature, value):
    transaction = valid_frame.iloc[0].to_dict()
    transaction[feature] = value
    assert feature in validate_required_fields(transaction)
    with pytest.raises(InputValidationError, match="required values missing"):
        validate_model_frame(pd.DataFrame([transaction]))


@pytest.mark.parametrize("value", ["abc", "100", [], {}, True, np.inf, -np.inf])
def test_invalid_numeric(valid_frame, value):
    transaction = valid_frame.iloc[0].to_dict()
    transaction["amt"] = value
    with pytest.raises(InputValidationError):
        validate_model_frame(pd.DataFrame([transaction]))


def test_unknown_category_allowed(valid_frame):
    valid_frame["category"] = "brand_new_unknown_category"
    assert validate_model_frame(valid_frame)["category"].iloc[0] == "brand_new_unknown_category"


@pytest.mark.parametrize("feature", BINARY_FEATURES)
def test_binary_values(valid_frame, feature):
    valid_frame[feature] = 2
    with pytest.raises(InputValidationError, match="0 or 1"):
        validate_model_frame(valid_frame)


def test_extra_and_duplicate_columns(valid_frame):
    with pytest.raises(InputValidationError, match="unexpected"):
        validate_model_frame(valid_frame.assign(is_fraud=0))
    with pytest.raises(InputValidationError, match="Duplicate"):
        validate_model_frame(pd.concat([valid_frame, valid_frame[["amt"]]], axis=1))


@pytest.mark.parametrize("frame", [None, {}, pd.DataFrame()])
def test_invalid_frame(frame):
    with pytest.raises(InputValidationError):
        validate_model_frame(frame)
