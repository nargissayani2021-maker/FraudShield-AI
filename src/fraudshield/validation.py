"""Validate already-engineered inputs without inventing history policies."""

from collections.abc import Mapping
from numbers import Real

import numpy as np
import pandas as pd

from .config import BINARY_FEATURES, CATEGORICAL_FEATURES, MODEL_FEATURES


class InputValidationError(ValueError):
    """Input cannot safely reach the saved model."""


def validate_required_fields(transaction):
    """Return missing or empty required fields, including pandas null values."""
    if not isinstance(transaction, Mapping):
        raise InputValidationError("A transaction must be a mapping of named features.")
    missing = []
    for name in MODEL_FEATURES:
        if name not in transaction:
            missing.append(name)
            continue
        value = transaction[name]
        if value is None or (isinstance(value, str) and not value.strip()):
            missing.append(name)
        elif pd.api.types.is_scalar(value) and bool(pd.isna(value)):
            missing.append(name)
    return missing


def validate_model_frame(frame):
    """Return a copy in canonical order; reject malformed schema and values."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise InputValidationError("Provide a nonempty transaction DataFrame.")
    if not frame.columns.is_unique:
        raise InputValidationError("Duplicate feature columns are not allowed.")
    missing = [name for name in MODEL_FEATURES if name not in frame.columns]
    extra = [str(name) for name in frame.columns if name not in MODEL_FEATURES]
    if missing or extra:
        raise InputValidationError(f"Schema mismatch. Missing: {missing}; unexpected: {extra}.")
    for position, row in enumerate(frame.to_dict(orient="records")):
        absent = validate_required_fields(row)
        if absent:
            raise InputValidationError(f"Row {position}: required values missing: {absent}.")
        for name, value in row.items():
            if name in CATEGORICAL_FEATURES:
                if not isinstance(value, str):
                    raise InputValidationError(f"Row {position}: {name} must be text.")
                # Unknown nonblank categories remain supported by the fitted encoder.
            elif isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                raise InputValidationError(f"Row {position}: {name} must be numeric.")
            elif not np.isfinite(value):
                raise InputValidationError(f"Row {position}: {name} must be finite.")
            elif name in BINARY_FEATURES and value not in (0, 1):
                raise InputValidationError(f"Row {position}: {name} must be 0 or 1.")
    return frame.loc[:, list(MODEL_FEATURES)].copy()
