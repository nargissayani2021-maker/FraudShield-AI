"""Load existing trusted artifacts; never train or substitute a model."""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted

from .config import (
    BINARY_FEATURES, CATEGORICAL_FEATURES, DEFAULT_CONFIG_PATH, DEFAULT_MODEL_PATH,
    DEFAULT_PREPROCESSOR_PATH,
    MODEL_FEATURES, NUMERICAL_FEATURES, ProductionConfig,
)


class ArtifactLoadError(RuntimeError):
    """Safe user-facing artifact failure."""


def load_config(path=DEFAULT_CONFIG_PATH):
    path = Path(path)
    if not path.is_file():
        raise ArtifactLoadError(f"Production configuration is missing: {path.name}.")
    try:
        values = json.loads(path.read_text(encoding="utf-8"))
        return ProductionConfig(model_name=values["model_name"], threshold=values["threshold"])
    except Exception as exc:
        raise ArtifactLoadError("Production configuration is unreadable or incompatible.") from exc


def validate_pipeline(model):
    """Verify notebook pipeline boundaries without rebuilding preprocessing."""
    try:
        if not isinstance(model, Pipeline) or list(model.named_steps) != ["preprocessor", "model"]:
            raise ValueError("Unexpected pipeline structure")
        preprocessor = model.named_steps["preprocessor"]
        classifier = model.named_steps["model"]
        # Imported only when an artifact is actually available.
        from xgboost import XGBClassifier
        if not isinstance(preprocessor, ColumnTransformer) or not isinstance(classifier, XGBClassifier):
            raise ValueError("Unexpected production components")
        check_is_fitted(preprocessor)
        check_is_fitted(classifier)
        if tuple(preprocessor.feature_names_in_) != MODEL_FEATURES:
            raise ValueError("Feature contract mismatch")
        transforms = {name: (transform, tuple(columns))
                      for name, transform, columns in preprocessor.transformers_
                      if name != "remainder"}
        encoder, categories = transforms["cat"]
        if not isinstance(encoder, OneHotEncoder) or encoder.handle_unknown != "ignore":
            raise ValueError("Unexpected categorical encoding")
        if categories != CATEGORICAL_FEATURES:
            raise ValueError("Categorical contract mismatch")
        for name, columns in (("num", NUMERICAL_FEATURES), ("binary", BINARY_FEATURES)):
            # Newer sklearn versions materialize passthrough as FunctionTransformer.
            original = {key: (transform, tuple(cols))
                        for key, transform, cols in preprocessor.transformers}
            if original[name][0] != "passthrough" or original[name][1] != columns:
                raise ValueError("Unexpected numerical/binary transformation")
            if transforms[name][1] != columns:
                raise ValueError("Fitted feature group mismatch")
        if not np.array_equal(model.classes_, [0, 1]) or not callable(model.predict_proba):
            raise ValueError("Unexpected output classes")
        if len(preprocessor.get_feature_names_out()) != classifier.n_features_in_:
            raise ValueError("Preprocessor output does not match classifier input width")
    except Exception as exc:
        raise ArtifactLoadError("Saved model is incompatible with the production contract.") from exc


def load_production_model(model_path=DEFAULT_MODEL_PATH, config_path=DEFAULT_CONFIG_PATH,
                          preprocessor_path=DEFAULT_PREPROCESSOR_PATH):
    """Compose exported fitted components in memory; never read the reference pickle."""
    config = load_config(config_path)
    path = Path(model_path)
    if not path.is_file():
        raise ArtifactLoadError(f"Production model is missing: {path.name}.")
    preprocessor_path = Path(preprocessor_path)
    if not preprocessor_path.is_file():
        raise ArtifactLoadError(f"Production preprocessor is missing: {preprocessor_path.name}.")
    try:
        from sklearn.exceptions import InconsistentVersionWarning
        from xgboost import XGBClassifier
        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)
            preprocessor = joblib.load(preprocessor_path)
        classifier = XGBClassifier()
        classifier.load_model(path)
        model = Pipeline([("preprocessor", preprocessor), ("model", classifier)])
    except Exception as exc:
        raise ArtifactLoadError("Portable production artifacts could not be loaded. Check artifact and environment compatibility.") from exc
    validate_pipeline(model)
    return model, config


def trained_categorical_values(model):
    """Read verified choices from the loaded fitted encoder, not a hard-coded list."""
    encoder = model.named_steps["preprocessor"].named_transformers_["cat"]
    return {name: tuple(values) for name, values in zip(CATEGORICAL_FEATURES, encoder.categories_)}
