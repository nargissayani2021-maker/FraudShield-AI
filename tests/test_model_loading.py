import json
from unittest.mock import patch

import pytest

from fraudshield.config import MODEL_NAME, PRODUCTION_THRESHOLD
from fraudshield.model_loader import ArtifactLoadError, load_config, load_production_model, validate_pipeline


def write_config(path, **changes):
    values = {"model_name": MODEL_NAME, "threshold": PRODUCTION_THRESHOLD}
    values.update(changes)
    path.write_text(json.dumps(values), encoding="utf-8")
    return path


def test_valid_config(tmp_path):
    config = load_config(write_config(tmp_path / "xgb_config.json"))
    assert config.threshold == PRODUCTION_THRESHOLD


def test_missing_configuration(tmp_path):
    with pytest.raises(ArtifactLoadError, match="configuration is missing"):
        load_production_model(tmp_path / "model.pkl", tmp_path / "missing.json")


def test_missing_model(tmp_path):
    config = write_config(tmp_path / "xgb_config.json")
    with pytest.raises(ArtifactLoadError, match="model is missing"):
        load_production_model(tmp_path / "missing.pkl", config)


@pytest.mark.parametrize("contents", ["not json", "{}", "[]", '{"threshold": 0.85}'])
def test_malformed_configuration(tmp_path, contents):
    path = tmp_path / "config.json"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(ArtifactLoadError, match="incompatible"):
        load_config(path)


@pytest.mark.parametrize("changes", [{"threshold": 0.5}, {"threshold": "0.85"}, {"model_name": "FNN"}])
def test_unapproved_configuration(tmp_path, changes):
    with pytest.raises(ArtifactLoadError):
        load_config(write_config(tmp_path / "config.json", **changes))


def test_corrupt_model_has_safe_error(tmp_path):
    model = tmp_path / "model.json"
    model.write_bytes(b"not a model artifact")
    preprocessor = tmp_path / "preprocessor.joblib"
    preprocessor.write_bytes(b"not a preprocessor artifact")
    with pytest.raises(ArtifactLoadError, match="could not be loaded"):
        load_production_model(model, write_config(tmp_path / "config.json"), preprocessor)


def test_missing_preprocessor(tmp_path):
    model = tmp_path / "model.json"
    model.write_text("{}")
    with pytest.raises(ArtifactLoadError, match="preprocessor is missing"):
        load_production_model(model, write_config(tmp_path / "config.json"), tmp_path / "absent.joblib")


def test_wrong_model_rejected():
    with pytest.raises(ArtifactLoadError, match="incompatible"):
        validate_pipeline(object())


@pytest.mark.integration
def test_real_artifact_reload(production_resources, valid_frame):
    model, config = production_resources
    reloaded, reloaded_config = load_production_model()
    assert config == reloaded_config
    assert (model.predict_proba(valid_frame) == reloaded.predict_proba(valid_frame)).all()


@pytest.mark.integration
def test_normal_loading_never_reads_reference_pickle(production_resources):
    import joblib
    original_load = joblib.load
    with patch("fraudshield.model_loader.joblib.load", wraps=original_load) as read:
        load_production_model()
    assert len(read.call_args_list) == 1
    assert read.call_args.args[0].name == "fraudshield_preprocessor.joblib"
