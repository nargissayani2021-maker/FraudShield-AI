"""Canonical production contract established by the completed notebook."""

from dataclasses import dataclass
from pathlib import Path

MODEL_NAME = "Tuned XGBoost"
PRODUCTION_THRESHOLD = 0.85
MODEL_FEATURES = (
    "amt", "category", "gender", "state", "city_pop", "age", "hour",
    "day_of_week", "month", "is_weekend", "cardholder_merchant_distance_km",
    "time_since_prev_txn_min", "customer_prev_avg_amt", "amt_ratio_to_prev_avg",
    "txn_count_prev_1h", "no_prior_history",
)
CATEGORICAL_FEATURES = ("category", "gender", "state", "day_of_week")
BINARY_FEATURES = ("is_weekend", "no_prior_history")
NUMERICAL_FEATURES = tuple(
    name for name in MODEL_FEATURES
    if name not in CATEGORICAL_FEATURES + BINARY_FEATURES
)
# Existing notebook/application policy, observed over the full training CSV.
# These are not statistically validated OOD thresholds.
OOD_LIMITS = {
    "amt": 28948.90,
    "txn_count_prev_1h": 6,
    "amt_ratio_to_prev_avg": 676.39,
    "time_since_prev_txn_min": 22357.85,
}
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "fraudshield_xgboost_final.pkl"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "models" / "xgb_config.json"


@dataclass(frozen=True)
class ProductionConfig:
    model_name: str = MODEL_NAME
    threshold: float = PRODUCTION_THRESHOLD

    def __post_init__(self):
        if self.model_name != MODEL_NAME or self.threshold != PRODUCTION_THRESHOLD:
            raise ValueError("Configuration must match the approved production contract.")
