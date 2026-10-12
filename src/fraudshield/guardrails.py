"""Notebook OOD policy; warnings never change the binary fraud label."""

from .config import OOD_LIMITS
from .validation import validate_model_frame


def check_ood_guardrails(frame):
    """Return warning tuples per row after validating the final model schema."""
    validated = validate_model_frame(frame)
    return [
        tuple(f"{name} exceeds the notebook guardrail limit ({limit})."
              for name, limit in OOD_LIMITS.items() if row[name] > limit)
        for row in validated.to_dict(orient="records")
    ]
