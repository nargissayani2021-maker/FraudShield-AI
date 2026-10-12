import pytest

from fraudshield.config import OOD_LIMITS
from fraudshield.guardrails import check_ood_guardrails


def test_normal_input(valid_frame):
    assert check_ood_guardrails(valid_frame) == [()]


@pytest.mark.parametrize("feature,limit", OOD_LIMITS.items())
@pytest.mark.parametrize("offset,triggered", [(-0.000001, False), (0, False), (0.000001, True)])
def test_ood_boundaries(valid_frame, feature, limit, offset, triggered):
    valid_frame[feature] = limit + offset
    warnings = check_ood_guardrails(valid_frame)[0]
    assert bool(warnings) is triggered
    if triggered:
        assert feature in warnings[0]


def test_multiple_warnings(valid_frame):
    for feature, limit in OOD_LIMITS.items():
        valid_frame[feature] = limit + 1
    assert len(check_ood_guardrails(valid_frame)[0]) == 4
