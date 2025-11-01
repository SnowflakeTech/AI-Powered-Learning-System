# tests/test_irt_engine.py

import math
import pytest

from core.irt_engine import prob_correct, fisher_info, update_theta_map_once
from core.schema import IRTParams


def test_prob_correct_range():
    pars = IRTParams(id=1, a=1.2, b=0.0, c=0.2)

    for theta in [-4, -2, 0, 2, 4]:
        p = prob_correct(theta, pars.a, pars.b, pars.c)
        assert 0.0 < p < 1.0, f"P(θ) không nằm trong (0,1): θ={theta}, p={p}"


def test_fisher_info_positive():
    pars = IRTParams(id=1, a=1.0, b=0.0, c=0.2)

    # θ gần độ khó (b=0.0) → nên có info cao
    info = fisher_info(0.0, pars)
    assert info > 0.01, f"Fisher info quá thấp: {info}"


def test_map_update_theta_increases_if_correct():
    pars = {
        1: IRTParams(id=1, a=1.2, b=0.0, c=0.2)
    }
    theta0 = 0.0
    answered = [(1, 1)]  # trả lời đúng

    theta1, se1 = update_theta_map_once(theta0, answered, pars)
    
    assert theta1 > theta0, "θ phải tăng khi trả lời đúng"
    assert math.isfinite(se1), "SE phải là số hữu hạn"
