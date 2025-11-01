# core/irt_engine.py

import math
from typing import Dict, List, Tuple
from .schema import IRTParams

# Hệ số logistic chuẩn của IRT 3PL
D = 1.7
THETA_MIN, THETA_MAX = -4.0, 4.0
EPS = 1e-6


def sigmoid_stable(x: float) -> float:
    """
    Sigmoid ổn định số học: tránh overflow exp(x)
    """
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def prob_correct(theta: float, a: float, b: float, c: float) -> float:
    """Xác suất trả lời đúng trong mô hình 3PL."""
    z = D * a * (theta - b)
    s = sigmoid_stable(z)
    return c + (1.0 - c) * s


def dprob_dtheta(theta: float, a: float, b: float, c: float) -> float:
    """Đạo hàm theo θ."""
    z = D * a * (theta - b)
    s = sigmoid_stable(z)
    return (1.0 - c) * D * a * s * (1.0 - s)


def fisher_info(theta: float, pars: IRTParams) -> float:
    """Fisher Information I(θ) cho 3PL."""
    a, b, c = pars.a, pars.b, pars.c
    if a <= 0 or not (0 <= c < 1):
        return 0.0

    p = prob_correct(theta, a, b, c)
    if p <= EPS or p >= 1 - EPS:
        return 0.0

    dp = dprob_dtheta(theta, a, b, c)
    return (dp * dp) / (p * (1.0 - p))


def update_theta_map_once(
    theta: float,
    answered_pairs: List[Tuple[int, int]],  # (item_id, correct)
    irt_params: Dict[int, IRTParams],
    prior_mean: float = 0.0,
    prior_var: float = 1.0
) -> Tuple[float, float]:
    """
    Một bước MAP Fisher scoring:
    Trả về (theta_new, SE)
    """
    U = 0.0  # Score
    I = 0.0  # Fisher
    prior_prec = 1.0 / prior_var

    for item_id, resp in answered_pairs:
        if item_id not in irt_params:
            continue
        pars = irt_params[item_id]
        a, b, c = pars.a, pars.b, pars.c
        p = prob_correct(theta, a, b, c)
        if not (EPS < p < 1.0 - EPS):
            continue
        dp = dprob_dtheta(theta, a, b, c)

        U += (resp - p) * dp / (p * (1.0 - p))
        I += (dp * dp) / (p * (1.0 - p))

    den = I + prior_prec
    if den <= 0:
        return theta, float("inf")

    theta_new = theta + (U - (theta - prior_mean) * prior_prec) / den
    theta_new = min(max(theta_new, THETA_MIN), THETA_MAX)

    se = 1.0 / math.sqrt(den)
    return theta_new, se
