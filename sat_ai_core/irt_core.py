"""
sat_ai_core/irt_core.py
-----------------------------------
Các hàm cốt lõi của IRT (Item Response Theory)
Dùng để ước lượng năng lực θ (theta) của thí sinh
và chọn câu hỏi tối ưu trong adaptive testing.
"""

import math
from typing import Dict, List, Tuple

# ===== Tham số chuẩn IRT =====
D = 1.7
THETA_BOUNDS = (-4.0, 4.0)

# ==============================
# 🧩 Sigmoid ổn định
# ==============================
def sigmoid_stable(x: float) -> float:
    """Phiên bản sigmoid ổn định số học."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)

# ==============================
# 📊 Xác suất trả lời đúng
# ==============================
def prob_correct(theta: float, a: float, b: float, c: float) -> float:
    """Tính xác suất thí sinh (θ) trả lời đúng một item (a,b,c)."""
    s = sigmoid_stable(D * a * (theta - b))
    return c + (1.0 - c) * s

# ==============================
# 🔍 Đạo hàm theo θ
# ==============================
def dprob_dtheta(theta: float, a: float, b: float, c: float) -> float:
    """Đạo hàm của P(θ) theo θ."""
    s = sigmoid_stable(D * a * (theta - b))
    return (1.0 - c) * D * a * s * (1.0 - s)

# ==============================
# 🧠 Fisher Information
# ==============================
def fisher_info(theta: float, a: float, b: float, c: float) -> float:
    """Tính thông tin Fisher cho một câu hỏi."""
    if a <= 0 or not (0 <= c < 1):
        return 0.0
    p = prob_correct(theta, a, b, c)
    if not (1e-6 < p < 1 - 1e-6):
        return 0.0
    dp = dprob_dtheta(theta, a, b, c)
    return (dp * dp) / (p * (1.0 - p))

# ==============================
# 🔁 Cập nhật θ (MAP Estimation)
# ==============================
def update_theta_map_once(
    theta: float,
    answered_items: List[Tuple[str, int]],
    irt_params: Dict[str, Dict[str, float]],
) -> Tuple[float, float]:
    """
    Cập nhật θ (theta) một bước theo công thức MAP.
    answered_items: [(item_id, is_correct)]
    irt_params: {id: {'a':..., 'b':..., 'c':...}}
    Trả về: (theta_new, standard_error)
    """
    U, I = 0.0, 0.0

    for item_id, resp in answered_items:
        pars = irt_params.get(str(item_id))
        if not pars:
            continue
        a, b, c = pars["a"], pars["b"], pars["c"]

        if a <= 0 or not (0 <= c < 1):
            continue

        p = prob_correct(theta, a, b, c)
        dp = dprob_dtheta(theta, a, b, c)
        if not (1e-6 < p < 1 - 1e-6):
            continue

        # Gradient & Fisher info tích lũy
        U += (resp - p) * dp / (p * (1.0 - p))
        I += (dp * dp) / (p * (1.0 - p))

    den = I + 1.0
    theta_new = theta + U / den
    theta_new = max(min(theta_new, THETA_BOUNDS[1]), THETA_BOUNDS[0])
    se = 1.0 / math.sqrt(den)

    return theta_new, se

# ==============================
# 🧪 Test nhanh
# ==============================
if __name__ == "__main__":
    irt_params = {
        "1": {"a": 1.0, "b": 0.0, "c": 0.2},
        "2": {"a": 1.2, "b": 0.5, "c": 0.2},
    }
    answered = [("1", 1), ("2", 0)]
    theta, se = update_theta_map_once(0.0, answered, irt_params)
    print(f"θ cập nhật: {theta:.3f} ± {se:.3f}")
