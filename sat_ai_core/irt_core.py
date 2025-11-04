"""
sat_ai_core/irt_core.py — Cải tiến
-----------------------------------
Các hàm cốt lõi của IRT (Item Response Theory)
Dùng để ước lượng năng lực θ (theta) của thí sinh
và chọn câu hỏi tối ưu trong adaptive testing (3PL model).
"""

import math
from typing import Dict, List, Tuple, Optional

# ===== Tham số chuẩn IRT =====
D = 1.7
THETA_BOUNDS = (-4.0, 4.0)

# ==============================
# 🧩 Sigmoid ổn định số học
# ==============================
def sigmoid_stable(x: float) -> float:
    """Phiên bản sigmoid ổn định cho x lớn hoặc nhỏ."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)

# ==============================
# 📊 Xác suất trả lời đúng (3PL)
# ==============================
def prob_correct(theta: float, a: float, b: float, c: float) -> float:
    """Tính xác suất P(θ) thí sinh trả lời đúng một item (a,b,c)."""
    s = sigmoid_stable(D * a * (theta - b))
    return c + (1.0 - c) * s

# ==============================
# 🔍 Đạo hàm theo θ
# ==============================
def dprob_dtheta(theta: float, a: float, b: float, c: float) -> float:
    """Đạo hàm của P(θ) theo θ (cho mô hình 3PL)."""
    s = sigmoid_stable(D * a * (theta - b))
    return (1.0 - c) * D * a * s * (1.0 - s)

# ==============================
# 🧠 Fisher Information
# ==============================
def fisher_info(theta: float, a: float, b: float, c: float) -> float:
    """Tính thông tin Fisher của một item tại θ."""
    if a <= 0 or not (0.0 <= c < 1.0) or not math.isfinite(b):
        return 0.0
    p = prob_correct(theta, a, b, c)
    if not (1e-6 < p < 1 - 1e-6):
        return 0.0
    dp = dprob_dtheta(theta, a, b, c)
    return (dp * dp) / (p * (1.0 - p))

# ==============================
# 🔁 Cập nhật θ (MAP Estimation)
# ==============================
def update_theta_map(
    theta: float,
    answered_items: List[Tuple[str, int]],
    irt_params: Dict[str, Dict[str, float]],
    prior_mean: float = 0.0,
    prior_var: float = 1.0,
    step_size: float = 1.0,
) -> Tuple[float, float]:
    """
    Cập nhật θ (theta) một bước theo công thức MAP thật sự.
    
    Args:
        theta: θ hiện tại
        answered_items: [(item_id, is_correct)] — danh sách câu đã trả lời
        irt_params: {item_id: {'a':..., 'b':..., 'c':...}}
        prior_mean: trung bình của prior (thường là 0)
        prior_var: phương sai của prior (thường là 1)
        step_size: hệ số học (giúp hội tụ ổn định hơn)
    
    Returns:
        (theta_new, standard_error)
    """
    U, I = 0.0, 0.0
    for item_id, resp in answered_items:
        if resp not in (0, 1):
            continue
        pars = irt_params.get(str(item_id))
        if not pars:
            continue

        a, b, c = pars.get("a", 1.0), pars.get("b", 0.0), pars.get("c", 0.0)
        if a <= 0 or not (0 <= c < 1):
            continue

        p = prob_correct(theta, a, b, c)
        if not (1e-6 < p < 1 - 1e-6):
            continue

        dp = dprob_dtheta(theta, a, b, c)
        # Gradient & Fisher info tích lũy
        U += (resp - p) * dp / (p * (1.0 - p))
        I += (dp * dp) / (p * (1.0 - p))

    # MAP update (với prior N(prior_mean, prior_var))
    prior_info = 1.0 / prior_var
    num = U - prior_info * (theta - prior_mean)
    den = I + prior_info

    if den == 0:
        return theta, float("inf")

    theta_new = theta + step_size * (num / den)
    theta_new = max(min(theta_new, THETA_BOUNDS[1]), THETA_BOUNDS[0])
    se = 1.0 / math.sqrt(den)

    return theta_new, se

# ==============================
# 🧮 Chọn câu hỏi tối ưu (MFI)
# ==============================
def select_next_item(
    theta: float,
    remaining_items: List[str],
    irt_params: Dict[str, Dict[str, float]],
) -> Optional[str]:
    """
    Chọn câu hỏi có thông tin Fisher cao nhất (Maximum Fisher Information rule).
    """
    best_item, best_info = None, -1.0
    for item_id in remaining_items:
        pars = irt_params.get(str(item_id))
        if not pars:
            continue
        info = fisher_info(theta, pars["a"], pars["b"], pars["c"])
        if info > best_info:
            best_info = info
            best_item = item_id
    return best_item

# ==============================
# 🧪 Test nhanh
# ==============================
if __name__ == "__main__":
    irt_params = {
        "1": {"a": 1.0, "b": 0.0, "c": 0.2},
        "2": {"a": 1.2, "b": 0.5, "c": 0.2},
        "3": {"a": 0.8, "b": -0.5, "c": 0.25},
    }

    answered = [("1", 1), ("2", 0)]
    theta, se = update_theta_map(0.0, answered, irt_params)
    print(f"θ cập nhật: {theta:.3f} ± {se:.3f}")

    remaining = ["3"]
    next_item = select_next_item(theta, remaining, irt_params)
    print(f"Câu hỏi nên chọn tiếp: {next_item}")
