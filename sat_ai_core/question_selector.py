"""
sat_ai_core/question_selector.py
-----------------------------------
Module chọn câu hỏi tối ưu dựa trên mô hình IRT (Item Response Theory).
Kết hợp thông tin Fisher, độ khó phù hợp, và trọng số kỹ năng yếu để
tăng tính cá nhân hóa cho bài thi thích ứng (Adaptive Testing).
"""

import random
from typing import List, Dict, Any, Optional
from .irt_core import fisher_info

def select_next_item(
    theta: float,
    asked_ids: List[str],
    items: List[Dict[str, Any]],
    irt_params: Dict[str, Dict[str, float]],
    *,
    history: Optional[List[Dict[str, Any]]] = None,
    focus_skill: Optional[str] = None,
    top_k: int = 4,
) -> Optional[Dict[str, Any]]:
    """
    Chọn câu hỏi tiếp theo dựa vào:
    - Fisher Information (độ nhạy của câu hỏi với θ hiện tại)
    - Độ khó phù hợp (|θ - b| nhỏ)
    - Kỹ năng yếu được ưu tiên (nếu có history)
    - Trọng số focus_skill nếu người dùng chọn chủ đề cụ thể

    Parameters
    ----------
    theta : float
        Năng lực hiện tại của thí sinh.
    asked_ids : list[str]
        Danh sách ID các câu đã hỏi (để tránh trùng).
    items : list[dict]
        Toàn bộ ngân hàng câu hỏi (phải chứa id, skill, choices, answer_index,...).
    irt_params : dict
        Tham số IRT cho từng câu hỏi {id: {"a":..., "b":..., "c":...}}.
    history : list[dict], optional
        Lịch sử các câu đã làm, mỗi phần tử có {"skill":..., "answered_correctly": bool}.
    focus_skill : str, optional
        Kỹ năng muốn tập trung (nếu có).
    top_k : int
        Số câu top theo Fisher info để chọn ngẫu nhiên 1 câu cuối cùng.

    Returns
    -------
    dict | None
        Câu hỏi được chọn, hoặc None nếu hết câu phù hợp.
    """

    # 1️⃣ Xác định kỹ năng yếu dựa vào lịch sử
    skill_wrong: Dict[str, int] = {}
    if history:
        for it in history:
            skill = it.get("skill", "Unknown")
            skill_wrong.setdefault(skill, 0)
            if not it.get("answered_correctly", True):
                skill_wrong[skill] += 1

    def skill_weight(skill: str) -> float:
        """
        Tính trọng số ưu tiên cho kỹ năng.
        - Nếu khác focus_skill → giảm 50% độ ưu tiên.
        - Nếu là kỹ năng sai nhiều → tăng điểm.
        """
        base = 1.0 + 0.5 * skill_wrong.get(skill, 0)
        if focus_skill and skill != focus_skill:
            base *= 0.5
        return base

    candidates = []

    # 2️⃣ Duyệt qua toàn bộ câu hỏi, tính điểm cho từng câu
    for item in items:
        item_id = str(item.get("id"))
        if not item_id or item_id in asked_ids:
            continue

        pars = irt_params.get(item_id)
        if not pars:
            continue

        a, b, c = pars["a"], pars["b"], pars["c"]

        # Fisher Information tại θ hiện tại
        info = fisher_info(theta, a, b, c)
        if info <= 0:
            continue

        # Độ phù hợp độ khó (θ gần b)
        diff_fit = 1.0 / (1.0 + abs(theta - b))

        # Trọng số theo kỹ năng yếu
        skill = item.get("skill", "Unknown")
        weight = skill_weight(skill)

        # Tổng hợp điểm ưu tiên
        final_score = info * diff_fit * weight
        candidates.append((final_score, item))

    # 3️⃣ Không có ứng viên phù hợp → kết thúc
    if not candidates:
        return None

    # 4️⃣ Lấy top_k câu có điểm cao nhất
    candidates.sort(key=lambda x: x[0], reverse=True)
    top_candidates = [itm for _, itm in candidates[:top_k]]

    # 5️⃣ Chọn ngẫu nhiên 1 câu trong nhóm top_k
    return random.choice(top_candidates)

if __name__ == "__main__":
    from .irt_core import update_theta
    items = [
        {"id": "1", "skill": "Algebra"},
        {"id": "2", "skill": "Geometry"},
        {"id": "3", "skill": "Functions"},
    ]
    irt_params = {
        "1": {"a": 1.2, "b": 0.0, "c": 0.2},
        "2": {"a": 1.0, "b": 1.0, "c": 0.2},
        "3": {"a": 0.8, "b": -0.5, "c": 0.25},
    }
    history = [
        {"skill": "Algebra", "answered_correctly": False},
        {"skill": "Geometry", "answered_correctly": True},
    ]

    q = select_next_item(theta=0.2, asked_ids=[], items=items, irt_params=irt_params, history=history, focus_skill=None)
    print("\n🎯 Câu được chọn:", q)
