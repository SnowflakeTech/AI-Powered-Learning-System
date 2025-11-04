"""
sat_ai_core/question_selector_v2.py
-----------------------------------
Phiên bản nâng cấp của bộ chọn câu hỏi thích ứng IRT.
Thêm giới hạn độ khó, cơ chế cooldown kỹ năng, log chi tiết,
và tham số điều chỉnh trọng số linh hoạt.
"""

import random
from typing import List, Dict, Any, Optional
from rich.console import Console
from .irt_core import fisher_info

console = Console()


def select_next_item(
    theta: float,
    asked_ids: List[str],
    items: List[Dict[str, Any]],
    irt_params: Dict[str, Dict[str, float]],
    *,
    history: Optional[List[Dict[str, Any]]] = None,
    focus_skill: Optional[str] = None,
    top_k: int = 4,
    alpha: float = 1.0,    # hệ số cho Fisher info
    beta: float = 0.8,     # hệ số cho độ phù hợp độ khó
    gamma: float = 1.2,    # hệ số cho trọng số kỹ năng yếu
    difficulty_range: float = 2.0,
    verbose: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Chọn câu hỏi tiếp theo trong Adaptive Testing dựa trên IRT.

    Tham số:
    ----------
    theta : float
        Năng lực hiện tại của học sinh.
    asked_ids : list[str]
        Các câu hỏi đã hỏi.
    items : list[dict]
        Ngân hàng câu hỏi.
    irt_params : dict
        Tham số IRT cho từng câu hỏi (a, b, c).
    history : list[dict], optional
        Lịch sử câu hỏi.
    focus_skill : str, optional
        Kỹ năng được ưu tiên.
    top_k : int
        Chọn ngẫu nhiên 1 câu trong top_k điểm cao nhất.
    """

    # 1️⃣ Thống kê kỹ năng sai nhiều
    skill_wrong: Dict[str, int] = {}
    if history:
        for it in history:
            skill = it.get("skill", "Unknown")
            skill_wrong.setdefault(skill, 0)
            if not it.get("answered_correctly", True):
                skill_wrong[skill] += 1

    # 2️⃣ Xác định kỹ năng vừa xuất hiện gần nhất (để cooldown)
    last_skill = history[-1]["skill"] if history else None

    def skill_weight(skill: str) -> float:
        """Tính trọng số ưu tiên cho kỹ năng."""
        base = 1.0 + gamma * skill_wrong.get(skill, 0)
        if focus_skill and skill != focus_skill:
            base *= 0.5
        if skill == last_skill:  # cooldown giảm 30%
            base *= 0.7
        return base

    candidates = []

    # 3️⃣ Duyệt toàn bộ câu hỏi và tính điểm
    for item in items:
        item_id = str(item.get("id"))
        if not item_id or item_id in asked_ids:
            continue

        pars = irt_params.get(item_id)
        if not pars:
            continue

        a, b, c = pars["a"], pars["b"], pars["c"]
        skill = item.get("skill", "Unknown")

        # Giới hạn độ khó trong khoảng phù hợp
        if abs(theta - b) > difficulty_range:
            continue

        info = fisher_info(theta, a, b, c)
        if info <= 0:
            continue

        diff_fit = 1.0 / (1.0 + abs(theta - b))
        weight = skill_weight(skill)

        final_score = (info ** alpha) * (diff_fit ** beta) * weight
        candidates.append((final_score, item, info, diff_fit, weight))

    # 4️⃣ Không có ứng viên phù hợp
    if not candidates:
        if verbose:
            console.print("[yellow]⚠️ Không tìm thấy câu hỏi phù hợp.[/yellow]")
        return None

    # 5️⃣ Sắp xếp và chọn top_k
    candidates.sort(key=lambda x: x[0], reverse=True)
    top_candidates = candidates[:top_k]

    if verbose:
        console.print("\n📊 [bold cyan]Top ứng viên theo điểm ưu tiên:[/bold cyan]")
        for i, (score, item, info, diff, w) in enumerate(top_candidates, 1):
            console.print(
                f"{i}. [green]{item.get('id')}[/green] | Skill: {item.get('skill')} "
                f"| Info={info:.3f} | Fit={diff:.3f} | Weight={w:.2f} | Score={score:.3f}"
            )

    # 6️⃣ Chọn ngẫu nhiên 1 trong top_k
    chosen = random.choice(top_candidates)
    _, selected_item, info, diff, w = chosen

    if verbose:
        console.print("\n🎯 [bold green]Câu hỏi được chọn:[/bold green]")
        console.print(
            f"ID: [yellow]{selected_item.get('id')}[/yellow] "
            f"| Skill: [blue]{selected_item.get('skill')}[/blue]\n"
            f"→ Info={info:.3f}, Fit={diff:.3f}, Weight={w:.2f}\n"
        )

    return selected_item


# ================= DEMO =================
if __name__ == "__main__":
    from .irt_core import update_theta_map
    items = [
        {"id": "1", "skill": "Algebra"},
        {"id": "2", "skill": "Geometry"},
        {"id": "3", "skill": "Functions"},
        {"id": "4", "skill": "Statistics"},
    ]
    irt_params = {
        "1": {"a": 1.2, "b": 0.0, "c": 0.2},
        "2": {"a": 1.0, "b": 1.0, "c": 0.2},
        "3": {"a": 0.8, "b": -0.5, "c": 0.25},
        "4": {"a": 1.1, "b": 2.5, "c": 0.2},
    }
    history = [
        {"skill": "Algebra", "answered_correctly": False},
        {"skill": "Geometry", "answered_correctly": True},
    ]
    q = select_next_item(
        theta=0.2,
        asked_ids=["3"],
        items=items,
        irt_params=irt_params,
        history=history,
        focus_skill=None,
        verbose=True,
    )
    print("\n✅ Câu hỏi được chọn:", q)
