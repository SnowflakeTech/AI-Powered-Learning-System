import json
from math import *
import logging
import random
import time
from typing import Any, Dict, Optional
from typing import List, Dict, Tuple, Optional

from explain_ai import explain_answer
from ai_evaluator import evaluate_student_performance

# =========================
# CẤU HÌNH
# =========================
D = 1.7
THETA_BOUNDS = (-4.0, 4.0)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")


# =========================
# IRT CORE (giữ nguyên)
# =========================
def sigmoid_stable(x: float) -> float:
    if x >= 0:
        z = exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = exp(x)
        return z / (1.0 + z)


def prob_correct(theta: float, a: float, b: float, c: float) -> float:
    s = sigmoid_stable(D * a * (theta - b))
    return c + (1.0 - c) * s


def dprob_dtheta(theta: float, a: float, b: float, c: float) -> float:
    s = sigmoid_stable(D * a * (theta - b))
    return (1.0 - c) * D * a * s * (1.0 - s)


def fisher_info(theta: float, a: float, b: float, c: float) -> float:
    if a <= 0 or not (0 <= c < 1):
        return 0.0
    p = prob_correct(theta, a, b, c)
    if not (1e-6 < p < 1 - 1e-6):
        return 0.0
    dp = dprob_dtheta(theta, a, b, c)
    return (dp * dp) / (p * (1.0 - p))


# =========================
# UPDATE θ (giữ nguyên)
# =========================
def update_theta_map_once(theta: float, answered_items: List[Tuple[str, int]], irt_params: Dict[str, Dict]) -> Tuple[float, float]:
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
        U += (resp - p) * dp / (p * (1.0 - p))
        I += (dp * dp) / (p * (1.0 - p))

    den = I + 1.0
    theta_new = theta + U / den
    theta_new = max(min(theta_new, THETA_BOUNDS[1]), THETA_BOUNDS[0])
    se = 1.0 / sqrt(den)

    return theta_new, se


# =========================
# CHỌN CÂU TỐI ƯU
# =========================
def select_next_item(
    theta: float,
    asked_ids: List[str],
    items,
    irt_params,
    *,
    history: List[Dict[str, Any]] = None,
    focus_skill: Optional[str] = None,
    top_k: int = 4,
) -> Optional[Dict]:

    # ===== 1️⃣ Skill Weak-based weighting =====
    skill_wrong = {}
    if history:
        for it in history:
            s = it.get("skill", "Unknown")
            skill_wrong.setdefault(s, 0)
            if not it.get("answered_correctly"):
                skill_wrong[s] += 1

    def skill_weight(skill: str) -> float:
        if focus_skill and skill != focus_skill:
            return 0.5  # giảm ưu tiên nếu khác chủ đề
        return 1.0 + skill_wrong.get(skill, 0) * 0.5

    candidates = []

    for item in items:
        item_id = str(item["id"])
        if item_id in asked_ids: continue

        pars = irt_params.get(item_id)
        if not pars: continue

        a, b, c = pars["a"], pars["b"], pars["c"]

        # Fisher Information
        info = fisher_info(theta, a, b, c)
        if info <= 0: continue
        
        # ===== 2️⃣ Difficulty Fit (θ gần b) =====
        diff_score = 1.0 / (1.0 + abs(theta - b))

        # ===== 3️⃣ Weak Skill Boost =====
        skill = item.get("skill", "Unknown")
        w = skill_weight(skill)

        final_score = info * diff_score * w
        candidates.append((final_score, item))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    top_candidates = [itm for _, itm in candidates[:top_k]]

    return random.choice(top_candidates)


# =========================
# ADAPTIVE TEST DEMO
# =========================
def run_adaptive_demo(max_questions=5):
    try:
        with open("data/items.json", encoding="utf-8") as f: items = json.load(f)
        with open("data/irt_params.json", encoding="utf-8") as f:
            params = {str(i["id"]): i for i in json.load(f)}
    except Exception as e:
        logging.error(f"Lỗi đọc file: {e}")
        return

    theta = 0.0
    asked_ids, answered = [], []
    history = []   # ✅ lưu để gửi qua ai_evaluator

    print("\n🎯 SAT Adaptive Test (Gemini + IRT)\n")

    for i in range(max_questions):
        item = select_next_item(theta, asked_ids, items, params)
        if not item:
            print("✅ Hết câu hỏi!")
            break

        print(f"\n📌 Câu {i+1}: {item['question']}")
        for idx, c in enumerate(item["choices"], 1):
            print(f" → {idx}. {c}")

        ans = input("\nChọn (1-4) hoặc q: ").lower()
        if ans == "q":
            break

        try:
            k = int(ans) - 1
        except:
            print("⚠️ Nhập sai!")
            continue

        asked_ids.append(str(item["id"]))
        correct = int(k == item["answer_index"])

        print("✅ Đúng!" if correct else "❌ Sai.")

        answered.append((item["id"], correct))
        theta, _ = update_theta_map_once(theta, answered, params)

        # ✅ Lưu lịch sử
        history.append({
            "question": item["question"],
            "skill": item.get("skill", "Unknown"),
            "answered_correctly": bool(correct),
        })

        # ✅ STREAMING GIẢI THÍCH NGAY SAU CÂU TRẢ LỜI
        correct_choice = item["choices"][item["answer_index"]]
        explain_answer(item["question"], correct_choice)

        print(f"\n📈 θ hiện tại: {theta:.2f}")

    print(f"\n🎯 θ cuối: {theta:.2f}")

    # ✅ Gọi báo cáo tổng kết
    report = evaluate_student_performance(history, theta)
    print("\n📊 BÁO CÁO TỔNG KẾT:\n")
    print(report)


# =========================
# ENTRY
# =========================
if __name__ == "__main__":
    run_adaptive_demo()
