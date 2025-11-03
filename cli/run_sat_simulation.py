"""
cli/run_sat_simulation.py
-----------------------------------
Chạy bài thi thích ứng SAT AI (Adaptive Test) trên CLI.
Kết hợp các module:
- sat_ai_core.question_selector
- sat_ai_core.irt_core
- sat_ai_core.ai_explainer
- sat_ai_core.ai_evaluator
"""

import os
import sys
import math
import time
import logging
from typing import List, Dict, Any, Tuple
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dotenv import load_dotenv

# ===== Load .env từ thư mục gốc =====
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
load_dotenv(dotenv_path=env_path)

# ===== Logging =====
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

# ===== Import nội bộ =====
if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from sat_ai_core import irt_core, question_selector, ai_explainer, ai_evaluator
else:
    from sat_ai_core import irt_core, question_selector, ai_explainer, ai_evaluator


# ==============================
# 🧩 Tải dữ liệu items & params
# ==============================
def load_data() -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, float]]]:
    """Đọc items.json và irt_params.json."""
    import json
    try:
        with open("data/items.json", "r", encoding="utf-8") as f:
            items = json.load(f)
        with open("data/irt_params.json", "r", encoding="utf-8") as f:
            params_data = json.load(f)
            irt_params = {str(i["id"]): i for i in params_data}
        return items, irt_params
    except Exception as e:
        logging.error(f"🚨 Lỗi khi đọc dữ liệu: {e}")
        return [], {}


# ==============================
# 🧠 Giao diện chọn kỹ năng
# ==============================
def choose_skill(items: List[Dict[str, Any]]) -> str | None:
    """Hiển thị danh sách kỹ năng và cho phép chọn bằng số hoặc tên."""
    all_skills = sorted({item.get("skill", "Unknown") for item in items})
    print("\n📚 Các kỹ năng có trong ngân hàng câu hỏi:\n")
    for i, skill in enumerate(all_skills, 1):
        print(f"  {i}. {skill}")

    raw_input_skill = input("\n👉 Chọn kỹ năng muốn tập trung (Enter = tất cả): ").strip()

    if raw_input_skill == "":
        focus_skill = None
    elif raw_input_skill.isdigit() and 1 <= int(raw_input_skill) <= len(all_skills):
        focus_skill = all_skills[int(raw_input_skill) - 1]
    elif raw_input_skill in all_skills:
        focus_skill = raw_input_skill
    else:
        print("⚠️ Lựa chọn không hợp lệ, mặc định: Tất cả.")
        focus_skill = None

    print(f"\n🎯 Tập trung vào kỹ năng: {focus_skill or 'Tất cả'}")
    return focus_skill


# ==============================
# 🚀 Chạy bài thi thích ứng
# ==============================
def run_sat_demo(
    max_steps: int | None = None,
    theta_convergence_eps: float = 0.01,
    se_threshold: float = 0.25,
    max_duration_minutes: float | None = None,
):
    """Chạy bài thi thích ứng trên CLI."""
    items, irt_params = load_data()
    if not items:
        print("⚠️ Không tìm thấy dữ liệu câu hỏi.")
        return

    total_available = len(items)
    print(f"\n🧠 Ngân hàng câu hỏi: {total_available} câu.")

    # --- Chọn kỹ năng ---
    focus_skill = choose_skill(items)

    # --- Chọn số câu ---
    if max_steps is None:
        try:
            raw = input(f"\n🧩 Nhập số câu muốn làm (Enter = 5, max {total_available}): ").strip()
            if raw in ("", "all"):
                max_steps = min(5, total_available)
            else:
                n = int(raw)
                max_steps = max(1, min(n, total_available))
        except Exception:
            max_steps = 5

    # --- Biến trạng thái ---
    theta = 0.0
    prev_theta = float("nan")
    se = float("nan")
    asked: List[str] = []
    answered_pairs: List[Tuple[str, int]] = []
    history: List[Dict[str, Any]] = []

    print("\n🚀 BẮT ĐẦU BÀI THI THÍCH ỨNG\n")
    start_time = time.time()
    step = 0

    # --- Vòng lặp chính ---
    while True:
        if step >= max_steps:
            print("⛳ Đã đạt số câu mong muốn.")
            break

        if len(asked) >= total_available:
            print("✅ Hết câu hỏi trong ngân hàng!")
            break

        if max_duration_minutes and (time.time() - start_time) / 60 > max_duration_minutes:
            print("⏱️ Hết thời gian làm bài.")
            break

        # Chọn câu tiếp theo
        item = question_selector.select_next_item(
            theta=theta,
            asked_ids=asked,
            items=items,
            irt_params=irt_params,
            history=history,
            focus_skill=focus_skill,
            top_k=4,
        )

        if not item:
            print("✅ Không còn câu hỏi phù hợp.")
            break

        step += 1
        print(f"\n📘 Câu {step}: {item['question']}")
        for idx, c in enumerate(item["choices"], 1):
            print(f"  {idx}. {c}")

        ans = input("→ Chọn đáp án (1–4 hoặc 'q' để thoát): ").strip().lower()
        if ans == "q":
            print("🛑 Dừng bài thi theo yêu cầu.")
            break

        if not ans.isdigit() or not (1 <= int(ans) <= len(item["choices"])):
            print("⚠️ Lựa chọn không hợp lệ. Bỏ qua câu này.")
            continue

        asked.append(str(item["id"]))
        ans_idx = int(ans) - 1
        correct = int(ans_idx == item["answer_index"])
        print("✅ Chính xác!" if correct else "❌ Sai rồi.")

        answered_pairs.append((str(item["id"]), correct))

        # Cập nhật θ
        prev_theta = theta
        theta, se = irt_core.update_theta_map_once(theta, answered_pairs, irt_params)

        # Giải thích bằng AI
        correct_choice = item["choices"][item["answer_index"]]
        try:
            explanation = ai_explainer.explain_answer(item["question"], correct_choice)
        except Exception as e:
            explanation = f"⚠️ Lỗi AI: {e}"

        print("\n💡 GIẢI THÍCH CỦA AI:\n")
        print(explanation or "⚠️ Không có phản hồi từ AI.")

        # Lưu lịch sử
        history.append({
            "id": item["id"],
            "question": item["question"],
            "answered_correctly": bool(correct),
            "theta": theta,
            "skill": item.get("skill", "Unknown"),
        })

        # Hiển thị θ ± SE
        if math.isfinite(se):
            print(f"\n📈 θ hiện tại: {theta:.2f} ± {se:.2f}")
        else:
            print(f"\n📈 θ hiện tại: {theta:.2f}")

        # Kiểm tra dừng theo SE
        if math.isfinite(se) and se < se_threshold:
            print(f"🎯 Độ tin cậy cao: SE = {se:.3f} < {se_threshold}")
            break

    # --- Kết thúc ---
    print("\n🏁 KẾT THÚC BÀI THI")
    if math.isfinite(se):
        print(f"🎯 Năng lực cuối cùng θ = {theta:.2f} ± {se:.2f}")
    else:
        print(f"🎯 Năng lực cuối cùng θ = {theta:.2f}")

    # Báo cáo tổng kết
    if history:
        final_theta = history[-1]["theta"]
        print("\n📊 Đang tạo báo cáo đánh giá năng lực...\n")
        try:
            report = ai_evaluator.evaluate_student_performance(history, final_theta)
        except Exception as e:
            report = f"⚠️ Lỗi khi tạo báo cáo: {e}"

        print("\n📘 BÁO CÁO NĂNG LỰC:\n")
        print(report or "⚠️ Không thể tạo báo cáo.")

        # Lưu kết quả
        os.makedirs("results", exist_ok=True)
        try:
            with open("results/evaluation_report.txt", "w", encoding="utf-8") as f:
                f.write(report or "")
            print("\n✅ Báo cáo đã lưu tại: results/evaluation_report.txt")
        except Exception as e:
            print(f"⚠️ Không thể lưu báo cáo: {e}")
    else:
        print("⚠️ Không có dữ liệu để đánh giá.")


# ==============================
# ENTRYPOINT
# ==============================
if __name__ == "__main__":
    run_sat_demo()
