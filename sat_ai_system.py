# sat_ai_system.py
import os
import time
import math
import logging
from typing import List, Dict, Any, Tuple

import sat_ai_core
from explain_ai import explain_answer
from ai_evaluator import evaluate_student_performance

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")


def _ensure_data_loaded() -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, float]]]:
    """Đảm bảo có items và irt_params từ sat_ai_core hoặc fallback."""
    items = getattr(sat_ai_core, "items", None)
    irt_params = getattr(sat_ai_core, "irt_params", None)

    if items is None or irt_params is None:
        import json
        with open("data/items.json", "r", encoding="utf-8") as f:
            items = json.load(f)
        with open("data/irt_params.json", "r", encoding="utf-8") as f:
            params_data = json.load(f)
            irt_params = {str(i["id"]): i for i in params_data}

    return items, irt_params


def run_sat_ai_simulation(
    max_steps: int | None = None,
    theta_convergence_eps: float = 0.01,
    se_threshold: float = 0.25,
    max_duration_minutes: float | None = None,
) -> List[Dict[str, Any]]:

    items, irt_params = _ensure_data_loaded()
    total_available = len(items)

    # ✅ Đặt ở đây: items đã tồn tại
    all_skills = sorted({item.get("skill", "Unknown") for item in items})
    print("\n📚 Các kỹ năng có trong ngân hàng câu hỏi:")
    print(" - " + "\n - ".join(all_skills))

    raw_skill = input("\n👉 Chọn kỹ năng muốn tập trung (Enter = tất cả): ").strip()
    focus_skill = raw_skill if raw_skill in all_skills else None
    print(f"🎯 Tập trung vào kỹ năng: {focus_skill or 'Tất cả'}")


    items, irt_params = _ensure_data_loaded()
    total_available = len(items)

    # ===== (C) Hỏi người dùng chọn số câu =====
    if max_steps is None:
        print("\n🧠 SAT-AI Adaptive System (Gemini + IRT)")
        print(f"📦 Ngân hàng câu hỏi hiện có: {total_available} câu.")
        raw = input(f"👉 Nhập số câu muốn làm (Enter = {total_available}, hoặc 'all' = toàn bộ): ").strip().lower()
        if raw in ("", "all"):
            max_steps = total_available
        else:
            try:
                n = int(raw)
                if n <= 0:
                    max_steps = total_available
                else:
                    max_steps = min(n, total_available)
            except Exception:
                max_steps = total_available

    theta: float = 0.0
    prev_theta: float = float("nan")
    se: float = float("nan")
    asked: List[str] = []
    answered_pairs: List[Tuple[str, int]] = []
    history: List[Dict[str, Any]] = []

    print("\n🚀 BẮT ĐẦU BÀI THI THÍCH ỨNG\n")

    start_time = time.time()
    step = 0

    # ===== Vòng lặp "không giới hạn cứng" (A) — dừng theo điều kiện =====
    while True:
        # Dừng nếu đạt số câu người dùng chọn (C)
        if step >= max_steps:
            print("⛳ Đã đạt số câu mong muốn.")
            break

        # Dừng nếu hết câu chưa hỏi (A)
        if len(asked) >= total_available:
            print("✅ Hết câu hỏi trong ngân hàng!")
            break

        # Dừng nếu quá thời gian (an toàn)
        if max_duration_minutes is not None:
            if (time.time() - start_time) / 60.0 > max_duration_minutes:
                print("⏱️ Hết thời gian làm bài.")
                break

        # Chọn câu tối ưu theo Fisher Info
        item = sat_ai_core.select_next_item(
    theta=theta,
    asked_ids=asked,
    items=items,
    irt_params=irt_params,
    history=history,
    focus_skill=focus_skill,
    top_k=4,
)


        if not item:
            print("✅ Không còn câu phù hợp để hỏi.")
            break

        step += 1
        print(f"\n📘 Câu {step}: {item['question']}")
        for idx, c in enumerate(item["choices"], 1):
            print(f"  {idx}. {c}")

        ans = input("→ Chọn đáp án (1-4 hoặc 'q' để thoát): ").lower().strip()
        if ans == "q":
            print("🛑 Kết thúc sớm theo yêu cầu.")
            break

        if not ans.isdigit() or not (1 <= int(ans) <= len(item["choices"])):
            print("⚠️ Lựa chọn không hợp lệ. Câu này sẽ được bỏ qua.")
            continue

        asked.append(str(item["id"]))
        ans_idx = int(ans) - 1
        correct = int(ans_idx == item["answer_index"])
        print("✅ Chính xác!" if correct else "❌ Sai rồi.")

        answered_pairs.append((str(item["id"]), correct))

        # Cập nhật θ theo MAP (dùng toàn bộ lịch sử)
        prev_theta = theta
        theta, se = sat_ai_core.update_theta_map_once(theta, answered_pairs, irt_params)

        # ===== (B) Dừng khi θ hội tụ
        if math.isfinite(prev_theta) and abs(theta - prev_theta) < theta_convergence_eps:
            print(f"🧲 Hội tụ θ: |Δθ| = {abs(theta - prev_theta):.4f} < {theta_convergence_eps}")
            # vẫn tiếp tục kiểm tra SE bên dưới trước khi dừng — hoặc dừng ngay tuỳ bạn
            # Ở đây: nếu đồng thời SE cũng nhỏ → dừng; nếu không thì cho làm thêm tới max_steps

        # ===== Streaming giải thích (Gemini)
        correct_choice = item["choices"][item["answer_index"]]
        try:
            explanation = explain_answer(item["question"], correct_choice)
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

        # ===== (D) Dừng khi độ tin cậy cao (SE nhỏ)
        if math.isfinite(se) and se < se_threshold:
            print(f"🎯 Độ tin cậy cao: SE = {se:.3f} < {se_threshold}")
            break

    # Kết thúc
    print("\n🏁 KẾT THÚC BÀI THI")
    if math.isfinite(se):
        print(f"🎯 Năng lực cuối cùng θ = {theta:.2f} ± {se:.2f}")
    else:
        print(f"🎯 Năng lực cuối cùng θ = {theta:.2f}")

    # Báo cáo tổng kết bằng Gemini
    if history:
        final_theta = history[-1]["theta"]
        print("\n📊 Đang tạo báo cáo đánh giá năng lực với AI...\n")
        try:
            report = evaluate_student_performance(history, final_theta)
        except Exception as e:
            report = f"⚠️ Lỗi khi đánh giá năng lực: {e}"

        print("\n📘 BÁO CÁO NĂNG LỰC SAT:\n")
        print(report or "⚠️ Không thể tạo báo cáo.")

        try:
            os.makedirs("results", exist_ok=True)
            with open("results/evaluation_report.txt", "w", encoding="utf-8") as f:
                f.write(report or "")
            print("\n✅ Báo cáo đã lưu tại: results/evaluation_report.txt")
        except Exception as e:
            print(f"⚠️ Không thể lưu báo cáo: {e}")
    else:
        print("⚠️ Không có dữ liệu để đánh giá.")

    return history


# ===== ENTRY =====
if __name__ == "__main__":
    if not os.getenv("GOOGLE_API_KEY"):
        print("⚠️ Bạn cần set GOOGLE_API_KEY trước khi chạy.")
        print("PowerShell:   $Env:GOOGLE_API_KEY=\"YOUR_KEY\"")
        print("CMD:          set GOOGLE_API_KEY=YOUR_KEY")
        raise SystemExit(1)

    # Gọi không truyền max_steps để bật prompt lựa chọn (C)
    run_sat_ai_simulation(
        max_steps=None,             # → hỏi người dùng
        theta_convergence_eps=0.01, # (B)
        se_threshold=0.25,          # (D)
        max_duration_minutes=None   # tuỳ chọn
    )
