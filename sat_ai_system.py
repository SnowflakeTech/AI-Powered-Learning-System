import os
import sat_ai_core
from explain_ai import explain_answer
from ai_evaluator import evaluate_student_performance

def run_sat_ai_simulation(max_steps=5):
    """Mô phỏng hệ thống SAT-AI hoàn chỉnh (IRT + GPT giải thích + đánh giá)."""
    theta = 0.0
    asked = []
    history = []

    print("=== 🧠 BẮT ĐẦU MÔ PHỎNG SAT-AI ===\n")

    for step in range(max_steps):
        # 1️⃣ Chọn câu hỏi kế tiếp dựa trên năng lực hiện tại
        item = sat_ai_core.select_next_item(theta, asked)
        if not item:
            print("✅ Hết câu hỏi trong ngân hàng!")
            break

        asked.append(item["id"])
        print(f"\n📘 Câu {step+1}: {item['question']}")
        for i, c in enumerate(item["choices"]):
            print(f"{i+1}. {c}")

        # 2️⃣ Nhập câu trả lời
        ans = input("Nhập đáp án (1–4 hoặc 'q' để thoát): ").strip()
        if ans.lower() == "q":
            print("🛑 Kết thúc sớm.")
            break

        if not ans.isdigit() or not (1 <= int(ans) <= len(item["choices"])):
            print("⚠️ Lựa chọn không hợp lệ, bỏ qua câu này.")
            continue

        ans_index = int(ans) - 1
        correct = ans_index == item["answer_index"]
        print("✅ Chính xác!" if correct else "❌ Sai rồi!")

        # 3️⃣ Cập nhật năng lực θ theo IRT
        params = sat_ai_core.irt_params[str(item["id"])]
        theta = sat_ai_core.update_theta(theta, int(correct),
                                         params["a"], params["b"], params["c"])

        # 4️⃣ AI giải thích cách làm (sử dụng explain_ai.py)
        correct_choice = item["choices"][item["answer_index"]]
        explanation = explain_answer(item["question"], correct_choice)
        print("\n💡 Giải thích của AI:")
        print(explanation)

        # 5️⃣ Lưu lại lịch sử
        history.append({
            "id": item["id"],
            "question": item["question"],
            "answered_correctly": correct,
            "theta": theta,
            "skill": item.get("skill", "Unknown")
        })

        print(f"\n🎯 Năng lực hiện tại (θ): {theta:.2f}")

    print("\n=== KẾT THÚC MÔ PHỎNG ===")
    print(f"🎯 Năng lực cuối cùng (θ_final): {theta:.2f}")

    # 🔥 Gọi AI để đánh giá năng lực tổng hợp sau bài thi
    if history:
        final_theta = history[-1]["theta"]
        print("\n📊 Đang tạo báo cáo đánh giá năng lực với AI...\n")
        report = evaluate_student_performance(history, final_theta)

        print("\n📘 BÁO CÁO ĐÁNH GIÁ NĂNG LỰC:\n")
        print(report)

        # Lưu ra file
        os.makedirs("results", exist_ok=True)
        with open("results/evaluation_report.txt", "w", encoding="utf-8") as f:
            f.write(report)
        print("\n✅ Báo cáo đã được lưu trong results/evaluation_report.txt")
    else:
        print("⚠️ Không có dữ liệu lịch sử để đánh giá.")

    return history


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ Thiếu biến môi trường OPENAI_API_KEY. Hãy đặt trước khi chạy.")
        print("   Ví dụ: export OPENAI_API_KEY='sk-proj-...'\n")
        exit(1)

    run_sat_ai_simulation(max_steps=5)
