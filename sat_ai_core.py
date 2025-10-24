import json, math, random

with open("data/items.json", "r", encoding="utf-8") as f:
    items = json.load(f)

with open("data/irt_params.json", "r", encoding="utf-8") as f:
    irt_params_data = json.load(f)
    # Ensure id is treated as string in irt_params
    irt_params = {str(i["id"]): i for i in irt_params_data}

def prob_correct(theta, a, b, c):
    """Xác suất trả lời đúng theo mô hình 3PL."""
    return c + (1 - c) / (1 + math.exp(-1.7 * a * (theta - b)))

def update_theta(theta, response, a, b, c, lr=0.4):
    """Cập nhật năng lực người học dựa trên kết quả."""
    p = prob_correct(theta, a, b, c)
    grad = a * (response - p)
    return theta + lr * grad

def select_next_item(theta, asked):
    """Chọn câu hỏi có độ thông tin lớn nhất."""
    candidates = [i for i in items if i["id"] not in asked]
    if not candidates:
        return None

    best_item = max(
        candidates,
        key=lambda i: (
            irt_params[str(i["id"])] ["a"] ** 2 *
            prob_correct(theta,
                         irt_params[str(i["id"])] ["a"],
                         irt_params[str(i["id"])] ["b"],
                         irt_params[str(i["id"])] ["c"]) *
            (1 - prob_correct(theta,
                              irt_params[str(i["id"])] ["a"],
                              irt_params[str(i["id"])] ["b"],
                              irt_params[str(i["id"])] ["c"]))
        )
    )
    return best_item

if __name__ == "__main__":
    theta = 0.0
    asked = []
    print("=== SAT Adaptive Demo ===\n")

    for step in range(10):  # tối đa 10 câu
        item = select_next_item(theta, asked)
        if not item:
            print("✅ Hết câu hỏi trong ngân hàng! Dừng bài thi.")
            break

        asked.append(item["id"])
        print(f"Câu {step+1}: {item['question']}")
        for i, c in enumerate(item["choices"]):
            print(f"{i+1}. {c}")
        ans = input("Đáp án của bạn (1-4, hoặc 'q' để thoát): ")

        if ans.lower() == "q":
            print("🛑 Kết thúc sớm.")
            break

        ans = int(ans) - 1
        correct = ans == item["answer_index"]
        print("✅ Đúng!" if correct else "❌ Sai.")

        params = irt_params[str(item["id"])]
        theta = update_theta(theta, int(correct),
                             params["a"], params["b"], params["c"])
        print(f"Năng lực hiện tại θ = {theta:.2f}\n")

    print(f"🎯 Năng lực cuối cùng (θ_final): {theta:.2f}")