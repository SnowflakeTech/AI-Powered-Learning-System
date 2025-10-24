import os
from openai import OpenAI
import json

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ Thiếu OPENAI_API_KEY trong môi trường. Hãy đặt biến này trước khi chạy.")

client = OpenAI(api_key=api_key)

def explain_answer(question, correct_choice, model="gpt-4o-mini", temperature=0.3):
    """Giải thích ngắn gọn cách giải câu hỏi SAT. Có xử lý lỗi và log an toàn."""

    prompt = f"""
    Giải thích ngắn gọn cách giải câu hỏi SAT sau:
    Câu hỏi: {question}
    Đáp án đúng: {correct_choice}
    """

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        answer = response.choices[0].message.content.strip()
        print("\n🧩 [DEBUG] Prompt gửi đến model:\n", prompt.strip())
        print("\n✅ [DEBUG] Phản hồi model:\n", answer)
        return answer

    except Exception as e:
        print(f"\n🚨 Lỗi khi gọi OpenAI API: {e}")
        return None


if __name__ == "__main__":
    # Ví dụ test nhanh
    q = "Nếu 2x + 3 = 7 thì x = ?"
    a = "2"
    explanation = explain_answer(q, a)
    if explanation:
        print("\n📘 Giải thích:\n", explanation)
    else:
        print("\n⚠️ Không thể lấy phản hồi từ API.")
