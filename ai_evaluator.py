import os
import json
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def evaluate_student_performance(history, final_theta, model="gpt-4o-mini"):
    """
    Dùng AI để sinh bản đánh giá năng lực & kỹ năng làm bài dựa trên lịch sử thi.
    history: list[dict] gồm {id, question, answered_correctly, theta, skill}
    """
    # Chuẩn bị dữ liệu tóm tắt gửi lên AI
    summary_lines = []
    for h in history:
        result = "✅ đúng" if h["answered_correctly"] else "❌ sai"
        summary_lines.append(f"- [{result}] {h['skill']} – {h['question']}")
    summary_text = "\n".join(summary_lines)

    prompt = f"""
    Bạn là chuyên gia giáo dục SAT. Dựa trên kết quả mô phỏng sau, hãy viết báo cáo đánh giá năng lực học viên.

    **Thông tin bài thi:**
    - Năng lực cuối cùng θ = {final_theta:.2f}
    - Số câu hỏi: {len(history)}
    - Chi tiết từng câu:
    {summary_text}

    Hãy xuất bản đánh giá gồm các phần:
    1. Tổng quan năng lực (θ, độ ổn định, so với trung bình)
    2. Kỹ năng mạnh và yếu (theo skill)
    3. Gợi ý luyện tập / cải thiện
    4. Dự đoán mức SAT tương ứng (ví dụ: Beginner / Intermediate / Advanced)
    """

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"🚨 Lỗi khi gọi OpenAI API: {e}"
