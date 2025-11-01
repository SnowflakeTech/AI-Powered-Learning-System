import os
import time
import logging
from typing import List, Dict, Any, Optional
from google import genai

# ✅ Logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

# ✅ Gemini Client
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("❌ GOOGLE_API_KEY chưa được set!")
client = genai.Client(api_key=api_key)

MODEL = "gemini-2.5-flash"

# -----------------------
# 🔹 Hàm phụ trợ
# -----------------------
def shorten_text(text: str, max_len: int = 160) -> str:
    if not isinstance(text, str): return ""
    t = " ".join(text.split())
    return t if len(t) <= max_len else t[:max_len].rsplit(" ", 1)[0] + "…"


def history_to_summary(history: List[Dict[str, Any]]) -> str:
    lines = []
    for h in history:
        result = "✅ đúng" if h.get("answered_correctly") else "❌ sai"
        skill = h.get("skill", "Unknown")
        question = shorten_text(h.get("question", ""))
        lines.append(f"- [{result}] *{skill}*: {question}")
    return "\n".join(lines)


# -----------------------
# ✅ Retry cho Gemini
# -----------------------
def call_gemini_with_retry(prompt: str, *, temperature: float, max_tokens: int, retries: int = 3) -> Optional[str]:

    for attempt in range(1, retries + 1):
        try:
            start = time.time()
            resp = client.models.generate_content(
                model=MODEL,
                contents=prompt
            )
            latency = time.time() - start

            text = resp.text.strip()
            token_est = len(text.split())

            logging.info(f"✅ Gemini success (lat={latency:.2f}s, tokens≈{token_est})")
            return text

        except Exception as e:
            wait = 2 ** attempt
            logging.warning(f"⚠️ Retry {attempt}/{retries} after {wait}s: {e}")
            time.sleep(wait)

    logging.error("🚨 API FAILED")
    return None


# -----------------------
# 🧠 Tạo báo cáo học lực học sinh
# -----------------------
def evaluate_student_performance(
    history: List[Dict[str, Any]],
    final_theta: float,
    *,
    language: str = "vi",
    temperature: float = 0.4,
    max_tokens: int = 800,
) -> str:

    if not history:
        return "⚠️ Không có dữ liệu bài thi."

    try:
        theta = round(float(final_theta), 2)
    except Exception:
        return "🚨 final_theta không hợp lệ!"

    summary_text = history_to_summary(history)

    system_prompt_vi = (
        "Bạn là chuyên gia giáo dục SAT. Hãy tạo báo cáo bằng Markdown rõ ràng, gồm 4 phần:\n"
        "1️⃣ Tổng quan năng lực\n"
        "2️⃣ Kỹ năng mạnh/yếu\n"
        "3️⃣ Gợi ý luyện tập 3–5 mục tiêu\n"
        "4️⃣ Dự đoán mức SAT tương ứng (Beginner / Intermediate / Advanced)\n\n"
        "Viết ngắn gọn, có bullet và tiêu đề phụ."
    )

    system_prompt_en = (
        "You are an SAT education expert. Write a Markdown report with 4 sections:\n"
        "1 Overview\n"
        "2 Strengths & Weaknesses\n"
        "3 Study Suggestions (3–5 bullets)\n"
        "4 Predicted SAT Level (B/I/A)\n"
        "Use clear bullets and sub-headings."
    )

    sys = system_prompt_vi if language == "vi" else system_prompt_en

    full_prompt = f"""
{sys}

📊 **Thông tin bài thi**
- Năng lực cuối cùng (θ): {theta}
- Số câu hỏi: {len(history)}
- Chi tiết từng câu:
{summary_text}
""".strip()

    report = call_gemini_with_retry(
        full_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return report or "🚨 Không thể tạo báo cáo sau retry. Thử lại sau."
