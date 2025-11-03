"""
sat_ai_core/ai_evaluator.py
-----------------------------------
Module tổng hợp & đánh giá năng lực học sinh SAT bằng OpenAI.
Sinh báo cáo Markdown gồm: Tổng quan – Điểm mạnh/yếu – Gợi ý luyện tập – Dự đoán Level.
"""

import os
import time
import logging
import hashlib
import sqlite3
from typing import List, Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

# ============ CẤU HÌNH CƠ BẢN ============
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
load_dotenv(dotenv_path=env_path)
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY chưa được set trong .env!")

model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
client = OpenAI(api_key=api_key)

# ============ DATABASE CACHE ============
DB_PATH = "ai_cache.db"

def _get_cache(key: str) -> Optional[str]:
    if not os.path.exists(DB_PATH):
        return None
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("SELECT response FROM cache WHERE key=?", (key,)).fetchone()
    conn.close()
    return row[0] if row else None

def _set_cache(key: str, text: str):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT OR REPLACE INTO cache VALUES (?, ?)", (key, text))
    conn.commit()
    conn.close()

# ============ HÀM TIỆN ÍCH ============

def _shorten_text(text: str, max_len: int = 120) -> str:
    if not isinstance(text, str): return ""
    t = " ".join(text.split())
    return t if len(t) <= max_len else t[:max_len].rsplit(" ", 1)[0] + "…"

def _history_summary(history: List[Dict[str, Any]]) -> str:
    """Rút gọn lịch sử câu hỏi cho prompt AI."""
    lines = []
    for h in history:
        res = "✅ đúng" if h.get("answered_correctly") else "❌ sai"
        skill = h.get("skill", "Unknown")
        q = _shorten_text(h.get("question", ""))
        lines.append(f"- [{res}] *{skill}*: {q}")
    return "\n".join(lines)


# ============ GỌI OPENAI ============

def _call_openai_with_retry(prompt: str, temperature: float = 0.5, retries: int = 3) -> Optional[str]:
    """Gọi OpenAI có retry, xử lý lỗi mạng nhẹ."""
    for attempt in range(1, retries + 1):
        try:
            start = time.time()
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Bạn là chuyên gia giáo dục SAT. Hãy viết báo cáo đánh giá ngắn gọn, rõ ràng bằng Markdown."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
            )
            latency = time.time() - start
            text = resp.choices[0].message.content.strip()
            logging.info(f"✅ OpenAI success (lat={latency:.2f}s, len={len(text.split())})")
            return text

        except Exception as e:
            wait = 2 ** attempt
            logging.warning(f"⚠️ Retry {attempt}/{retries} sau {wait}s do lỗi: {e}")
            time.sleep(wait)
    return None


# ============ HÀM CHÍNH ============

def evaluate_student_performance(
    history: List[Dict[str, Any]],
    final_theta: float,
    *,
    language: str = "vi",
    temperature: float = 0.5,
) -> str:
    """
    Sinh báo cáo năng lực học sinh dựa trên lịch sử và θ cuối.

    Parameters
    ----------
    history : list[dict]
        Danh sách các câu hỏi đã làm cùng kết quả.
    final_theta : float
        Năng lực cuối (θ).
    language : str
        "vi" hoặc "en" để chọn ngôn ngữ.
    """

    if not history:
        return "⚠️ Không có dữ liệu bài thi để đánh giá."

    try:
        theta = round(float(final_theta), 2)
    except Exception:
        return "🚨 Giá trị θ không hợp lệ!"

    summary = _history_summary(history)

    sys_vi = (
        "Bạn là chuyên gia giáo dục SAT. Viết báo cáo Markdown gồm 4 phần:\n"
        "1 **Tổng quan năng lực:** mô tả trình độ và độ ổn định dựa vào θ.\n"
        "2 **Kỹ năng mạnh / yếu:** phân tích các kỹ năng học sinh làm tốt và chưa tốt.\n"
        "3 **Gợi ý luyện tập:** đề xuất 3–5 hướng cải thiện cụ thể.\n"
        "4 **Dự đoán cấp độ SAT:** Beginner / Intermediate / Advanced.\n"
        "Viết ngắn gọn, rõ ràng, dùng bullet points."
    )

    sys_en = (
        "You are an SAT education expert. Write a short Markdown report with 4 sections:\n"
        "1 Overview of ability (based on theta)\n"
        "2 Strengths & Weaknesses\n"
        "3 Study Recommendations (3–5 concise bullet points)\n"
        "4 Predicted SAT Level (Beginner / Intermediate / Advanced)"
    )

    system_prompt = sys_vi if language == "vi" else sys_en

    prompt = f"""
{system_prompt}

📊 **Thông tin bài thi**
- Năng lực cuối (θ): {theta}
- Số câu hỏi: {len(history)}

📄 **Chi tiết từng câu:**
{summary}
""".strip()

    # Cache key
    key = hashlib.sha256(prompt.encode()).hexdigest()
    cached = _get_cache(key)
    if cached:
        print("⚡ Đã có cache báo cáo AI!\n")
        print(cached)
        return cached

    print("\n🤖 Đang tạo báo cáo năng lực bằng OpenAI...\n")
    report = _call_openai_with_retry(prompt, temperature=temperature)

    if not report:
        return "🚨 Không thể tạo báo cáo năng lực. Vui lòng thử lại."

    _set_cache(key, report)

    print("✅ Báo cáo hoàn tất!\n")
    return report


# ============ DEMO ============
if __name__ == "__main__":
    demo_history = [
        {"question": "Nếu 3x + 5 = 20, tìm x?", "skill": "Algebra", "answered_correctly": True},
        {"question": "Tính diện tích hình tròn bán kính 4.", "skill": "Geometry", "answered_correctly": False},
        {"question": "Một đường thẳng có hệ số góc bằng 2, đi qua (1,3)...", "skill": "Functions", "answered_correctly": True},
    ]
    report = evaluate_student_performance(demo_history, 0.85)
    print("\n📘 BÁO CÁO MẪU:\n")
    print(report)
