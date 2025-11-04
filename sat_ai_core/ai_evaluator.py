import os
import time
import logging
import hashlib
import sqlite3
from datetime import datetime
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown
from sat_ai_core.api_throttler import ApiThrottler, ThrottlerError

PROMPT_VERSION = "v2"

env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
load_dotenv(dotenv_path=env_path)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY chưa được set trong .env!")

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
client = OpenAI(api_key=api_key)
throttler = ApiThrottler(min_interval=2.0, max_retries=5, max_wait=25.0, per_model=True)

DB_PATH = "ai_cache.db"
os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)

def _init_db():
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                model TEXT,
                created_at TEXT,
                tokens INTEGER,
                response TEXT NOT NULL
            );
        """)
    except sqlite3.OperationalError:
        for col, definition in [
            ("model", "TEXT DEFAULT 'unknown'"),
            ("created_at", "TEXT DEFAULT CURRENT_TIMESTAMP"),
            ("tokens", "INTEGER DEFAULT 0"),
        ]:
            try:
                conn.execute(f"ALTER TABLE cache ADD COLUMN {col} {definition};")
            except sqlite3.OperationalError:
                pass
    conn.commit()
    conn.close()

def _get_cache(key: str, model: str) -> Optional[str]:
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("SELECT response FROM cache WHERE key=? AND model=?", (key, model)).fetchone()
    conn.close()
    return row[0] if row else None

def _set_cache(key: str, model: str, text: str, tokens: int):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT OR REPLACE INTO cache VALUES (?, ?, ?, ?, ?)",
        (key, model, datetime.now().isoformat(), tokens, text),
    )
    conn.commit()
    conn.close()

_init_db()

def _shorten_text(text: str, max_len: int = 120) -> str:
    if not isinstance(text, str):
        return ""
    t = " ".join(text.split())
    return t if len(t) <= max_len else t[:max_len].rsplit(" ", 1)[0] + "…"

def _history_summary(history: List[Dict[str, Any]]) -> str:
    lines = []
    for h in history:
        res = "✅ đúng" if h.get("answered_correctly") else "❌ sai"
        skill = h.get("skill", "Unknown")
        q = _shorten_text(h.get("question", ""))
        lines.append(f"- [{res}] *{skill}*: {q}")
    return "\n".join(lines)

def evaluate_student_performance(
    history: List[Dict[str, Any]],
    final_theta: float,
    *,
    language: str = "vi",
    temperature: float = 0.5,
    verbose: bool = True,
) -> str:
    if not history:
        return "⚠️ Không có dữ liệu bài thi để đánh giá."
    try:
        theta = round(float(final_theta), 2)
    except Exception:
        return "🚨 Giá trị θ không hợp lệ!"

    summary = _history_summary(history)

    sys_vi = (
        "Bạn là chuyên gia giáo dục SAT. Viết báo cáo Markdown với 4 phần:\n"
        "① **Tổng quan năng lực:** mô tả trình độ và độ ổn định dựa trên θ.\n"
        "② **Kỹ năng mạnh / yếu:** liệt kê các kỹ năng tốt và yếu.\n"
        "③ **Gợi ý luyện tập:** đề xuất 3–5 hướng cải thiện cụ thể.\n"
        "④ **Dự đoán cấp độ SAT:** Beginner / Intermediate / Advanced.\n"
        "Viết ngắn gọn, rõ ràng, có định dạng Markdown."
    )

    sys_en = (
        "You are an SAT education expert. Write a Markdown report with 4 sections:\n"
        "① Overview of ability (based on theta)\n"
        "② Strengths & Weaknesses\n"
        "③ Study Recommendations (3–5 concise bullet points)\n"
        "④ Predicted SAT Level (Beginner / Intermediate / Advanced)"
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

    key_src = f"{PROMPT_VERSION}::{MODEL}::{prompt}"
    key = hashlib.sha256(key_src.encode()).hexdigest()
    cached = _get_cache(key, MODEL)

    console = Console()
    if cached:
        if verbose:
            console.print("⚡ [bold yellow]Đã có cache báo cáo AI![/bold yellow]\n")
            console.print(Markdown(cached))
        return cached

    console.print("\n🤖 [cyan]Đang tạo báo cáo năng lực bằng OpenAI...[/cyan]\n")

    try:
        response = throttler.safe_openai_chat(
            client,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            model=MODEL,
            temperature=temperature,
        )

        report = response.choices[0].message.content.strip()
        token_count = len(report.split())
        _set_cache(key, MODEL, report, token_count)

        console.print("\n✅ [green]Báo cáo hoàn tất![/green]")
        logging.info(f"📊 Tokens ~ {token_count}\n")
        console.print("\n📘 [bold]BÁO CÁO:[/bold]\n")
        console.print(Markdown(report))
        return report

    except ThrottlerError as e:
        logging.error(f"❌ API thất bại sau {e.attempts} lần retry: {e.last_exception}")
        return f"Lỗi API: {e}"
    except Exception as e:
        logging.error(f"🚨 Lỗi không xác định khi gọi OpenAI: {e}")
        return f"Lỗi không xác định: {e}"

if __name__ == "__main__":
    demo_history = [
        {"question": "Nếu 3x + 5 = 20, tìm x?", "skill": "Algebra", "answered_correctly": True},
        {"question": "Tính diện tích hình tròn bán kính 4.", "skill": "Geometry", "answered_correctly": False},
        {"question": "Một đường thẳng có hệ số góc bằng 2, đi qua (1,3)...", "skill": "Functions", "answered_correctly": True},
    ]
    report = evaluate_student_performance(demo_history, 0.85)
