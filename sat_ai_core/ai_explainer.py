import os
import re
import time
import hashlib
import sqlite3
import logging
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown
from sat_ai_core.api_throttler import ApiThrottler, ThrottlerError

PROMPT_VERSION = "v4"

env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
load_dotenv(dotenv_path=env_path)
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY chưa được thiết lập trong .env!")

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

def _build_tagged_prompt(question: str, correct_choice: str) -> str:
    return f"""
Bạn là gia sư SAT chuyên nghiệp. Trả lời CHÍNH XÁC theo MẪU THẺ dưới đây.
<MESSAGE>
<SUMMARY>
- Tóm tắt ngắn gọn đề bài (1–3 câu).
</SUMMARY>
<STEPS>
- Liệt kê các bước giải ngắn gọn, mỗi bước 1 gạch đầu dòng.
- Có thể kèm công thức ngắn trong `code` hoặc $math$.
</STEPS>
<CONCLUSION>
- Kết luận rõ ràng; nói đáp án đúng là gì và vì sao.
</CONCLUSION>
</MESSAGE>
[CÂU HỎI]: {question}
[ĐÁP ÁN ĐÚNG]: {correct_choice}
"""

def _extract_tag(text: str, tag: str) -> str:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, flags=re.DOTALL | re.IGNORECASE)
    return (m.group(1) if m else "").strip()

def _sanitize_lines(s: str) -> str:
    s = re.sub(r"^\s*#{1,6}\s*", "", s, flags=re.MULTILINE)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s*\n\s*\n\s*\n+", "\n\n", s)
    return s.strip()

def _steps_to_bullets(steps: str) -> str:
    parts = re.split(r"(?:\n|^)\s*[-•*]\s*|(?:\r?\n)+", steps)
    parts = [p.strip(" -•*\t") for p in parts if p and p.strip(" -•*\t")]
    more = []
    for p in parts:
        more.extend(re.split(r"\s*\d+\.\s+", p))
    bullets = [b for b in more if b.strip()]
    return "\n".join(f"- {b.strip()}" for b in bullets) if bullets else "- (Không có bước giải rõ ràng)"

def _format_response(raw: str, correct_choice: str) -> str:
    raw = _sanitize_lines(raw)
    summary = _extract_tag(raw, "SUMMARY")
    steps = _extract_tag(raw, "STEPS")
    concl = _extract_tag(raw, "CONCLUSION")
    if not (summary and steps and concl):
        text = _sanitize_lines(raw)
        blocks = re.split(r"(?i)(?:tóm tắt|summary)|(?:bước|steps)|(?:kết luận|conclusion)", text)
        summary = (blocks[1] if len(blocks) > 1 else text).strip()
        steps = (blocks[2] if len(blocks) > 2 else "").strip()
        concl = (blocks[3] if len(blocks) > 3 else "").strip()
    steps_md = _steps_to_bullets(steps)
    state = "✅ ĐÚNG" if correct_choice and correct_choice in (summary + steps_md + concl) else "🔎 Kiểm tra lại"
    return f"""
### 🧾 1 Tóm tắt đề
{summary}

---

### 🧠 2 Các bước giải
{steps_md}

---

### 🎯 3 Kết luận ({state})
{concl}
""".strip()

def explain_answer(question: str, correct_choice: str, verbose: bool = True) -> str:
    prompt = _build_tagged_prompt(question, correct_choice)
    key_src = f"{PROMPT_VERSION}::{MODEL}::{prompt}"
    key = hashlib.sha256(key_src.encode()).hexdigest()
    cached = _get_cache(key, MODEL)
    console = Console()
    if cached:
        if verbose:
            console.print("⚡ [bold yellow]Đã có cache, không cần gọi API.[/bold yellow]\n")
            console.print(Markdown(cached))
        return cached
    console.print(f"\n📘 [cyan]Đang giải thích câu hỏi bằng {MODEL}...[/cyan]\n")
    try:
        response = throttler.safe_openai_chat(
            client,
            messages=[
                {"role": "system", "content": "Bạn là gia sư SAT chuyên nghiệp, trả lời rõ ràng và dễ hiểu."},
                {"role": "user", "content": prompt},
            ],
            model=MODEL,
            temperature=0.6,
        )
        full_text = response.choices[0].message.content or ""
        token_count = len(full_text.split())
        formatted = _format_response(full_text, correct_choice)
        _set_cache(key, MODEL, formatted, token_count)
        console.print("\n✅ [green]Hoàn tất giải thích![/green]")
        logging.info(f"📊 Tokens ~ {token_count}")
        console.print("\n🎯 [bold]Kết quả:[/bold]\n")
        console.print(Markdown(formatted))
        return formatted
    except ThrottlerError as e:
        logging.error(f"❌ API thất bại sau {e.attempts} lần retry: {e.last_exception}")
        return f"Lỗi API: {e}"
    except Exception as e:
        logging.error(f"🚨 Lỗi không xác định: {e}")
        return f"Lỗi không xác định: {e}"

if __name__ == "__main__":
    q = "Một hình chữ nhật có chiều dài gấp đôi chiều rộng. Chu vi là 36 thì diện tích là bao nhiêu?"
    a = "81"
    explain_answer(q, a, verbose=True)
