"""
sat_ai_core/ai_explainer.py
-----------------------------------
Module giải thích câu trả lời SAT bằng mô hình OpenAI (ví dụ gpt-4o-mini).
Bao gồm caching SQLite, streaming hiển thị trên CLI, và định dạng Markdown.
"""

import os
import time
import re
import hashlib
import sqlite3
import logging
from openai import OpenAI
from dotenv import load_dotenv

# ============ KHỞI TẠO ============
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
load_dotenv(dotenv_path=env_path)
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY chưa được set trong .env!")

model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
client = OpenAI(api_key=api_key)

# ============ CẤU HÌNH DATABASE CACHE ============
DB_PATH = "ai_cache.db"
os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)

def _init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cache (
            key TEXT PRIMARY KEY,
            response TEXT NOT NULL
        );
    """)
    conn.commit()
    conn.close()

def _get_cache(key: str):
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("SELECT response FROM cache WHERE key=?", (key,)).fetchone()
    conn.close()
    return row[0] if row else None

def _set_cache(key: str, text: str):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT OR REPLACE INTO cache VALUES (?, ?)", (key, text))
    conn.commit()
    conn.close()

_init_db()

# ============ HÀM XỬ LÝ VĂN BẢN ============

def _format_response(raw: str, correct_choice: str) -> str:
    """Định dạng văn bản đầu ra thành Markdown 3 phần: tóm tắt, bước giải, kết luận"""
    raw = raw.strip()
    summary, steps, conclusion = "", "", ""

    parts = re.split(r"1|2|3|Tóm|Bước|Kết", raw, flags=re.IGNORECASE)
    if len(parts) >= 3:
        summary, steps, conclusion = parts[:3]
    else:
        segs = raw.split(".")
        if len(segs) > 2:
            summary, steps, conclusion = segs[0], " ".join(segs[1:-1]), segs[-1]
        else:
            summary = raw

    step_lines = re.split(r";|\n|•|-|\*", steps)
    step_lines = [s.strip() for s in step_lines if s.strip()]
    steps_md = "\n".join([f"- {s}" for s in step_lines])

    # Chuyển các phép tính sang LaTeX `$...$`
    for target in [summary, steps_md, conclusion]:
        target = re.sub(r"(\d+\s*[+\-*/=]\s*\d+)", r"$\1$", target)

    state = "✅ ĐÚNG" if correct_choice in raw else "❌ SAI"

    return f"""
1 **Tóm tắt đề:**
{summary.strip()}

2 **Các bước chính:**
{steps_md.strip()}

3 **Kết luận ({state}):**
{conclusion.strip()}
""".strip()


# ============ HÀM CHÍNH ============

def explain_answer(question: str, correct_choice: str) -> str:
    """
    Giải thích câu hỏi SAT bằng OpenAI.
    Có cache để tránh gọi lại API nhiều lần.
    """
    prompt = f"""
Bạn là gia sư SAT. Hãy giải thích ngắn gọn bằng Markdown, gồm 3 phần:
1 Tóm tắt đề
2 Các bước giải
3 Kết luận
---
Câu hỏi: {question}
Đáp án đúng: {correct_choice}
"""

    key = hashlib.sha256(prompt.encode()).hexdigest()
    cached = _get_cache(key)
    if cached:
        print("⚡ Đã có cache, không gọi lại API.\n")
        print(cached)
        return cached

    print(f"\n📘 Đang giải thích câu hỏi bằng {model}...\n")

    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Bạn là gia sư SAT chuyên giải thích rõ ràng và chính xác."},
            {"role": "user", "content": prompt},
        ],
        stream=True,
        temperature=0.6,
    )

    full_text = ""
    token_count = 0

    for chunk in stream:
        delta = chunk.choices[0].delta
        text = getattr(delta, "content", None)
        if text:
            print(text, end="", flush=True)
            full_text += text
            token_count += len(text.split())
            time.sleep(0.003)

    print("\n\n✅ Hoàn tất giải thích!")
    logging.info(f"📊 Tokens ước lượng: {token_count}")

    formatted = _format_response(full_text, correct_choice)
    print("\n🎯 Kết quả format:\n")
    print(formatted)

    _set_cache(key, formatted)
    return formatted


# ============ DEMO ============
if __name__ == "__main__":
    q = "Một hình chữ nhật có chiều dài gấp đôi chiều rộng. Chu vi là 36 thì diện tích là bao nhiêu?"
    a = "81"
    explain_answer(q, a)
