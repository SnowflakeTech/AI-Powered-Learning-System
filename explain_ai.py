import os
import time
import sqlite3
import hashlib
import logging
import re
from typing import Optional
from google import genai

# === CLI Colors ===
GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RESET = "\033[0m"

# === Logging ===
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
)

# === API Key ===
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("❌ GOOGLE_API_KEY chưa được set!")
client = genai.Client(api_key=api_key)

MODEL = "gemini-2.5-flash"

# === SQLite Cache ===
DB_PATH = "api_cache.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS cache (
        key TEXT PRIMARY KEY,
        response TEXT NOT NULL
    );""")
    conn.commit()
    conn.close()

init_db()

def get_cache(key):
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("SELECT response FROM cache WHERE key=?", (key,)).fetchone()
    conn.close()
    return row[0] if row else None

def set_cache(key, text):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT OR REPLACE INTO cache VALUES(?,?)", (key, text))
    conn.commit()
    conn.close()


# ===============================================================
# ✅ Format nội dung sau khi stream: hệ thống 3 phần + LaTeX + bullet steps + chấm điểm
# ===============================================================
def post_process(raw: str, correct_choice: str) -> str:
    raw = raw.strip()

    # Tách bằng từ khóa hoặc fallback
    summary = ""
    steps = ""
    conclusion = ""

    parts = re.split(r"1️⃣|2️⃣|3️⃣|Tóm|Bước|Kết", raw, flags=re.IGNORECASE)
    if len(parts) >= 3:
        summary, steps, conclusion = parts[:3]
    else:
        # fallback → chia theo câu
        segs = raw.split(".")
        if len(segs) > 2:
            summary = segs[0]
            conclusion = segs[-1]
            steps = " ".join(segs[1:-1])
        else:
            summary = raw

    # ✅ Bullet hóa các bước dạng toán
    step_lines = re.split(r";|\n|•|-|\*", steps)
    step_lines = [s.strip() for s in step_lines if s.strip()]
    steps_md = "\n".join([f"- {s}" for s in step_lines])

    # ✅ Chuyển biểu thức ẩn sang LaTeX `$...$`
    steps_md = re.sub(r"(\d+\s*[+\-*/=]\s*\d+)", r"$\1$", steps_md)
    summary = re.sub(r"(\d+\s*[+\-*/=]\s*\d+)", r"$\1$", summary)
    conclusion = re.sub(r"(\d+\s*[+\-*/=]\s*\d+)", r"$\1$", conclusion)

    # ✅ Tự chấm điểm
    state = "✅ ĐÚNG" if correct_choice in raw else "❌ SAI"

    # ✅ Format chuẩn
    formatted = f"""
1️⃣ **Tóm tắt đề:**
{summary.strip()}

2️⃣ **Các bước chính:**
{steps_md.strip()}

3️⃣ **Kết luận ({state}):**
{conclusion.strip()}

"""
    return formatted.strip()


# ===============================================================
# ✅ Streaming Explain SAT
# ===============================================================
def explain_answer(question: str, correct_choice: str):
    prompt = (
        "Bạn là gia sư SAT. Giải thích rõ ràng dạng Markdown:\n"
        f"Câu hỏi: {question}\n"
        f"Đáp án đúng: {correct_choice}\n"
    )

    key = hashlib.sha256(prompt.encode()).hexdigest()
    cached = get_cache(key)
    if cached:
        print(f"{CYAN}⚡ Cache: đã lưu!{RESET}\n")
        print(cached)
        return cached

    print(f"{GREEN}📘 Giải thích Streaming...{RESET}\n")

    stream = client.models.generate_content_stream(
        model=MODEL,
        contents=prompt,
    )

    full = ""
    tokens = 0

    # ✅ Nhận từng chunk text
    for chunk in stream:
        if not hasattr(chunk, "text") or not chunk.text:
            continue

        tokens += len(chunk.text.split())
        print(chunk.text, end="", flush=True)
        full += chunk.text
        time.sleep(0.005)

    print("\n\n✅ Streaming Done!")
    logging.info(f"📊 Estimated tokens: {tokens}")

    # ✅ Hậu xử lý format mạnh mẽ
    formatted = post_process(full, correct_choice)

    print("\n\n🎯 Format hoàn chỉnh:\n")
    print(formatted)

    set_cache(key, formatted)
    return formatted


# ===============================================================
# ✅ Test
# ===============================================================
if __name__ == "__main__":
    q = "Một hình chữ nhật có chiều dài gấp đôi chiều rộng. Chu vi là 36 thì diện tích là bao nhiêu?"
    a = "81"
    explain_answer(q, a)
    q2 = "Nếu 3x + 5 = 20, thì giá trị của x là bao nhiêu?"
    a2 = "5"
    explain_answer(q2, a2)
