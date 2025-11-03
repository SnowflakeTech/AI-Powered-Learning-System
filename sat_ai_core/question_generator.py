"""
sat_ai_core/question_generator.py
-----------------------------------
Sinh câu hỏi SAT tự động bằng OpenAI.
Dùng cho module CLI: cli/generate_questions.py
"""

import os
import json
import uuid
import time
import random
import logging
from typing import List, Dict, Optional
from openai import OpenAI
from dotenv import load_dotenv

# ===== Load .env từ thư mục gốc =====
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
load_dotenv(dotenv_path=env_path)

# ===== Logging =====
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

# ===== OpenAI Client =====
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ Bạn chưa thiết lập OPENAI_API_KEY trong .env!")
client = OpenAI(api_key=api_key)

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ===== Danh sách kỹ năng =====
GEN_SKILLS = {
    "Math": [
        "Algebra",
        "Geometry",
        "Functions",
        "Statistics",
        "Ratios & Proportions",
    ],
    "RW": [
        "Vocabulary",
        "Information & Ideas",
        "Craft & Structure",
        "Expression of Ideas",
        "Standard English Conventions",
    ],
}

# ==========================
# 🧠 Sinh Prompt cho AI
# ==========================
def make_prompt(section: str, skill: str, difficulty: str) -> str:
    """Tạo prompt ra đề chuẩn cho từng section."""
    if section == "Math":
        return f"""
Bạn là chuyên gia ra đề SAT Math. Hãy tạo 1 câu hỏi SAT dạng trắc nghiệm.

YÊU CẦU:
- Skill: {skill}
- Độ khó: {difficulty}
- Có biểu thức toán LaTeX chuẩn ($...$)
- Có 4 đáp án A/B/C/D
- Một đáp án đúng DUY NHẤT
- Không có lời giải

Kết quả trả về phải là JSON hợp lệ:
{{
  "id": "auto",
  "section": "Math",
  "skill": "{skill}",
  "question": "Câu hỏi ...",
  "choices": ["A ...", "B ...", "C ...", "D ..."],
  "answer_index": <0-3>,
  "difficulty": "{difficulty}"
}}
"""
    else:
        return f"""
Bạn là chuyên gia ra đề SAT Reading & Writing.

YÊU CẦU:
- Skill: {skill}
- Độ khó: {difficulty}
- Có 1 đoạn passage ≤ 70 từ
- Có 4 đáp án A/B/C/D
- Một đáp án đúng DUY NHẤT
- Không có lời giải

Kết quả trả về phải là JSON hợp lệ:
{{
  "id": "auto",
  "section": "RW",
  "skill": "{skill}",
  "passage": "Đoạn văn 50-70 từ...",
  "question": "Câu hỏi ...?",
  "choices": ["A ...", "B ...", "C ...", "D ..."],
  "answer_index": <0-3>,
  "difficulty": "{difficulty}"
}}
"""

# ==========================
# ⚙️ Sinh 1 câu hỏi
# ==========================
def generate_sat_question(section: str, skill: str, difficulty: str, retries: int = 3) -> Optional[Dict]:
    """Gọi OpenAI để sinh 1 câu hỏi SAT."""
    prompt = make_prompt(section, skill, difficulty)

    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert SAT question writer."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
            )

            raw_text = response.choices[0].message.content.strip()
            # Làm sạch JSON
            raw_text = raw_text.replace("```json", "").replace("```", "").strip()
            data = json.loads(raw_text)

            if isinstance(data, dict):
                data["id"] = str(uuid.uuid4())
                logging.info(f"✅ Sinh câu hỏi mới ({section}/{skill}, độ khó={difficulty})")
                return data

        except Exception as e:
            logging.warning(f"⚠️ Lỗi khi sinh câu hỏi (attempt {attempt}/{retries}): {e}")
            time.sleep(1 + random.random())

    logging.error("❌ Không thể sinh câu hỏi sau nhiều lần thử.")
    return None

# ==========================
# 🔁 Sinh nhiều câu hỏi
# ==========================
def generate_batch(section: str, skill: str, difficulty: str, n: int) -> List[Dict]:
    """Sinh một batch gồm n câu hỏi."""
    qs = []
    for i in range(n):
        q = generate_sat_question(section, skill, difficulty)
        if q:
            qs.append(q)
        else:
            logging.warning(f"⚠️ Bỏ qua câu hỏi thứ {i+1} vì lỗi sinh.")
    return qs

# ==========================
# 💾 Lưu câu hỏi vào ngân hàng
# ==========================
def save_to_bank(new_items: List[Dict], items_path: str):
    """Thêm câu hỏi mới vào file items.json."""
    try:
        with open(items_path, "r", encoding="utf-8") as f:
            bank = json.load(f)
    except:
        bank = []

    bank.extend(new_items)
    with open(items_path, "w", encoding="utf-8") as f:
        json.dump(bank, f, ensure_ascii=False, indent=2)

    logging.info(f"📦 Đã lưu thêm {len(new_items)} câu hỏi vào {items_path}")

# ==========================
# 🧪 Test độc lập
# ==========================
if __name__ == "__main__":
    print("🧪 Demo sinh 1 câu SAT (Math / Algebra / Easy)")
    q = generate_sat_question("Math", "Algebra", "easy")
    print(json.dumps(q, ensure_ascii=False, indent=2))
