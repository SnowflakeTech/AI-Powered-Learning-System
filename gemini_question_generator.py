import os
import json
import uuid
import time
import random
import logging
from typing import List, Dict, Optional
from google import genai

# ============ CẤU HÌNH LOGGING ============
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

# ============ KẾT NỐI GEMINI ============
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("❌ Bạn chưa set GOOGLE_API_KEY!")
client = genai.Client(api_key=api_key)


# ============ DANH SÁCH SKILL ============
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
        "Standard English Conventions"
    ]
}


# ============ TẠO PROMPT THEO SECTION ============
def make_prompt(section: str, skill: str, difficulty: str) -> str:
    if section == "Math":
        return f"""
Bạn là chuyên gia ra đề SAT Math. Hãy tạo 1 câu hỏi SAT Math dạng trắc nghiệm.

YÊU CẦU:
- Skill: {skill}
- Độ khó: {difficulty}
- Có biểu thức toán LaTeX chuẩn ($ ... $)
- Có 4 đáp án A/B/C/D (choices)
- Một đáp án đúng DUY NHẤT
- Không có lời giải

OUTPUT DẠNG JSON CHUẨN:
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
    else:  # Reading & Writing
        return f"""
Bạn là chuyên gia ra đề SAT Reading & Writing.

YÊU CẦU:
- Skill: {skill}
- Độ khó: {difficulty}
- Thêm passage ≤ 70 từ liên quan chặt chẽ câu hỏi
- Một đáp án đúng duy nhất
- Không có lời giải
- Ngữ pháp/logic chuẩn SAT

OUTPUT DẠNG JSON CHUẨN:
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


# ============ CHUYỂN RESPON JSON SẠCH ============
def parse_json_output(raw_text: str) -> Optional[Dict]:
    raw_text = raw_text.strip()

    # Loại bỏ các phần thừa nếu AI thêm trước/sau JSON
    if raw_text.startswith("```"):
        raw_text = raw_text.split("```")[1]
    raw_text = raw_text.strip("{} \n ")
    raw_text = "{" + raw_text + "}"

    try:
        data = json.loads(raw_text)
        return data
    except Exception as e:
        logging.warning(f"⚠️ JSON parse error: {e}")
        return None


# ============ TẠO 1 CÂU HỎI ============
def generate_sat_question(section: str, skill: str, difficulty: str, retries: int = 3) -> Optional[Dict]:
    prompt = make_prompt(section, skill, difficulty)

    for attempt in range(1, retries + 1):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            data = parse_json_output(response.text)
            if isinstance(data, dict):
                data["id"] = str(uuid.uuid4())  # Gán ID mới
                return data

        except Exception as e:
            logging.warning(f"⚠️ Lỗi gen: {e}. Retry {attempt}/{retries}")
            time.sleep(1 + random.uniform(0, 1))

    logging.error("❌ Không thể sinh câu hỏi sau nhiều lần thử!")
    return None


# ============ BATCH GENERATOR ============
def generate_batch(section: str, skill: str, difficulty: str, n: int) -> List[Dict]:
    qs = []
    for _ in range(n):
        q = generate_sat_question(section, skill, difficulty)
        if q:
            qs.append(q)
    return qs


# ============ LƯU VÀO ITEM BANK ============
def save_to_bank(new_items: List[Dict], items_path: str):
    try:
        with open(items_path, "r", encoding="utf-8") as f:
            bank = json.load(f)
    except:
        bank = []

    bank.extend(new_items)

    with open(items_path, "w", encoding="utf-8") as f:
        json.dump(bank, f, ensure_ascii=False, indent=2)

    logging.info(f"✅ Đã lưu thêm {len(new_items)} câu hỏi vào: {items_path}")


# ============ CLI ============
if __name__ == "__main__":
    print("\n🚀 Gemini SAT Question Generator\n")
    items_path = input("📂 Đường dẫn Items JSON (ví dụ /mnt/data/items.json): ").strip()
    section = input("📍 Section? (Math hoặc RW): ").strip()
    skill = input(f"🎯 Skill ({', '.join(GEN_SKILLS.get(section, []))}): ").strip()
    difficulty = input("📈 Độ khó (easy/medium/hard): ").strip()
    n = int(input("🔢 Số lượng câu cần tạo: ").strip())

    qs = generate_batch(section, skill, difficulty, n)
    save_to_bank(qs, items_path)

    print("\n📌 HOÀN THÀNH! Câu hỏi mới đã nằm trong ngân hàng ✅\n")
