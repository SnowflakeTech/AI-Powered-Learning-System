import os
import json
import uuid
import time
import random
import hashlib
import logging
from datetime import datetime
from typing import List, Dict, Optional
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
from sat_ai_core.api_throttler import ApiThrottler, ThrottlerError

env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
load_dotenv(dotenv_path=env_path)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY chưa được cấu hình trong .env")

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
client = OpenAI(api_key=api_key)
throttler = ApiThrottler(min_interval=2.0, max_retries=5, max_wait=25.0, per_model=True)

GEN_SKILLS = {
    "Math": ["Algebra", "Geometry", "Functions", "Statistics", "Ratios & Proportions"],
    "RW": ["Vocabulary", "Information & Ideas", "Craft & Structure", "Expression of Ideas", "Standard English Conventions"],
}

def make_prompt(section: str, skill: str, difficulty: str) -> str:
    if section == "Math":
        return f"""
Bạn là chuyên gia ra đề SAT Math. Hãy tạo 1 câu hỏi trắc nghiệm.

YÊU CẦU:
- Skill: {skill}
- Độ khó: {difficulty}
- Có biểu thức toán LaTeX ($...$)
- Có 4 đáp án A/B/C/D
- Một đáp án đúng duy nhất
- Không có lời giải

Kết quả trả về JSON:
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
    return f"""
Bạn là chuyên gia ra đề SAT Reading & Writing.

YÊU CẦU:
- Skill: {skill}
- Độ khó: {difficulty}
- Có 1 đoạn passage ≤ 70 từ
- Có 4 đáp án A/B/C/D
- Một đáp án đúng duy nhất
- Không có lời giải

Kết quả trả về JSON:
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

def generate_irt_params(difficulty: str) -> Dict[str, float]:
    d = difficulty.lower()
    if "easy" in d:
        a, b = random.uniform(0.8, 1.2), random.uniform(-1.5, -0.5)
    elif "hard" in d:
        a, b = random.uniform(1.2, 1.8), random.uniform(0.5, 1.5)
    else:
        a, b = random.uniform(1.0, 1.5), random.uniform(-0.5, 0.5)
    return {"a": round(a, 2), "b": round(b, 2), "c": 0.25}

def _try_parse_json(text: str) -> Optional[Dict]:
    try:
        clean = text.strip().replace("```json", "").replace("```", "")
        return json.loads(clean)
    except Exception:
        fixed = text.replace("\n", " ").replace("“", "\"").replace("”", "\"")
        try:
            return json.loads(fixed)
        except Exception:
            return None

def _validate_item(data: Dict) -> bool:
    required = ["section", "skill", "question", "choices", "answer_index", "difficulty"]
    return all(k in data for k in required)

def generate_sat_question(section: str, skill: str, difficulty: str) -> Optional[Dict]:
    prompt = make_prompt(section, skill, difficulty)
    try:
        response = throttler.safe_openai_chat(
            client,
            messages=[
                {"role": "system", "content": "You are an expert SAT question writer."},
                {"role": "user", "content": prompt},
            ],
            model=MODEL,
            temperature=0.7,
        )
        raw = response.choices[0].message.content.strip()
        data = _try_parse_json(raw)
        if not data or not _validate_item(data):
            logging.warning("⚠️ JSON không hợp lệ, bỏ qua.")
            return None

        qid = str(uuid.uuid4())
        irt = generate_irt_params(data["difficulty"])
        hash_id = hashlib.sha1((data["question"] + str(time.time())).encode()).hexdigest()[:12]

        data.update({
            "id": qid,
            "created_at": datetime.now().isoformat(),
            "model_used": MODEL,
            "hash_id": hash_id
        })
        return {"item": data, "irt": {"id": qid, **irt}}

    except ThrottlerError as e:
        logging.error(f"❌ Lỗi API (retry={e.attempts}): {e.last_exception}")
    except Exception as e:
        logging.error(f"🚨 Lỗi không xác định: {e}")
    return None

def generate_batch(section: Optional[str], skill: Optional[str], difficulty: str, n: int) -> List[Dict]:
    if not section:
        section = random.choice(list(GEN_SKILLS.keys()))
    if not skill:
        skill = random.choice(GEN_SKILLS[section])

    new_items, new_irt = [], []
    with tqdm(total=n, desc=f"{section}/{skill}/{difficulty}") as bar:
        for _ in range(n):
            res = generate_sat_question(section, skill, difficulty)
            if res:
                new_items.append(res["item"])
                new_irt.append(res["irt"])
            bar.update(1)
    return new_items, new_irt, section, skill

def save_to_bank(new_items: List[Dict], new_irt: List[Dict], section: str, skill: str):
    base_dir = os.path.join("data", section, skill)
    os.makedirs(base_dir, exist_ok=True)
    items_path = os.path.join(base_dir, "items.json")
    irt_path = os.path.join(base_dir, "irt_params.json")

    def load_json(path): 
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except: 
            return []

    items, irts = load_json(items_path), load_json(irt_path)
    existing_hashes = {i.get("hash_id") for i in items}

    new_unique = [i for i in new_items if i["hash_id"] not in existing_hashes]
    irts_unique = [r for r in new_irt if r["id"] in {i["id"] for i in new_unique}]

    if not new_unique:
        logging.warning("⚠️ Không có câu hỏi mới (trùng hash).")
        return

    items.extend(new_unique)
    irts.extend(irts_unique)
    with open(items_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    with open(irt_path, "w", encoding="utf-8") as f:
        json.dump(irts, f, ensure_ascii=False, indent=2)
    logging.info(f"📦 Lưu {len(new_unique)} câu hỏi mới vào {base_dir}")

if __name__ == "__main__":
    print("🚀 Sinh batch SAT câu hỏi tự động")
    for diff in ["easy", "medium", "hard"]:
        items, irts, section, skill = generate_batch(None, None, diff, 3)
        save_to_bank(items, irts, section, skill)
