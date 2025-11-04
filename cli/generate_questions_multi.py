import os
import json
import uuid
import logging
import random
from tqdm import tqdm
from typing import List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv
from sat_ai_core.api_throttler import ApiThrottler
from sat_ai_core.question_generator import generate_irt_params

# ===== Config =====
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("OPENAI_API_KEY")
model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
client = OpenAI(api_key=api_key)
throttler = ApiThrottler(min_interval=2.0, max_retries=5, per_model=True)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

# ===== Prompt =====
def make_reform_prompt(item: Dict[str, Any]) -> str:
    """Sinh câu hỏi mới dựa trên câu gốc"""
    base_q = item.get("question", "")
    section = item.get("section", "Math")
    skill = item.get("skill", "Unknown")
    difficulty = item.get("difficulty", "medium")

    return f"""
Bạn là chuyên gia biên soạn đề thi SAT {section}.
Hãy tạo 1 biến thể mới của câu hỏi dưới đây, giữ nguyên kỹ năng ({skill}) và độ khó tương đương ({difficulty}),
nhưng thay đổi ngữ cảnh, số liệu hoặc cách diễn đạt. Đừng sao chép lại nguyên văn.

Câu gốc:
{base_q}

Đáp án gốc:
{item['choices'][item['answer_index']]}

Kết quả trả về phải là JSON hợp lệ:
{{
  "id": "auto",
  "section": "{section}",
  "skill": "{skill}",
  "question": "Câu hỏi mới...",
  "choices": ["A ...", "B ...", "C ...", "D ..."],
  "answer_index": <0-3>,
  "difficulty": "{difficulty}"
}}
""".strip()


# ===== Generate variant =====
def generate_variant(item: Dict[str, Any]) -> Dict[str, Any]:
    prompt = make_reform_prompt(item)
    response = throttler.safe_openai_chat(
        client,
        messages=[
            {"role": "system", "content": "You are an expert SAT question writer."},
            {"role": "user", "content": prompt},
        ],
        model=model,
        temperature=0.8,
    )

    text = response.choices[0].message.content.strip()
    text = text.replace("```json", "").replace("```", "").strip()

    data = json.loads(text)
    data["id"] = str(uuid.uuid4())
    irt = generate_irt_params(data.get("difficulty", "medium"))
    return {"item": data, "irt": {"id": data["id"], **irt}}


# ===== Process all folders =====
def expand_all_questions(base_dir="data", n_variants=2):
    total_new = 0
    for root, _, files in os.walk(base_dir):
        if "items.json" in files and "irt_params.json" in files:
            section = os.path.basename(os.path.dirname(root))
            skill = os.path.basename(root)
            logging.info(f"📘 Đang xử lý: {section}/{skill}")

            items_path = os.path.join(root, "items.json")
            irt_path = os.path.join(root, "irt_params.json")

            try:
                with open(items_path, "r", encoding="utf-8") as f:
                    items = json.load(f)
            except:
                logging.warning(f"⚠️ Không thể đọc {items_path}")
                continue

            new_items, new_irts = [], []

            for item in tqdm(items, desc=f"{section}/{skill}", ncols=100):
                for _ in range(n_variants):
                    try:
                        variant = generate_variant(item)
                        new_items.append(variant["item"])
                        new_irts.append(variant["irt"])
                    except Exception as e:
                        logging.warning(f"Lỗi sinh biến thể: {e}")
                        continue

            if new_items:
                items.extend(new_items)
                with open(items_path, "w", encoding="utf-8") as f:
                    json.dump(items, f, ensure_ascii=False, indent=2)

                with open(irt_path, "r", encoding="utf-8") as f:
                    irt_data = json.load(f)
                irt_data.extend(new_irts)
                with open(irt_path, "w", encoding="utf-8") as f:
                    json.dump(irt_data, f, ensure_ascii=False, indent=2)

                total_new += len(new_items)
                logging.info(f"✅ Thêm {len(new_items)} câu mới → {root}")

    logging.info(f"\n🎯 Hoàn tất: Sinh tổng cộng {total_new} câu hỏi mới.")


if __name__ == "__main__":
    print("\n╔════════════════════════════════════════╗")
    print("║   🚀 SAT Multi-Skill Question Expander  ║")
    print("╚════════════════════════════════════════╝\n")
    expand_all_questions("data", n_variants=2)
