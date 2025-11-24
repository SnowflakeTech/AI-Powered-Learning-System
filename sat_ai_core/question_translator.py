# ===============================================
#  sat_ai_core/question_translator.py
#  ---------------------------------------------
#  Module dịch toàn bộ câu hỏi SAT sang nhiều ngôn ngữ
#  Giữ nguyên cấu trúc JSON, không thay đổi đáp án
#  Tích hợp OpenAI + ApiThrottler + Batch translation
# ===============================================

import os
import json
import logging
from typing import Dict, Any, List
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
from sat_ai_core.api_throttler import ApiThrottler

# ---------------------------
# LOGGING
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s"
)

# ---------------------------
# LOAD ENV
# ---------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
env_path = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("OPENAI_API_KEY")
model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

if not api_key:
    raise ValueError("❌ OPENAI_API_KEY chưa được cấu hình trong .env")

client = OpenAI(api_key=api_key)
throttler = ApiThrottler(min_interval=2.0, max_retries=5, max_wait=25.0)


# ===============================================
#  🔥 Prompt Builder
# ===============================================

def build_translate_prompt(item: Dict[str, Any], lang: str) -> str:
    """
    Tạo prompt dịch câu hỏi SAT sang ngôn ngữ mới
    mà KHÔNG thay đổi cấu trúc JSON
    """

    return f"""
Bạn là chuyên gia dịch thuật SAT quốc tế.

Nhiệm vụ của bạn:
- Dịch nội dung câu hỏi sang tiếng "{lang}"
- KHÔNG thay đổi cấu trúc hoặc logic của câu hỏi.
- KHÔNG thay đổi số lượng lựa chọn hoặc thứ tự đáp án.
- KHÔNG dịch các key JSON (id, section, skill, answer_index, ...).
- Đáp án đúng (answer_index) phải giữ nguyên.
- Chỉ dịch text bên trong:
    * question
    * passage (nếu có)
    * choices[]
- Tuyệt đối không thêm giải thích, không thêm ký tự khác.
- Không bọc output bằng ``` hoặc mã code.

Dưới đây là JSON gốc cần dịch:

{json.dumps(item, ensure_ascii=False, indent=2)}

Hãy trả về JSON đã dịch (JSON thuần).
""".strip()


# ===============================================
#  🔥 Translate 1 Item
# ===============================================

def translate_item(item: Dict[str, Any], lang: str) -> Dict[str, Any]:
    """Dịch 1 câu hỏi → trả về item JSON đã dịch"""

    prompt = build_translate_prompt(item, lang)

    response = throttler.safe_openai_chat(
        client,
        messages=[
            {"role": "system", "content": "Bạn là AI chuyên dịch câu hỏi SAT một cách an toàn."},
            {"role": "user", "content": prompt}
        ],
        model=model,
        temperature=0.1,
    )

    text = response.choices[0].message.content.strip()
    text = text.replace("```json", "").replace("```", "").strip()

    try:
        data = json.loads(text)
    except Exception as e:
        logging.error(f"❌ JSON dịch lỗi: {e}\n{text[:200]}")
        raise

    # đảm bảo JSON vẫn đầy đủ field
    for key in ["question", "choices", "answer_index"]:
        if key not in data:
            raise ValueError(f"❌ JSON bị thiếu trường bắt buộc: {key}")

    return data


# ===============================================
#  🔥 Translate All Items in data/*
# ===============================================

def translate_all(base_dir="data", target_lang="vi"):
    """
    Duyệt qua toàn bộ data/<Section>/<Skill>/items.json
    và dịch toàn bộ sang ngôn ngữ target_lang
    """

    out_base = os.path.join("data_translated", target_lang)
    os.makedirs(out_base, exist_ok=True)

    logging.info(f"🌍 Bắt đầu dịch sang ngôn ngữ: {target_lang}")

    total_translated = 0

    for root, _, files in os.walk(base_dir):
        if "items.json" not in files:
            continue

        section = os.path.basename(os.path.dirname(root))
        skill = os.path.basename(root)

        in_file = os.path.join(root, "items.json")

        try:
            with open(in_file, "r", encoding="utf-8") as f:
                items = json.load(f)
        except Exception as e:
            logging.warning(f"⚠️ Không đọc được {in_file}: {e}")
            continue

        # output folder
        out_dir = os.path.join(out_base, section, skill)
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, "items.json")

        translated = []

        for item in tqdm(items, desc=f"{section}/{skill}", ncols=100):
            try:
                new_item = translate_item(item, target_lang)
                translated.append(new_item)
                total_translated += 1
            except Exception as e:
                logging.warning(f"⚠️ Lỗi dịch item {item.get('id')}: {e}")
                continue

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(translated, f, ensure_ascii=False, indent=2)

        logging.info(f"📁 Đã dịch {len(translated)} câu → {out_file}")

    logging.info(f"\n🎯 HOÀN TẤT — Tổng số câu đã dịch: {total_translated}")


# ===============================================
#  🔥 CLI Entry
# ===============================================

if __name__ == "__main__":
    print("\n╔══════════════════════════════════════════════╗")
    print("║      🌍 SAT Question Translator — PRO        ║")
    print("╚══════════════════════════════════════════════╝\n")

    lang = input("Nhập mã ngôn ngữ cần dịch (vd: vi, ja, zh-cn, fr): ").strip()
    if not lang:
        lang = "vi"

    translate_all("data", target_lang=lang)
    print("\n🎉 Hoàn tất dịch câu hỏi!\n")
