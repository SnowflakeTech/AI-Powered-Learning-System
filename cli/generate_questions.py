"""
cli/generate_questions.py
-----------------------------------
CLI tool: Sinh câu hỏi SAT tự động bằng OpenAI.
Kết hợp với module sat_ai_core.question_generator.
"""

import os
import time
import logging
from dotenv import load_dotenv
from sat_ai_core.question_generator import generate_batch, save_to_bank, GEN_SKILLS

# ============ CẤU HÌNH ============
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
load_dotenv(dotenv_path=env_path)
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

# ============ CLI CHÍNH ============

def run_question_generator():
    print("\n🚀 SAT Question Generator (OpenAI Edition)\n")

    # --- Đường dẫn ---
    items_path = input("📂 Đường dẫn tới file items.json (Enter = data/items.json): ").strip()
    if items_path == "":
        items_path = "data/items.json"
    os.makedirs(os.path.dirname(items_path) or ".", exist_ok=True)

    # --- Section ---
    section = input("📘 Section (Math / RW): ").strip().title()
    if section not in GEN_SKILLS:
        print("⚠️ Section không hợp lệ! Mặc định dùng Math.")
        section = "Math"

    # --- Skill ---
    skills = GEN_SKILLS[section]
    print(f"\n🎯 Các kỹ năng khả dụng: {', '.join(skills)}")
    skill = input("👉 Chọn skill (Enter = ngẫu nhiên): ").strip()
    if skill == "":
        import random
        skill = random.choice(skills)
        print(f"📌 Chọn ngẫu nhiên skill: {skill}")
    elif skill not in skills:
        print(f"⚠️ Skill không hợp lệ! Mặc định: {skills[0]}")
        skill = skills[0]

    # --- Độ khó ---
    difficulty = input("📈 Độ khó (easy / medium / hard, Enter = medium): ").strip().lower()
    if difficulty not in ("easy", "medium", "hard"):
        difficulty = "medium"

    # --- Số lượng ---
    try:
        n = int(input("🔢 Số lượng câu cần tạo (Enter = 3): ").strip() or 3)
        if n <= 0:
            raise ValueError
    except ValueError:
        print("⚠️ Giá trị không hợp lệ, dùng mặc định: 3.")
        n = 3

    # --- Xác nhận ---
    print(f"\n📋 Tóm tắt yêu cầu:")
    print(f"- Section: {section}")
    print(f"- Skill: {skill}")
    print(f"- Độ khó: {difficulty}")
    print(f"- Số lượng: {n}")
    confirm = input("\n✅ Xác nhận? (Enter = tiếp tục, 'q' = hủy): ").strip().lower()
    if confirm == "q":
        print("🛑 Hủy thao tác.")
        return

    # --- Sinh câu hỏi ---
    print("\n🤖 Đang sinh câu hỏi bằng OpenAI...\n")
    start_time = time.time()

    try:
        new_items = generate_batch(section, skill, difficulty, n)
        if not new_items:
            print("⚠️ Không sinh được câu hỏi nào!")
            return

        save_to_bank(new_items, items_path)
        duration = time.time() - start_time
        print(f"\n✅ Đã sinh và lưu {len(new_items)} câu hỏi trong {duration:.1f}s.")
        print(f"📁 File lưu tại: {items_path}\n")

    except Exception as e:
        print(f"🚨 Lỗi khi sinh câu hỏi: {e}")


# ============ ENTRYPOINT ============
if __name__ == "__main__":
    run_question_generator()
