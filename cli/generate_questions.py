"""
cli/generate_questions.py
-----------------------------------
CLI tool: Sinh câu hỏi SAT tự động bằng OpenAI.
Kết hợp với module sat_ai_core.question_generator.
Có màu ANSI để hiển thị chuyên nghiệp trong terminal.
"""

import os
import time
import logging
from dotenv import load_dotenv
from sat_ai_core.question_generator import generate_batch, save_to_bank, GEN_SKILLS

# ============ ANSI COLORS ============
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
BLUE = "\033[94m"

# ============ CẤU HÌNH ============

env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
load_dotenv(dotenv_path=env_path)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s"
)

# ============ HÀM CHÍNH ============

def run_question_generator():
    print(f"\n{BOLD}{CYAN}🚀 SAT Question Generator (OpenAI Edition){RESET}\n")

    # --- Đường dẫn file ---
    items_path = input(f"{BLUE}📂 Đường dẫn tới file items.json (Enter = data/items.json): {RESET}").strip()
    if items_path == "":
        items_path = "data/items.json"
    os.makedirs(os.path.dirname(items_path) or ".", exist_ok=True)

    # --- Section ---
    print(f"\n{MAGENTA}📘 Chọn Section:{RESET}")
    sections = list(GEN_SKILLS.keys())
    for i, s in enumerate(sections, 1):
        print(f"  {CYAN}{i}.{RESET} {s}")

    raw_section = input(f"\n👉 Chọn Section (1 hoặc 2, Enter = Math): ").strip()

    if raw_section == "":
        section = "Math"
    elif raw_section.isdigit() and 1 <= int(raw_section) <= len(sections):
        section = sections[int(raw_section) - 1]
    else:
        raw_upper = raw_section.strip().title()
        section = raw_upper if raw_upper in sections else "Math"
        if raw_upper not in sections:
            print(f"{YELLOW}⚠️ Lựa chọn không hợp lệ, dùng Math làm mặc định.{RESET}")

    print(f"{GREEN}🎯 Section đã chọn:{RESET} {section}")

    # --- Skill ---
    skills = GEN_SKILLS[section]
    print(f"\n{MAGENTA}🎯 Các kỹ năng khả dụng trong {section}:{RESET}\n")
    for i, s in enumerate(skills, 1):
        print(f"  {CYAN}{i}.{RESET} {s}")

    raw_skill = input(f"\n👉 Chọn skill (nhập số hoặc tên, Enter = ngẫu nhiên): ").strip()

    if raw_skill == "":
        import random
        skill = random.choice(skills)
        print(f"{BLUE}📌 Chọn ngẫu nhiên skill:{RESET} {skill}")
    elif raw_skill.isdigit() and 1 <= int(raw_skill) <= len(skills):
        skill = skills[int(raw_skill) - 1]
    else:
        raw_skill_cap = raw_skill.strip().title()
        skill = raw_skill_cap if raw_skill_cap in skills else skills[0]
        if raw_skill_cap not in skills:
            print(f"{YELLOW}⚠️ Skill không hợp lệ, mặc định:{RESET} {skills[0]}")

    print(f"{GREEN}🎯 Skill đã chọn:{RESET} {skill}")

    # --- Độ khó ---
    difficulty = input(f"\n📈 Độ khó (easy / medium / hard, Enter = medium): ").strip().lower()
    if difficulty not in ("easy", "medium", "hard"):
        print(f"{YELLOW}⚠️ Độ khó không hợp lệ, mặc định: medium{RESET}")
        difficulty = "medium"

    # --- Số lượng ---
    try:
        n = int(input(f"🔢 Số lượng câu cần tạo (Enter = 3): ").strip() or 3)
        if n <= 0:
            raise ValueError
    except ValueError:
        print(f"{YELLOW}⚠️ Giá trị không hợp lệ, dùng mặc định: 3.{RESET}")
        n = 3

    # --- Xác nhận ---
    print(f"\n{BOLD}📋 Tóm tắt yêu cầu:{RESET}")
    print(f"- Section: {section}")
    print(f"- Skill: {skill}")
    print(f"- Độ khó: {difficulty}")
    print(f"- Số lượng: {n}")

    confirm = input(f"\n✅ Xác nhận? (Enter = tiếp tục, 'q' = hủy): ").strip().lower()
    if confirm == "q":
        print(f"{RED}🛑 Hủy thao tác.{RESET}")
        return

    # --- Sinh câu hỏi ---
    print(f"\n{CYAN}🤖 Đang sinh câu hỏi bằng OpenAI...{RESET}\n")
    start_time = time.time()

    try:
        new_items = generate_batch(section, skill, difficulty, n)
        if not new_items:
            print(f"{YELLOW}⚠️ Không sinh được câu hỏi nào!{RESET}")
            return

        save_to_bank(new_items, items_path)
        duration = time.time() - start_time
        print(f"\n{GREEN}✅ Đã sinh và lưu {len(new_items)} câu hỏi trong {duration:.1f}s.{RESET}")
        print(f"{CYAN}📁 File lưu tại:{RESET} {items_path}\n")

    except Exception as e:
        print(f"{RED}🚨 Lỗi khi sinh câu hỏi:{RESET} {e}")


# ============ ENTRYPOINT ============
if __name__ == "__main__":
    run_question_generator()
