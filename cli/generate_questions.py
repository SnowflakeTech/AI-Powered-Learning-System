import os
import time
import random
import logging
from tqdm import tqdm
from dotenv import load_dotenv
from sat_ai_core.sat_full_bank_generator import generate_batch, save_to_bank, GEN_SKILLS

RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
BLUE = "\033[94m"

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
env_path = os.path.join(BASE_DIR, ".env")
os.makedirs("logs", exist_ok=True)

load_dotenv(dotenv_path=env_path)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/question_gen.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

def banner():
    print(f"\n{BOLD}{CYAN}╔══════════════════════════════════════════════════╗{RESET}")
    print(f"{BOLD}{CYAN}║     🚀 SAT Question Generator — PRO Edition       ║{RESET}")
    print(f"{BOLD}{CYAN}╚══════════════════════════════════════════════════╝{RESET}\n")

def run_question_generator():
    banner()
    sections = list(GEN_SKILLS.keys())
    print(f"{MAGENTA}📘 Chọn Section:{RESET}")
    for i, s in enumerate(sections, 1):
        print(f"  {CYAN}{i}.{RESET} {s}")
    raw_section = input(f"\n👉 Nhập số (1–2, Enter = ngẫu nhiên): ").strip()
    section = random.choice(sections) if raw_section == "" else (
        sections[int(raw_section) - 1] if raw_section.isdigit() and 1 <= int(raw_section) <= len(sections) else random.choice(sections)
    )
    print(f"{GREEN}🎯 Section đã chọn:{RESET} {section}")

    skills = GEN_SKILLS[section]
    print(f"\n{MAGENTA}🎯 Các kỹ năng khả dụng trong {section}:{RESET}")
    for i, s in enumerate(skills, 1):
        print(f"  {CYAN}{i}.{RESET} {s}")
    raw_skill = input(f"\n👉 Nhập số (1–{len(skills)}, Enter = ngẫu nhiên): ").strip()
    skill = random.choice(skills) if raw_skill == "" else (
        skills[int(raw_skill) - 1] if raw_skill.isdigit() and 1 <= int(raw_skill) <= len(skills) else random.choice(skills)
    )
    print(f"{GREEN}🎯 Skill đã chọn:{RESET} {skill}")

    difficulties = ["easy", "medium", "hard"]
    print(f"\n{MAGENTA}📈 Chọn độ khó:{RESET}")
    for i, d in enumerate(difficulties, 1):
        print(f"  {CYAN}{i}.{RESET} {d.title()}")
    raw_diff = input(f"\n👉 Nhập số (1–3, Enter = 2): ").strip()
    difficulty = "medium" if raw_diff == "" else (
        difficulties[int(raw_diff) - 1] if raw_diff.isdigit() and 1 <= int(raw_diff) <= 3 else "medium"
    )
    print(f"{GREEN}📊 Độ khó đã chọn:{RESET} {difficulty}")

    try:
        n = int(input(f"\n🔢 Số lượng câu cần tạo (Enter = 3): ").strip() or 3)
        if n <= 0:
            raise ValueError
    except ValueError:
        print(f"{YELLOW}⚠️ Giá trị không hợp lệ, mặc định: 3{RESET}")
        n = 3

    print(f"\n{BOLD}📋 Tóm tắt yêu cầu:{RESET}")
    print(f"  Section     : {section}")
    print(f"  Skill       : {skill}")
    print(f"  Độ khó      : {difficulty}")
    print(f"  Số lượng    : {n}")

    confirm = input(f"\n✅ Xác nhận? (Enter = tiếp tục, 'q' = hủy): ").strip().lower()
    if confirm == "q":
        print(f"{RED}🛑 Hủy thao tác.{RESET}")
        return

    print(f"\n{CYAN}🤖 Đang sinh câu hỏi bằng OpenAI...{RESET}\n")
    start = time.time()
    try:
        new_items, new_irt, section, skill = generate_batch(section, skill, difficulty, n)
        for _ in tqdm(range(10), desc=f"{BLUE}🧠 Đang xử lý dữ liệu...{RESET}", ncols=80):
            time.sleep(0.05)

        if not new_items:
            print(f"{YELLOW}⚠️ Không sinh được câu hỏi nào.{RESET}")
            return

        save_to_bank(new_items, new_irt, section, skill)
        elapsed = time.time() - start

        print(f"\n{GREEN}✅ Đã sinh và lưu {len(new_items)} câu hỏi trong {elapsed:.1f}s.{RESET}")
        print(f"{CYAN}📁 Thư mục lưu tại:{RESET} data/{section}/{skill}")
        logging.info(f"Sinh {len(new_items)} câu hỏi {section}/{skill}/{difficulty}")

        preview = new_items[0]
        print(f"\n{BOLD}{MAGENTA}📖 Xem trước câu hỏi đầu tiên:{RESET}")
        print(f"  🧩 {preview.get('question', 'Không có dữ liệu')}")
        for i, ch in enumerate(preview.get('choices', []), 1):
            print(f"   {chr(64+i)}. {ch}")
        print(f"  ✅ Đáp án đúng: {chr(65 + preview.get('answer_index', 0))}")

    except Exception as e:
        print(f"{RED}🚨 Lỗi khi sinh câu hỏi:{RESET} {e}")
        logging.exception("Lỗi khi sinh câu hỏi")

if __name__ == "__main__":
    run_question_generator()
