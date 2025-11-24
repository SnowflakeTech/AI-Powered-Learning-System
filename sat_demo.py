import os
import sys
import pathlib
from dotenv import load_dotenv

ROOT = pathlib.Path(__file__).parent   # thư mục sat_ai_core hoặc mức hiện tại
ENV_FILE = ROOT.parent / ".env"        # nhảy lên root project

load_dotenv(ENV_FILE)
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RED = "\033[91m"

def check_env():
    if not OPENAI_KEY:
        print(f"{RED}Chua thiet lap OPENAI_API_KEY trong file .env{RESET}")
        print("Vi du noi dung .env:")
        print("OPENAI_API_KEY=sk-xxxx")
        print("OPENAI_MODEL=gpt-4o-mini\n")
        sys.exit(1)

def main():
    check_env()
    print(f"\n{BOLD}{CYAN}SAT AI SYSTEM - CLI DEMO{RESET}")
    print("-" * 40)
    print("1. Lam bai thi thich ung (Adaptive Test)")
    print("2. Sinh cau hoi SAT moi (Question Generator)")
    print("0. Thoat")
    print("-" * 40)
    choice = input("Chon chuc nang (0-2): ").strip()
    if choice == "1":
        from cli.run_sat_simulation import run_sat_demo
        run_sat_demo()
    elif choice == "2":
        from cli.generate_questions import run_question_generator
        run_question_generator()
    elif choice == "0":
        print(f"{GREEN}Tam biet!{RESET}")
        sys.exit(0)
    else:
        print(f"{YELLOW}Lua chon khong hop le, vui long nhap 0-2.{RESET}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{RED}Da dung chuong trinh.{RESET}")
