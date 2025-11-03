"""
sat_demo.py
-----------------------------------
Menu CLI chính cho hệ thống SAT AI.
Kết nối các module:
- CLI thi thử thích ứng (run_sat_simulation)
- CLI sinh câu hỏi SAT mới (generate_questions)
"""

import os
import sys
from dotenv import load_dotenv

# ============ KHỞI TẠO ============
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
load_dotenv(dotenv_path=env_path)
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

# ============ KIỂM TRA ENV ============
def check_env():
    if not OPENAI_KEY:
        print("⚠️  Bạn chưa thiết lập OPENAI_API_KEY trong file .env")
        print("➡️  Ví dụ nội dung .env:")
        print("OPENAI_API_KEY=sk-xxxx\nOPENAI_MODEL=gpt-4o-mini\n")
        sys.exit(1)


# ============ MENU CLI ============
def main():
    check_env()

    print("\n🎓 SAT AI SYSTEM — CLI DEMO")
    print("────────────────────────────")
    print("1️⃣  Làm bài thi thích ứng (Adaptive Test)")
    print("2️⃣  Sinh câu hỏi SAT mới (Question Generator)")
    print("0️⃣  Thoát")
    print("────────────────────────────")

    choice = input("👉  Chọn chức năng (0–2): ").strip()

    if choice == "1":
        from cli.run_sat_simulation import run_sat_demo
        run_sat_demo()

    elif choice == "2":
        from cli.generate_questions import run_question_generator
        run_question_generator()

    elif choice == "0":
        print("👋  Tạm biệt! Hẹn gặp lại.")
        sys.exit(0)

    else:
        print("⚠️  Lựa chọn không hợp lệ, vui lòng nhập 0–2.")


# ============ ENTRYPOINT ============
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑  Đã dừng chương trình.")
