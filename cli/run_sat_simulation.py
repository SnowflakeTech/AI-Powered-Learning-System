import os
import time
import math
import json
import logging
from dotenv import load_dotenv
from typing import List, Dict, Any, Tuple
from sat_ai_core import irt_core, question_selector, ai_explainer, ai_evaluator

RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
BLUE = "\033[94m"

env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
load_dotenv(dotenv_path=env_path)
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

def load_all_data(base_dir="data") -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, float]]]:
    items, irt_params = [], {}
    for root, _, files in os.walk(base_dir):
        if "items.json" in files and "irt_params.json" in files:
            try:
                with open(os.path.join(root, "items.json"), "r", encoding="utf-8") as f:
                    loaded_items = json.load(f)
                    for it in loaded_items:
                        section = os.path.basename(os.path.dirname(root))
                        skill = os.path.basename(root)
                        it["section"] = section
                        it["skill"] = it.get("skill", skill)
                    items.extend(loaded_items)
                with open(os.path.join(root, "irt_params.json"), "r", encoding="utf-8") as f:
                    params = json.load(f)
                    for p in params:
                        irt_params[str(p["id"])] = p
                logging.info(f"Loaded {len(loaded_items)} items from {root}")
            except Exception as e:
                logging.warning(f"Cannot read {root}: {e}")
    if not items:
        logging.warning("No nested data found, fallback to old data/items.json")
        try:
            with open("data/items.json", "r", encoding="utf-8") as f:
                items = json.load(f)
            with open("data/irt_params.json", "r", encoding="utf-8") as f:
                params = json.load(f)
                irt_params = {str(p['id']): p for p in params}
        except Exception as e:
            logging.error(f"Failed fallback: {e}")
    return items, irt_params

def determine_section_from_skill(skill: str) -> str:
    rw_skills = ["Vocabulary", "Information & Ideas", "Craft & Structure",
                 "Expression of Ideas", "Standard English Conventions"]
    return "RW" if skill in rw_skills else "Math"

def run_sat_demo():
    items, irt_params = load_all_data()
    if not items:
        print(f"{RED}Khong tim thay du lieu cau hoi!{RESET}")
        return

    all_skills = sorted({item.get("skill", "Unknown") for item in items})
    print(f"\n{BOLD}{CYAN}He thong co {len(items)} cau hoi tu {len(all_skills)} ky nang.{RESET}\n")
    print(f"{MAGENTA}Cac ky nang kha dung:{RESET}")
    for i, sk in enumerate(all_skills, 1):
        print(f"  {CYAN}{i}.{RESET} {sk}")

    raw = input(f"\nChon ky nang muon tap trung (Enter = tat ca): ").strip()
    focus_skill = None
    if raw:
        if raw.isdigit() and 1 <= int(raw) <= len(all_skills):
            focus_skill = all_skills[int(raw) - 1]
        elif raw in all_skills:
            focus_skill = raw
        else:
            print(f"{YELLOW}Khong hop le, dung tat ca ky nang.{RESET}")

    print(f"\nKy nang dang tap trung: {focus_skill or 'Tat ca'}")

    if focus_skill:
        section = determine_section_from_skill(focus_skill)
        items = [it for it in items if it.get("section") == section]
        filtered = [it for it in items if focus_skill.lower() in it.get("skill", '').lower()]
        if filtered:
            items = filtered
            print(f"{GREEN}Da loc {len(filtered)} cau hoi cho ky nang {focus_skill}.{RESET}")
        else:
            print(f"{YELLOW}Khong co cau hoi phu hop, dung tat ca trong Section {section}.{RESET}")

    if not items:
        print(f"{RED}Khong co cau hoi phu hop!{RESET}")
        return

    try:
        n = int(input(f"\nNhap so cau muon lam (Enter = 5, max {len(items)}): ").strip() or 5)
        n = min(n, len(items))
    except ValueError:
        n = 5

    print(f"\n{BOLD}{CYAN}BAT DAU BAI THI THICH UNG{RESET}\n")

    theta = 0.0
    asked, answered, history = [], [], []
    start_time = time.time()

    for step in range(1, n + 1):
        item = question_selector.select_next_item(theta, asked, items, irt_params,
                                                  history=history, focus_skill=focus_skill, top_k=4)
        if not item:
            print(f"{YELLOW}Het cau hoi phu hop.{RESET}")
            break

        print(f"\n{BLUE}Cau {step}:{RESET} {item['question']}")
        for i, choice in enumerate(item["choices"], 1):
            print(f"  {i}. {choice}")

        ans = input("Chon dap an (1–4 hoac q de thoat): ").strip().lower()
        if ans == "q":
            print(f"{RED}Ket thuc som.{RESET}")
            break
        if not ans.isdigit() or not (1 <= int(ans) <= 4):
            print(f"{YELLOW}Lua chon khong hop le.{RESET}")
            continue

        ans_idx = int(ans) - 1
        correct = int(ans_idx == item["answer_index"])
        print(f"{GREEN}Dung!{RESET}" if correct else f"{RED}Sai.{RESET}")

        asked.append(str(item["id"]))
        answered.append((str(item["id"]), correct))
        theta, se = irt_core.update_theta_map(theta, answered, irt_params)

        try:
            correct_choice = item["choices"][item["answer_index"]]
            explanation = ai_explainer.explain_answer(item["question"], correct_choice)
        except Exception as e:
            explanation = f"{YELLOW}Loi AI: {e}{RESET}"

        print(f"\n{MAGENTA}Giai thich AI:{RESET}\n{explanation}")
        print(f"{CYAN}Theta hien tai: {theta:.2f} ± {se:.2f}{RESET}")

        history.append({
            "id": item["id"],
            "question": item["question"],
            "answered_correctly": bool(correct),
            "theta": theta,
            "skill": item.get("skill", "Unknown"),
            "section": item.get("section", "Unknown"),
        })

        if math.isfinite(se) and se < 0.25:
            print(f"{GREEN}Do tin cay cao (SE = {se:.3f}){RESET}")
            break

    print(f"\n{BOLD}{CYAN}KET THUC BAI THI{RESET}")
    print(f"Ket qua cuoi: Theta = {theta:.2f}\n")

    if history:
        print(f"{MAGENTA}Dang tao bao cao AI...{RESET}")
        try:
            report = ai_evaluator.evaluate_student_performance(history, theta)
        except Exception as e:
            report = f"{RED}Loi danh gia: {e}{RESET}"
        os.makedirs("results", exist_ok=True)
        with open("results/evaluation_report.txt", "w", encoding="utf-8") as f:
            f.write(report or "")
        print(f"{GREEN}Bao cao luu tai: results/evaluation_report.txt{RESET}")
        print(f"\n{BLUE}Bao cao tong ket:{RESET}\n{report}")
    else:
        print(f"{YELLOW}Khong co du lieu danh gia.{RESET}")

if __name__ == "__main__":
    run_sat_demo()
