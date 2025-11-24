"""
SAT FULL EXAM BANK GENERATOR
Sinh đúng tổng 98 câu theo chuẩn Digital SAT
- RW: 54 câu
- Math: 44 câu
Phân bố độ khó theo tỷ lệ chính thức
Phủ đều toàn bộ 29 SAT skills
"""

import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from sat_ai_core.api_throttler import ApiThrottler
from sat_ai_core.question_generator_sat_full import make_prompt, to_json

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
env_path = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path=env_path)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
throttler = ApiThrottler(min_interval=2.0)


# =======================================
# SAT OFFICIAL SKILLS
# =======================================
SAT_SKILLS = {
    "RW": [
        "Central Ideas and Details",
        "Craft and Structure",
        "Inferences",
        "Command of Evidence",
        "Words in Context",
        "Text Structure and Purpose",
        "Cross-text Relationships",
        "Sentence Structure",
        "Boundaries",
        "Form, Agreement, Possessives",
        "Transitions",
        "Rhetorical Synthesis",
        "Effective Language Use"
    ],
    "Math": [
        "Linear Equations",
        "Linear Inequalities",
        "Systems of Linear Equations",
        "Equivalent Expressions",
        "Quadratic Functions",
        "Exponential Functions",
        "Polynomial Algebra",
        "Rational Functions",
        "Ratios and Proportions",
        "Percentages",
        "Data Interpretation",
        "Probability and Statistics",
        "Area and Volume",
        "Angles and Lines",
        "Trigonometric Functions",
        "Geometric Transformations"
    ]
}

# =======================================
# OFFICIAL DISTRIBUTIONS
# =======================================
DIFFICULTY_DIST_RW = {"easy": 0.30, "medium": 0.40, "hard": 0.30}
DIFFICULTY_DIST_MATH = {"easy": 0.35, "medium": 0.40, "hard": 0.25}

TOTAL_RW = 54
TOTAL_MATH = 44


# =======================================
# CALCULATE HOW MANY QUESTIONS PER SKILL
# =======================================
def distribute(total, n_skills):
    """Phân đều câu hỏi cho mỗi skill"""
    base = total // n_skills
    remainder = total % n_skills
    arr = [base] * n_skills
    for i in range(remainder):
        arr[i] += 1
    return arr


def difficulty_split(total, dist):
    """Trả về số câu easy/medium/hard theo tỉ lệ"""
    e = round(total * dist["easy"])
    m = round(total * dist["medium"])
    h = total - e - m
    return [("easy", e), ("medium", m), ("hard", h)]


# =======================================
# GENERATE ONE QUESTION
# =======================================
def generate_one(section, skill, difficulty):
    from sat_ai_core.question_generator_sat_full import generate_one as gen
    return gen(section, skill, difficulty)


# =======================================
# GENERATE THE FULL BANK
# =======================================
def generate_sat_exam_bank(outfile="sat_exam_bank.json"):
    results = []

    # RW
    rw_skills = SAT_SKILLS["RW"]
    rw_counts = distribute(TOTAL_RW, len(rw_skills))
    rw_difficulties = difficulty_split(TOTAL_RW, DIFFICULTY_DIST_RW)

    # Math
    math_skills = SAT_SKILLS["Math"]
    math_counts = distribute(TOTAL_MATH, len(math_skills))
    math_difficulties = difficulty_split(TOTAL_MATH, DIFFICULTY_DIST_MATH)

    # Generate RW
    print("\n📚 GENERATING RW QUESTIONS...")
    for skill, n in zip(rw_skills, rw_counts):
        for diff, cnt in rw_difficulties:
            for _ in range(cnt // len(rw_skills)):
                q = generate_one("RW", skill, diff)
                results.append(q)

    # Generate Math
    print("\n🧮 GENERATING MATH QUESTIONS...")
    for skill, n in zip(math_skills, math_counts):
        for diff, cnt in math_difficulties:
            for _ in range(cnt // len(math_skills)):
                q = generate_one("Math", skill, diff)
                results.append(q)

    # Export
    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n🎉 DONE! Generated {len(results)} questions → {outfile}")
    return results


if __name__ == "__main__":
    generate_sat_exam_bank()
