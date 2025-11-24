"""
SAT FULL QUESTION GENERATOR — DIGITAL SAT 2025 EDITION
Sinh tất cả câu hỏi theo chuẩn SAT (34 skills).
Format hoàn toàn khớp hệ thống backend.
"""

import os
import json
import random
from dotenv import load_dotenv
from openai import OpenAI
from sat_ai_core.api_throttler import ApiThrottler

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
env_path = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path=env_path)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
throttler = ApiThrottler(min_interval=2.0)

# ==========================================================
#  FULL SAT SKILL LIST (34 SKILLS)
# ==========================================================
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

# ==========================================================
# PROMPT GENERATOR
# ==========================================================
def make_prompt(section: str, skill: str, difficulty: str):
    if section == "RW":
        return f"""
Bạn là chuyên gia đề thi SAT Reading & Writing.

Sinh 1 câu hỏi duy nhất theo đúng chuẩn Digital SAT:

Yêu cầu:
- Section: RW
- Skill: {skill}
- Difficulty: {difficulty}
- Passage 25–80 từ
- Một câu hỏi (content)
- Không sinh thêm đáp án
- Không sinh giải thích
- Format JSON duy nhất:

{{
  "section": "RW",
  "skill": "{skill}",
  "passage": "Đoạn văn 25–80 từ...",
  "content": "Câu hỏi ...?",
  "difficulty": "{difficulty}"
}}
"""
    else:
        return f"""
Bạn là chuyên gia đề thi SAT Math.

Sinh 1 câu hỏi toán theo chuẩn Digital SAT:

Yêu cầu:
- Section: Math
- Skill: {skill}
- Difficulty: {difficulty}
- Chỉ có content (không passage)
- Không sinh đáp án
- Không sinh lựa chọn
- Format JSON duy nhất:

{{
  "section": "Math",
  "skill": "{skill}",
  "content": "Câu hỏi toán ...",
  "difficulty": "{difficulty}"
}}
"""

# ==========================================================
# PARSE JSON SAFE
# ==========================================================
def to_json(text: str):
    text = text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(text)
    except:
        fixed = text.replace("\n", " ").replace("“", "\"").replace("”", "\"")
        return json.loads(fixed)

# ==========================================================
# GENERATE SINGLE ITEM
# ==========================================================
def generate_one(section: str, skill: str, difficulty: str):
    prompt = make_prompt(section, skill, difficulty)

    response = throttler.safe_openai_chat(
        client,
        model=model,
        messages=[
            {"role": "system", "content": "Bạn là AI tạo câu hỏi SAT chính xác theo chuẩn."},
            {"role": "user", "content": prompt}
        ]
    )

    raw = response.choices[0].message.content.strip()
    return to_json(raw)

# ==========================================================
# GENERATE FULL BANK
# ==========================================================
def generate_full_sat_bank(outfile="sat_questions.json", per_skill=10):
    difficulties = ["easy", "medium", "hard"]
    all_items = []

    for section, skills in SAT_SKILLS.items():
        for skill in skills:
            for diff in difficulties:
                for _ in range(per_skill):
                    print(f"🧠 Generating: {section} | {skill} | {diff}")
                    try:
                        q = generate_one(section, skill, diff)
                        all_items.append(q)
                    except Exception as e:
                        print("❌ Error:", e)

    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(all_items, f, indent=2, ensure_ascii=False)

    print(f"\n🎉 DONE! Generated {len(all_items)} SAT questions → {outfile}")


if __name__ == "__main__":
    generate_full_sat_bank("sat_questions.json", per_skill=5)
