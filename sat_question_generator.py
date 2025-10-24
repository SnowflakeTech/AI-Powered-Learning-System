import json
import math
import random
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict

# ============================
# Data structures
# ============================
@dataclass
class Item:
    id: int
    question: str
    choices: List[str]
    answer_index: int
    skill: str

@dataclass
class IRT:
    id: int
    a: float  # discrimination
    b: float  # difficulty
    c: float  # guessing

# ============================
# Utility helpers
# ============================

def _unique_choices(correct: int, distractors: List[int]) -> List[int]:
    """Return up to 3 unique distractors not equal to correct."""
    seen = set([correct])
    uniq = []
    for d in distractors:
        if d not in seen:
            uniq.append(d)
            seen.add(d)
        if len(uniq) == 3:
            break
    # If still short, pad with nearby numbers
    t = correct
    k = 1
    while len(uniq) < 3:
        for cand in (t + k, t - k):
            if cand not in seen:
                uniq.append(cand)
                seen.add(cand)
                if len(uniq) == 3:
                    break
        k += 1
    return uniq


def _shuffle_choices(correct_value: int, distractor_values: List[int]) -> Tuple[List[str], int]:
    pool = [correct_value] + distractor_values
    random.shuffle(pool)
    answer_index = pool.index(correct_value)
    return list(map(lambda x: str(x), pool)), answer_index


def _irt_for_difficulty(item_id: int, level: str) -> IRT:
    """Map a friendly difficulty label to rough 3PL parameters.
    level in {"easy","medium","hard"}
    """
    if level == "easy":
        a, b, c = round(random.uniform(0.8, 1.5), 2), round(random.uniform(-1.5, -0.3), 2), 0.25
    elif level == "hard":
        a, b, c = round(random.uniform(0.9, 1.8), 2), round(random.uniform(0.6, 1.8), 2), 0.2
    else:  # medium
        a, b, c = round(random.uniform(0.8, 1.6), 2), round(random.uniform(-0.3, 0.6), 2), 0.2
    return IRT(id=item_id, a=a, b=b, c=c)

# ============================
# Generators (Math SAT-style)
# ============================

def gen_linear_equation(start_id: int) -> Tuple[Item, IRT]:
    # ax + b = c
    a = random.randint(2, 9)
    x_true = random.randint(-6, 12)
    b = random.randint(-12, 12)
    c = a * x_true + b
    q = f"Nếu {a}x {'+' if b >= 0 else '-'} {abs(b)} = {c} thì x = ?"
    correct = x_true
    distractors = _unique_choices(correct, [x_true + a, x_true - a, int((c - b) / (a + 1) if a != -1 else x_true + 1)])
    choices, ans_idx = _shuffle_choices(correct, distractors)
    item = Item(id=start_id, question=q, choices=choices, answer_index=ans_idx, skill="Algebra: Linear Equations")
    return item, _irt_for_difficulty(start_id, "easy")


def gen_quadratic_value(start_id: int) -> Tuple[Item, IRT]:
    # f(x) = x^2 + px + q ; evaluate f(k)
    p = random.randint(-6, 6)
    q = random.randint(-8, 8)
    k = random.randint(-5, 7)
    correct = k * k + p * k + q
    question = f"Cho f(x) = x² {'+' if p >= 0 else '-'} {abs(p)}x {'+' if q >= 0 else '-'} {abs(q)}. Tính f({k})."
    distractors = _unique_choices(correct, [k * k + (p + 1) * k + q, k * k + p * (k + 1) + q, k * k + p * k + (q + 1)])
    choices, ans_idx = _shuffle_choices(correct, distractors)
    item = Item(id=start_id, question=question, choices=choices, answer_index=ans_idx, skill="Functions: Quadratic Evaluation")
    return item, _irt_for_difficulty(start_id, "medium")


def gen_percent_change(start_id: int) -> Tuple[Item, IRT]:
    # increase/decrease percent
    base = random.randint(40, 300)
    pct = random.choice([5, 8, 10, 12, 15, 20, 25])
    up = random.choice([True, False])
    new_val = round(base * (1 + pct / 100) if up else base * (1 - pct / 100))
    question = (
        f"Một giá trị {('tăng' if up else 'giảm')} {pct}% từ {base}. Giá trị mới xấp xỉ bằng bao nhiêu?"
    )
    correct = new_val
    distractors = _unique_choices(correct, [round(base * (1 + (pct + 5) / 100)), round(base * (1 + (pct - 5) / 100)), base])
    choices, ans_idx = _shuffle_choices(correct, distractors)
    item = Item(id=start_id, question=question, choices=choices, answer_index=ans_idx, skill="Ratios & Percentage")
    return item, _irt_for_difficulty(start_id, "medium")


def gen_systems_of_equations(start_id: int) -> Tuple[Item, IRT]:
    # Solve simple system: {x + y = s ; x - y = d}
    x = random.randint(-6, 10)
    y = random.randint(-6, 10)
    s, d = x + y, x - y
    question = f"Cho hệ: x + y = {s} và x − y = {d}. Giá trị của x là?"
    correct = (s + d) // 2
    distractors = _unique_choices(correct, [s - d, (s - d) // 2, (s + d)])
    choices, ans_idx = _shuffle_choices(correct, distractors)
    item = Item(id=start_id, question=question, choices=choices, answer_index=ans_idx, skill="Algebra: Systems of Equations")
    return item, _irt_for_difficulty(start_id, "hard")


def gen_ratio_word_problem(start_id: int) -> Tuple[Item, IRT]:
    a = random.randint(2, 7)
    b = random.randint(3, 9)
    total = random.randint(20, 90)
    # Suppose a : b; find part for 'a' given total
    part_a = round(total * a / (a + b))
    question = f"Tỉ lệ A:B là {a}:{b}. Nếu tổng là {total}, thì phần của A bằng?"
    correct = part_a
    distractors = _unique_choices(correct, [total - part_a, part_a + a, part_a - b])
    choices, ans_idx = _shuffle_choices(correct, distractors)
    item = Item(id=start_id, question=question, choices=choices, answer_index=ans_idx, skill="Ratios & Proportions")
    return item, _irt_for_difficulty(start_id, "easy")

# ============================
# Batch generation API
# ============================

GENS = [
    gen_linear_equation,
    gen_quadratic_value,
    gen_percent_change,
    gen_systems_of_equations,
    gen_ratio_word_problem,
]


def generate_sat_items(n: int, start_id: int = 1, seed: int = None) -> Tuple[List[Item], List[IRT]]:
    """Generate n SAT-like math items and corresponding rough IRT params.
    Returns (items, irt_params) ready to dump to items.json and irt_params.json
    """
    if seed is not None:
        random.seed(seed)

    items_out: List[Item] = []
    irt_out: List[IRT] = []

    for k in range(n):
        gen = random.choice(GENS)
        item, irt = gen(start_id + k)
        items_out.append(item)
        irt_out.append(irt)

    return items_out, irt_out


def save_as_json(items: List[Item], irt_params: List[IRT], items_path: str = "data/items.json", irt_path: str = "data/irt_params.json"):
    items_payload = [asdict(i) for i in items]
    irt_payload = [asdict(i) for i in irt_params]
    with open(items_path, "w", encoding="utf-8") as f:
        json.dump(items_payload, f, ensure_ascii=False, indent=4)
    with open(irt_path, "w", encoding="utf-8") as f:
        json.dump(irt_payload, f, ensure_ascii=False, indent=4)


# ============================
# Demo run
# ============================
if __name__ == "__main__":
    N = 10
    items, irts = generate_sat_items(N, start_id=1, seed=42)
    print(f"Generated {len(items)} items.")
    # Preview first two
    for it in items[:2]:
        print("\n--- ITEM ---")
        print(asdict(it))
    # Save to data/
    save_as_json(items, irts, items_path="data/items_generated.json", irt_path="data/irt_params_generated.json")
    print("\n✅ Saved to data/items_generated.json and data/irt_params_generated.json")
