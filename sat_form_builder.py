"""
SAT Form Builder
- Chọn câu hỏi từ ngân hàng theo IRT + Blueprint
- Hỗ trợ Math & Reading/Writing (RW)
- Cân bằng Difficulty + Skills
- Xáo đáp án, random Token-K
- Xuất Form JSON chuẩn SAT Digital
"""

import json
import math
import random
from collections import Counter
from typing import List, Dict, Tuple, Optional

# ============ IRT CORE ===========

D = 1.7  # logistic scale

def sigmoid_stable(x: float) -> float:
    if x >= 0: return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)

def prob_correct(theta: float, a: float, b: float, c: float) -> float:
    s = sigmoid_stable(D * a * (theta - b))
    return c + (1 - c) * s

def dprob_dtheta(theta: float, a: float, b: float, c: float) -> float:
    s = sigmoid_stable(D * a * (theta - b))
    return (1 - c) * D * a * s * (1 - s)

def fisher_info(theta: float, a: float, b: float, c: float) -> float:
    if a <= 0 or not (0 <= c < 1): return 0.0
    p = prob_correct(theta, a, b, c)
    if not (1e-6 < p < 1 - 1e-6): return 0.0
    dp = dprob_dtheta(theta, a, b, c)
    return (dp * dp) / (p * (1 - p))


# ============ ĐỘ KHÓ ===========

def bucket_difficulty(b: float) -> str:
    if b <= -1.0: return "easy"
    if b <= 0.5: return "medium"
    return "hard"


# ============ BLUEPRINT ===========

class SATBlueprint:
    def __init__(
        self,
        section: str,
        n_modules: int,
        q_per_module: int,
        diff_mix: Dict[str, float],
        skill_mix: Optional[Dict[str, float]],
        top_k: int = 3,
        theta_target: float = 0.0,
    ):
        assert section in ["Math", "RW"]
        assert n_modules >= 1 and q_per_module >= 1
        assert abs(sum(diff_mix.values()) - 1) < 1e-6, "Tổng diff_mix phải = 1.0"

        self.section = section
        self.n_modules = n_modules
        self.qpm = q_per_module
        self.diff_mix = diff_mix
        self.skill_mix = skill_mix
        self.top_k = top_k
        self.theta_target = theta_target


# ============ LOAD NGÂN HÀNG CÂU ===========

def load_bank(items_path: str, irt_path: str) -> List[Dict]:
    with open(items_path, encoding="utf-8") as f:
        items = json.load(f)
    with open(irt_path, encoding="utf-8") as f:
        params = {str(x["id"]): x for x in json.load(f)}

    bank = []
    for it in items:
        item_id = str(it["id"])
        if item_id not in params: continue
        par = params[item_id]
        entry = {
            **it,
            "_irt": {"a": par["a"], "b": par["b"], "c": par["c"]},
            "_difficulty": bucket_difficulty(par["b"]),
        }
        bank.append(entry)
    return bank


# ============ HỖ TRỢ ===========

def split_counts(total: int, mix: Dict[str, float]) -> Dict[str, int]:
    keys = list(mix.keys())
    base = {k: int(total * mix[k]) for k in keys}
    remain = total - sum(base.values())
    # phân dư theo phần thập phân lớn
    fracs = sorted(keys, key=lambda k: (total * mix[k]) - base[k], reverse=True)
    for k in fracs:
        if remain <= 0: break
        base[k] += 1
        remain -= 1
    return base

def filter_candidates(bank, asked_ids, diff, skill):
    return [
        it for it in bank
        if it["id"] not in asked_ids
        and it["_difficulty"] == diff
        and (skill is None or it.get("skill") == skill)
    ]

def pick_one(cands, theta, top_k):
    if not cands: return None
    score = []
    for it in cands:
        a, b, c = it["_irt"]["a"], it["_irt"]["b"], it["_irt"]["c"]
        info = fisher_info(theta, a, b, c)
        fit = 1 / (1 + abs(theta - b))
        score.append((info * fit, it))
    score.sort(key=lambda x: x[0], reverse=True)
    return random.choice([it for _, it in score[:top_k]])


# Xáo trộn đáp án nhưng giữ đúng answer_index
def shuffle_choices(it):
    new = dict(it)
    ch = list(it["choices"])
    correct = ch[it["answer_index"]]
    random.shuffle(ch)
    new["choices"] = ch
    new["answer_index"] = ch.index(correct)
    return new


# ============ BUILD MODULE ===========

def build_module(bank, asked_ids, bp: SATBlueprint):
    diff_count = split_counts(bp.qpm, bp.diff_mix)
    if bp.skill_mix:
        skill_count = split_counts(bp.qpm, bp.skill_mix)
    else:
        skill_count = {None: bp.qpm}  # không ép skill

    picked = []
    for skill, num_s in skill_count.items():
        local_diff = split_counts(num_s, bp.diff_mix)
        for diff, need in local_diff.items():
            for _ in range(need):
                cands = filter_candidates(bank, asked_ids, diff, skill)
                it = pick_one(cands, bp.theta_target, bp.top_k)
                if it:
                    asked_ids.add(it["id"])
                    picked.append(it)

    # bù thiếu
    while len(picked) < bp.qpm:
        remain = [it for it in bank if it["id"] not in asked_ids]
        if not remain: break
        it = pick_one(remain, bp.theta_target, bp.top_k)
        if it:
            asked_ids.add(it["id"])
            picked.append(it)

    # xáo thứ tự câu + lựa chọn
    picked = [shuffle_choices(it) for it in picked]
    random.shuffle(picked)

    # thống kê
    return picked, {
        "difficulty_counts": dict(Counter([it["_difficulty"] for it in picked])),
        "skill_counts": dict(Counter([it.get("skill") for it in picked])),
        "n_items": len(picked),
    }


# ============ BUILD FORM ===========

def build_sat_form(
    items_path: str,
    irt_path: str,
    section: str = "Math",
    n_modules: int = 2,
    q_per_module: int = 27,
    diff_mix: Dict[str, float] = None,
    skill_mix: Optional[Dict[str, float]] = None,
    seed: int = 2025,
    top_k: int = 3,
    theta_target: float = 0.0,
    out_path: Optional[str] = None
):
    random.seed(seed)
    bank = load_bank(items_path, irt_path)

    all_skills = sorted({it.get("skill") for it in bank})
    diff_mix = diff_mix or {"easy":0.25,"medium":0.5,"hard":0.25}

    bp = SATBlueprint(section, n_modules, q_per_module, diff_mix, skill_mix, top_k, theta_target)

    asked_ids = set()
    modules = []
    stats = []

    for i in range(n_modules):
        mod, st = build_module(bank, asked_ids, bp)
        modules.append({"module": i+1, "items": mod})
        stats.append({"module": i+1, **st})

    form = {
        "section": section,
        "n_modules": n_modules,
        "q_per_module": q_per_module,
        "diff_mix": diff_mix,
        "skill_mix": skill_mix,
        "seed": seed,
        "modules": modules,
        "stats": stats,
        "skills_available": all_skills,
    }

    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(form, f, ensure_ascii=False, indent=2)
        print(f"✅ Xuất Form: {out_path}")

    print("🎯 Stats:")
    for st in stats:
        print(f" - Module {st['module']}: n={st['n_items']} diff={st['difficulty_counts']} skills={st['skill_counts']}")

    return form
