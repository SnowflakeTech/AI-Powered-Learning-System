# core/blueprint_policy.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Tuple, List
import math


# ============================
# Kiểu dữ liệu blueprint
# ============================

@dataclass(frozen=True)
class DifficultyMix:
    """Tỉ lệ độ khó trong mỗi domain/skill."""
    easy: float
    medium: float
    hard: float


@dataclass(frozen=True)
class SkillSpec:
    """Đặc tả cho một kỹ năng: tỉ lệ target trong domain và mix độ khó."""
    weight: float                      # tỉ lệ của skill trong domain (sum=1 trong domain)
    difficulty: DifficultyMix          # tỉ lệ easy/medium/hard trong skill (sum=1)


@dataclass(frozen=True)
class DomainSpec:
    """Đặc tả cho một domain: tỉ lệ target trong bài và cấu hình từng skill."""
    weight: float                      # tỉ lệ domain trong bài (sum=1 toàn bài)
    skills: Mapping[str, SkillSpec]    # ví dụ: {"Algebra": SkillSpec(...), ...}


@dataclass(frozen=True)
class BlueprintSpec:
    """
    Blueprint tổng thể của bài thi.
    - domains: khai báo domain và skill kèm tỉ lệ.
    - length: độ dài mục tiêu (số câu).
    """
    domains: Mapping[str, DomainSpec]
    length: int


# ============================
# Trạng thái blueprint runtime
# ============================

@dataclass
class BucketCounter:
    """Bộ đếm cho 3 mức độ khó trong một skill."""
    easy: int = 0
    medium: int = 0
    hard: int = 0

    def total(self) -> int:
        return self.easy + self.medium + self.hard


@dataclass
class SkillCounter:
    """Bộ đếm theo skill: tổng và chia theo độ khó."""
    total: int = 0
    by_diff: BucketCounter = field(default_factory=BucketCounter)


@dataclass
class DomainCounter:
    """Bộ đếm theo domain: tổng và chia theo skill."""
    total: int = 0
    by_skill: Dict[str, SkillCounter] = field(default_factory=dict)


@dataclass
class BlueprintState:
    """
    Trạng thái tiêu thụ blueprint trong một phiên thi.
    Sử dụng cùng BlueprintSpec để quyết định còn quota cho domain/skill/difficulty nào.
    """
    spec: BlueprintSpec
    served: Dict[str, DomainCounter] = field(default_factory=dict)

    def total_served(self) -> int:
        return sum(dc.total for dc in self.served.values())


# ============================
# Các hàm tiện ích
# ============================

def _safe_norm(weights: Dict[str, float]) -> Dict[str, float]:
    """Chuẩn hóa từ điển trọng số về tổng 1. Nếu tổng bằng 0, chia đều."""
    s = sum(max(0.0, v) for v in weights.values())
    if s <= 0:
        n = len(weights) or 1
        return {k: 1.0 / n for k in weights}
    return {k: max(0.0, v) / s for k, v in weights.items()}


def _clip_nonneg(x: float) -> int:
    """Làm tròn xuống và không âm."""
    return max(0, int(math.floor(x)))


def _targets_for_domain(spec: BlueprintSpec, domain_name: str) -> Tuple[int, Dict[str, int], Dict[str, Dict[str, int]]]:
    """
    Tính chỉ tiêu:
    - domain_target
    - skill_target[skill]
    - diff_target[skill][difficulty]
    """
    dspec = spec.domains[domain_name]
    L = spec.length

    # Chỉ tiêu cho domain
    domain_target = _clip_nonneg(L * max(0.0, dspec.weight))

    # Chỉ tiêu skill trong domain
    skill_weights = {s: dspec.skills[s].weight for s in dspec.skills}
    skill_weights = _safe_norm(skill_weights)

    skill_target: Dict[str, int] = {s: _clip_nonneg(domain_target * skill_weights[s]) for s in dspec.skills}

    # Chỉ tiêu difficulty trong mỗi skill
    diff_target: Dict[str, Dict[str, int]] = {}
    for s, ss in dspec.skills.items():
        mix = _safe_norm({"easy": ss.difficulty.easy, "medium": ss.difficulty.medium, "hard": ss.difficulty.hard})
        t = skill_target[s]
        diff_target[s] = {
            "easy": _clip_nonneg(t * mix["easy"]),
            "medium": _clip_nonneg(t * mix["medium"]),
            "hard": _clip_nonneg(t * mix["hard"]),
        }

    return domain_target, skill_target, diff_target


def compute_all_targets(spec: BlueprintSpec) -> Dict[str, Dict]:
    """
    Trả về cấu trúc:
    {
      domain: {
        "total": int,
        "by_skill": {
          skill: {
            "total": int,
            "by_diff": {"easy": int, "medium": int, "hard": int}
          }
        }
      }
    }
    """
    out: Dict[str, Dict] = {}
    for d in spec.domains:
        dt, st_map, diff_map = _targets_for_domain(spec, d)
        out[d] = {
            "total": dt,
            "by_skill": {
                s: {
                    "total": st_map[s],
                    "by_diff": diff_map[s]
                } for s in st_map
            }
        }
    # Phần dư do làm tròn: phân bổ sau ở adaptive_selector nếu cần.
    return out


def initialize_state(spec: BlueprintSpec) -> BlueprintState:
    """Khởi tạo BlueprintState với các bộ đếm rỗng."""
    state = BlueprintState(spec=spec, served={})
    for dname, dspec in spec.domains.items():
        dc = DomainCounter(total=0, by_skill={})
        for sname in dspec.skills.keys():
            dc.by_skill[sname] = SkillCounter(total=0, by_diff=BucketCounter())
        state.served[dname] = dc
    return state


# ============================
# API chính dùng trong chọn câu
# ============================

def remaining_quota(state: BlueprintState) -> Dict[str, Dict]:
    """
    Tính quota còn lại theo domain/skill/difficulty.
    Kết hợp targets (tính từ spec) với served (đã phát).
    """
    targets = compute_all_targets(state.spec)
    remain: Dict[str, Dict] = {}

    for dname, dtarget in targets.items():
        dc = state.served.get(dname)
        if not dc:
            continue
        dt_left = max(0, dtarget["total"] - dc.total)

        skill_left: Dict[str, Dict] = {}
        for sname, st_spec in dtarget["by_skill"].items():
            sc = dc.by_skill.get(sname)
            if not sc:
                continue

            st_total_left = max(0, st_spec["total"] - sc.total)

            bd = st_spec["by_diff"]
            sd = sc.by_diff
            diff_left = {
                "easy": max(0, bd["easy"] - sd.easy),
                "medium": max(0, bd["medium"] - sd.medium),
                "hard": max(0, bd["hard"] - sd.hard),
            }
            skill_left[sname] = {"total": st_total_left, "by_diff": diff_left}

        remain[dname] = {"total": dt_left, "by_skill": skill_left}
    return remain


def blueprint_ok(
    item_domain: str,
    item_skill: str,
    item_difficulty: str,
    state: BlueprintState
) -> bool:
    """
    Kiểm tra item có nên được phục vụ dựa trên quota còn lại không.
    Trả True nếu domain/skill/difficulty tương ứng còn quota.
    """
    rem = remaining_quota(state)

    d = rem.get(item_domain)
    if not d or d["total"] <= 0:
        return False

    s = d["by_skill"].get(item_skill)
    if not s or s["total"] <= 0:
        return False

    if item_difficulty not in ("easy", "medium", "hard"):
        return False

    if s["by_diff"].get(item_difficulty, 0) <= 0:
        return False

    return True


def update_state_on_serve(
    item_domain: str,
    item_skill: str,
    item_difficulty: str,
    state: BlueprintState
) -> None:
    """
    Cập nhật bộ đếm khi một item đã được phục vụ cho thí sinh.
    Gọi hàm này ngay sau khi quyết định chọn item.
    """
    dc = state.served[item_domain]
    sc = dc.by_skill[item_skill]

    # Cộng bộ đếm
    dc.total += 1
    sc.total += 1
    if item_difficulty == "easy":
        sc.by_diff.easy += 1
    elif item_difficulty == "medium":
        sc.by_diff.medium += 1
    else:
        sc.by_diff.hard += 1


def should_stop_by_blueprint(state: BlueprintState) -> bool:
    """
    Kiểm tra đã đạt đủ tổng độ dài bài theo blueprint hay chưa.
    Nếu tổng phát ra >= spec.length thì dừng theo blueprint.
    """
    return state.total_served() >= state.spec.length


# ============================
# Blueprint mặc định (ví dụ)
# ============================

def make_default_blueprint(total_length: int = 52) -> BlueprintSpec:
    """
    Blueprint ví dụ bám theo tinh thần SAT Digital (có thể chỉnh sửa).
    - Math 50%, R&W 50%.
    - Bên trong mỗi domain phân bổ skill và difficulty mix minh họa.
    """
    math_skills = {
        "Algebra": SkillSpec(
            weight=0.35,
            difficulty=DifficultyMix(easy=0.35, medium=0.50, hard=0.15)
        ),
        "Advanced Math": SkillSpec(
            weight=0.35,
            difficulty=DifficultyMix(easy=0.25, medium=0.55, hard=0.20)
        ),
        "Problem Solving": SkillSpec(
            weight=0.30,
            difficulty=DifficultyMix(easy=0.40, medium=0.45, hard=0.15)
        ),
    }

    rw_skills = {
        "Vocabulary": SkillSpec(
            weight=0.25,
            difficulty=DifficultyMix(easy=0.45, medium=0.45, hard=0.10)
        ),
        "Rhetoric": SkillSpec(
            weight=0.40,
            difficulty=DifficultyMix(easy=0.30, medium=0.50, hard=0.20)
        ),
        "Grammar": SkillSpec(
            weight=0.35,
            difficulty=DifficultyMix(easy=0.40, medium=0.45, hard=0.15)
        ),
    }

    domains = {
        "Math": DomainSpec(weight=0.50, skills=math_skills),
        "R&W": DomainSpec(weight=0.50, skills=rw_skills),
    }

    return BlueprintSpec(domains=domains, length=total_length)
