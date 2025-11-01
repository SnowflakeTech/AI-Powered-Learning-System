# core/schema.py

from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class ItemV2:
    """
    Cấu trúc Item chuẩn SAT Digital:
    - Hỗ trợ domain Math & R&W
    - Có difficulty_tag để hỗ trợ blueprint
    - options có id ("A","B","C","D") để UI mapping ổn định
    """
    id: int
    domain: str  # Math | R&W
    skill: str   # Ví dụ: Algebra, Vocabulary, Rhetoric...
    difficulty_tag: str  # "easy" | "medium" | "hard"

    stem: str  # Nội dung câu hỏi chính
    options: List[Dict[str, str]]  # [{id: "A", text: "..."}]
    answer_key: str  # "A"/"B"/"C"/"D"
    
    # Optional fields
    stimulus: Optional[str] = None  # Đoạn văn hỗ trợ nếu có (R&W)
    rationale: Optional[Dict[str, str]] = None  # Giải thích từng đáp án

    # Metadata phục vụ quản trị item
    version: int = 1
    status: str = "active"  # active | trial | retired


@dataclass
class IRTParams:
    """
    Tham số 3PL phục vụ adaptive CAT:
    - a: discrimination
    - b: difficulty
    - c: guessing (lower asymptote)
    """
    id: int
    a: float
    b: float
    c: float

    # Trường bổ sung cho giám sát thống kê
    exposure: float = 0.0  # tỷ lệ xuất hiện trong CAT
    drift_flag: bool = False  # kiểm định item drift theo thời gian


@dataclass
class TestState:
    """
    Đại diện trạng thái một phiên thi adaptive:
    """
    theta: float = 0.0
    se: float = float("inf")
    asked: List[int] = field(default_factory=list)
    answered: List[Dict] = field(default_factory=list)  # [{id:..., correct:...}]
