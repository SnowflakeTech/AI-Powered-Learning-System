# core/__init__.py

"""
Core module for SAT-AI System v2

Bao gồm:
- Mô hình IRT 3PL và thuật toán cập nhật theta
- Cơ chế chọn câu thích ứng theo Fisher Information
- Blueprint kiểm soát phân phối nội dung (domain/skill/difficulty)
- Schema chuẩn cho Item và IRTParams

Các thành phần xuất khẩu phổ biến:
    ItemV2, IRTParams, TestState
    prob_correct, fisher_info, update_theta_map_once
    make_default_blueprint, initialize_state, blueprint_ok
    select_next_item
"""

# Schema models
from .schema import (
    ItemV2,
    IRTParams,
    TestState,
)

# IRT computation & scoring
from .irt_engine import (
    prob_correct,
    fisher_info,
    update_theta_map_once,
)

# Blueprint spec & runtime balancing
from .blueprint_policy import (
    make_default_blueprint,
    initialize_state,
    blueprint_ok,
    remaining_quota,
    should_stop_by_blueprint,
)

# Adaptive selection algorithm
from .adaptive_selector import (
    select_next_item,
)


__all__ = [
    # Schema
    "ItemV2",
    "IRTParams",
    "TestState",

    # IRT
    "prob_correct",
    "fisher_info",
    "update_theta_map_once",

    # Blueprint
    "make_default_blueprint",
    "initialize_state",
    "blueprint_ok",
    "remaining_quota",
    "should_stop_by_blueprint",

    # Adaptive selector
    "select_next_item",
]
