# core/adaptive_selector.py

from __future__ import annotations

import math
from typing import Dict, List, Optional

from .schema import ItemV2, IRTParams
from .irt_engine import fisher_info
from .blueprint_policy import (
    BlueprintState,
    blueprint_ok,
    update_state_on_serve,
    should_stop_by_blueprint,
)


# ============================
# Thuật toán chọn câu
# ============================

def select_next_item(
    theta: float,
    items: List[ItemV2],
    irt_params: Dict[int, IRTParams],
    asked_ids: List[int],
    bp_state: Optional[BlueprintState] = None,
    exposure_limit: float = 0.30,
    debug: bool = False,
) -> Optional[ItemV2]:
    """
    Chọn câu tiếp theo theo Fisher Information + Blueprint + Exposure.

    Ưu tiên:
        1) Item chưa hỏi
        2) Item còn quota theo blueprint
        3) Item có información Fisher cao tại θ hiện tại
        4) Hạn chế chọn item có exposure vượt 'exposure_limit'

    Tham số:
        theta: ước lượng năng lực hiện tại
        items: ngân hàng câu hỏi
        irt_params: tham số IRT 3PL cho từng item
        asked_ids: danh sách id đã hỏi
        bp_state: trạng thái blueprint đang thực thi
        exposure_limit: ngưỡng cắt nếu item xuất hiện > 30%
    """
    if bp_state and should_stop_by_blueprint(bp_state):
        if debug:
            print("🔚 Blueprint đã đạt tổng số câu. Dừng chọn.")
        return None

    best_item: Optional[ItemV2] = None
    best_info: float = -1.0

    for item in items:
        if item.id in asked_ids:
            continue

        # Kiểm soát blueprint
        if bp_state:
            if not blueprint_ok(item.domain, item.skill, item.difficulty_tag, bp_state):
                continue

        # Kiểm tra exposure (nếu có tham số IRT lưu exposure)
        pars = irt_params.get(item.id)
        if not pars:
            continue
        if pars.exposure >= exposure_limit:
            continue

        # Tính Fisher info
        info = fisher_info(theta, pars)

        if info > best_info:
            best_info = info
            best_item = item

    # Nếu không tìm được item thỏa blueprint/exposure,
    # fallback: bỏ ràng buộc blueprint nhưng vẫn tránh đã hỏi
    if not best_item and bp_state:
        if debug:
            print("⚠️ Không tìm được câu theo blueprint. Fallback Fisher-only.")

        for item in items:
            if item.id in asked_ids:
                continue

            pars = irt_params.get(item.id)
            if not pars:
                continue

            info = fisher_info(theta, pars)
            if info > best_info:
                best_info = info
                best_item = item

    # Nếu vẫn không có, hết item trong ngân hàng
    if not best_item:
        if debug:
            print("❌ Hết item có thể chọn.")
        return None

    # Cập nhật blueprint (nếu có)
    if bp_state:
        update_state_on_serve(best_item.domain, best_item.skill, best_item.difficulty_tag, bp_state)

    # Cập nhật exposure (nếu có)
    pars = irt_params.get(best_item.id)
    if pars:
        pars.exposure = min(1.0, pars.exposure + 0.05)  # tăng nhẹ mỗi lần dùng

    if debug:
        print(f"✅ Chọn Item ID={best_item.id} Info={best_info:.4f} Domain={best_item.domain} Skill={best_item.skill}")

    return best_item
