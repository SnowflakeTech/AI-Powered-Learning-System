# tests/test_blueprint_policy.py

from core.blueprint_policy import (
    make_default_blueprint,
    initialize_state,
    remaining_quota,
    blueprint_ok,
    update_state_on_serve,
    should_stop_by_blueprint,
)


def test_blueprint_initial_targets():
    bp = make_default_blueprint(total_length=10)
    state = initialize_state(bp)

    remain = remaining_quota(state)

    # Tổng quota = tổng độ dài bài
    total = sum(remain[d]["total"] for d in remain)
    assert total == 10, f"Total quota phải bằng 10, đang là {total}"


def test_blueprint_ok_logic():
    bp = make_default_blueprint(total_length=4)
    state = initialize_state(bp)

    # Kiểm tra domain Math & skill Algebra easy
    assert blueprint_ok("Math", "Algebra", "easy", state)


def test_blueprint_update_state():
    bp = make_default_blueprint(total_length=4)
    state = initialize_state(bp)

    update_state_on_serve("Math", "Algebra", "easy", state)
    remain = remaining_quota(state)

    # Sau khi phục vụ 1 item => quota domain Math giảm 1
    assert remain["Math"]["total"] == max(0, bp.domains["Math"].weight * 4 - 1)


def test_blueprint_should_stop():
    bp = make_default_blueprint(total_length=2)
    state = initialize_state(bp)

    update_state_on_serve("Math", "Algebra", "easy", state)
    update_state_on_serve("R&W", "Grammar", "easy", state)

    assert should_stop_by_blueprint(state)
