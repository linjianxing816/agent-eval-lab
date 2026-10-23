"""D5 任务校验器：12 项检查，一次报全。

重点验三件事：
1. `refunds[-1].amount` 在空列表上不误报（靠 schema 不靠实例）
2. 拼错的字段（`refunds[-1].amout`）能查出来
3. 一次能报多条，不是遇到第一个就停
"""
from __future__ import annotations

import json

from evalkit.env.state import DEFAULT_INITIAL_STATE
from evalkit.tasks import (
    MISSING,
    STATE_SCHEMA,
    check_file,
    resolve_with_schema,
    validate_task,
)

GOOD = {
    "id": "task-001",
    "instruction": "用户 U1001 要求退订单 A123，请按规则处理",
    "tools_allowed": ["get_user", "get_order", "calc_refund", "cancel_shipment", "submit_refund"],
    "success_criteria": [
        {"path": "orders.A123.status", "expect": "refunded"},
        {"path": "refunds[-1].amount", "expect": 299.0},
    ],
    "constraints": {"max_steps": 12, "forbid_tools": ["delete_order"]},
}


def _codes(task, line=1, seen=None):
    return [i.code for i in validate_task(task, line, {} if seen is None else seen)]


def test_good_task_has_no_issues():
    assert _codes(GOOD) == []


def test_empty_refunds_path_is_not_a_false_positive():
    # 初始状态 refunds 是空列表，但这条路径合法；值是 MISSING（运行时才有）
    error, actual = resolve_with_schema("refunds[-1].amount", DEFAULT_INITIAL_STATE)
    assert error is None
    assert actual is MISSING


def test_misspelled_field_is_caught_on_empty_list():
    error, _ = resolve_with_schema("refunds[-1].amout", DEFAULT_INITIAL_STATE)
    assert error is not None and "amout" in error


def test_unknown_order_id_is_caught():
    error, _ = resolve_with_schema("orders.A999.status", DEFAULT_INITIAL_STATE)
    assert error is not None and "A999" in error


def test_schema_matches_real_state():
    # schema 和 DEFAULT_INITIAL_STATE 不能漂移
    assert set(STATE_SCHEMA) == set(DEFAULT_INITIAL_STATE)
    for key in ("name", "vip"):
        assert key in STATE_SCHEMA["users"]["*"]
    for key in DEFAULT_INITIAL_STATE["orders"]["A123"]:
        assert key in STATE_SCHEMA["orders"]["*"]
    for key in DEFAULT_INITIAL_STATE["policy"]:
        assert key in STATE_SCHEMA["policy"]
    refund = {"order_id": "A123", "amount": 299.0, "ts": "2026-01-01T00:00:00Z"}
    assert set(refund) == set(STATE_SCHEMA["refunds"][0])


def test_missing_required_field():
    task = dict(GOOD)
    task.pop("success_criteria")
    assert _codes(task) == ["E01"]


def test_duplicate_id():
    seen = {"task-001": 3}
    assert "E02" in _codes(GOOD, line=7, seen=seen)


def test_empty_criteria():
    task = dict(GOOD, success_criteria=[])
    assert _codes(task) == ["E04"]


def test_unknown_tool_and_overlap():
    task = dict(GOOD, tools_allowed=["get_order", "drop_table"], constraints={"forbid_tools": ["get_order"]})
    codes = _codes(task)
    assert "E05" in codes and "E10" in codes


def test_empty_tools_allowed_is_warning_not_error():
    task = dict(GOOD, tools_allowed=[])
    issues = validate_task(task, 1, {})
    assert [i.code for i in issues] == ["E11"]
    assert issues[0].level == "warning"


def test_bad_max_steps():
    task = dict(GOOD, constraints={"max_steps": 0})
    assert "E07" in _codes(task)


def test_instruction_empty_and_too_long():
    assert _codes(dict(GOOD, instruction="   ")) == ["E12"]
    assert _codes(dict(GOOD, instruction="x" * 501)) == ["E12"]


def test_int_expectation_is_warning():
    # 空列表时拿不到实例值，E09 不触发；换成订单金额才比得出类型
    task = dict(GOOD, success_criteria=[{"path": "orders.A123.amount", "expect": 299}])
    issues = validate_task(task, 1, {})
    assert [i.code for i in issues] == ["E09"]
    assert issues[0].level == "warning"


def test_string_expectation_is_error():
    # 手滑加引号："299" 和 299.0 永远不相等，这条任务写下去就是死的
    task = dict(GOOD, success_criteria=[{"path": "orders.A123.amount", "expect": "299"}])
    issues = validate_task(task, 1, {})
    assert [i.code for i in issues] == ["E13"]
    assert issues[0].level == "error"
    assert "永远失败" in issues[0].message


def test_check_file_reports_many_issues_at_once(tmp_path):
    rows = [
        json.dumps(GOOD, ensure_ascii=False),
        "{不是 JSON",
        json.dumps(dict(GOOD, id="task-002", success_criteria=[]), ensure_ascii=False),
        json.dumps(dict(GOOD, id="task-001"), ensure_ascii=False),          # id 重复
        json.dumps(dict(GOOD, id="task-004",
                        success_criteria=[{"path": "orders.A999.status", "expect": "refunded"}]),
                   ensure_ascii=False),
    ]
    path = tmp_path / "broken.jsonl"
    path.write_text("\n".join(rows), encoding="utf-8")
    tasks, issues = check_file(path)
    codes = [i.code for i in issues]
    assert len(tasks) == 4                      # 坏 JSON 那行不算任务
    assert codes == ["E08", "E04", "E02", "E03"]
    assert {i.line for i in issues} == {2, 3, 4, 5}


def test_check_file_missing_file(tmp_path):
    tasks, issues = check_file(tmp_path / "nope.jsonl")
    assert tasks == [] and issues[0].code == "E08"
