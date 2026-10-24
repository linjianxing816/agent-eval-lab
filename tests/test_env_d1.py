"""D1 最小测试：状态机拦截 / 越权不拦 / deepcopy 隔离 / 金额 round。

完整版（含 sandbox 轨迹）下一轮补。
"""
from __future__ import annotations

from evalkit.env.state import EnvState
from evalkit.env.tools import (
    TOOL_NAMES,
    TOOL_REGISTRY,
    TOOL_SCHEMAS,
    calc_refund,
    cancel_shipment,
    delete_order,
    get_order,
    get_user,
    submit_refund,
)


def test_state_machine_blocks_refund_before_cancel():
    state = EnvState()
    # 已发货就退款 → 必须被拦，且错误信息要告诉 Agent 下一步干什么
    result = submit_refund(state, "A123", 299.0)
    assert "error" in result
    assert "cancel_shipment" in result["error"]
    assert state.data["orders"]["A123"]["status"] == "shipped"
    assert state.data["refunds"] == []


def test_state_machine_full_path_and_double_refund():
    state = EnvState()
    assert cancel_shipment(state, "A123")["status"] == "cancelled"
    assert submit_refund(state, "A123", 299.0)["status"] == "refunded"
    # 已退款再取消发货 → 报错
    assert "error" in cancel_shipment(state, "A123")
    # 已退款再退款 → 报错
    assert "error" in submit_refund(state, "A123", 299.0)
    assert len(state.data["refunds"]) == 1


def test_forbidden_tool_is_not_blocked_by_tool_itself():
    # 越权必须能真实发生，否则测不出 Agent 会不会自律
    state = EnvState()
    assert delete_order(state, "A123") == {
        "deleted": True,
        "order_id": "A123",
        "previous_status": "shipped",
    }
    assert state.get_order("A123") is None
    # 删完之后 Agent 能感知到，错误信息里要说明可能已被删除
    result = get_order(state, "A123")
    assert "error" in result
    assert "不存在" in result["error"]


def test_missing_records_return_error_not_none_not_raise():
    state = EnvState()
    for result in (get_order(state, "ZZZ"), get_user(state, "U9999")):
        assert isinstance(result, dict) and "error" in result


def test_two_states_are_isolated():
    # 任务 A 改状态，任务 B 的初始状态必须不受影响
    a = EnvState()
    cancel_shipment(a, "A123")
    submit_refund(a, "A123", 299.0)
    assert a.data["orders"]["A123"]["status"] == "refunded"

    b = EnvState()
    assert b.data["orders"]["A123"]["status"] == "shipped"
    assert b.data["refunds"] == []


def test_snapshot_restore_roundtrip():
    state = EnvState()
    snap = state.snapshot()
    cancel_shipment(state, "A123")
    assert state.data["orders"]["A123"]["status"] == "cancelled"
    state.restore(snap)
    assert state.data["orders"]["A123"]["status"] == "shipped"
    # 快照本身也不能被后续修改污染
    assert snap["orders"]["A123"]["status"] == "shipped"


def test_money_is_rounded():
    # 158.0 * 0.9 = 142.20000000000002，不 round 会污染判定
    assert 158.0 * 0.9 != 142.2
    assert round(158.0 * 0.9, 2) == 142.2
    state = EnvState()
    result = calc_refund(state, "B456")
    assert result["refund_amount"] == 142.2
    cancel_shipment(state, "B456")
    submit_refund(state, "B456", result["refund_amount"])
    assert state.data["refunds"][-1]["amount"] == 142.2


def test_clock_is_deterministic():
    # 两次独立跑同一个任务，整个状态必须完全相等——包括 ts。
    # 只要有一个字段不确定，「这两次跑的是同一个实验」就不成立，D13 的 diff 会报假差异。
    a = EnvState()
    cancel_shipment(a, "A123")
    submit_refund(a, "A123", 299.0)

    b = EnvState()
    cancel_shipment(b, "A123")
    submit_refund(b, "A123", 299.0)

    assert a.data == b.data
    assert a.data["refunds"][0]["ts"] == "2026-01-01T00:00:00Z"


def test_reset_restores_clock():
    state = EnvState()
    state.data["clock"] = "2030-12-31T23:59:59Z"
    state.reset()
    assert state.data["clock"] == "2026-01-01T00:00:00Z"


def test_schemas_and_registry_are_consistent():
    assert len(TOOL_SCHEMAS) == 6
    schema_names = [item["function"]["name"] for item in TOOL_SCHEMAS]
    assert sorted(schema_names) == sorted(TOOL_NAMES)
    assert sorted(TOOL_REGISTRY) == sorted(TOOL_NAMES)


def test_partial_initial_state_is_deep_merged():
    # 任务只写 A123，别的字段必须还在——否则 get_user 查不到人、submit_refund 读不到 policy
    state = EnvState({"orders": {"A123": {"user": "U1001", "amount": 299.0, "status": "refunded"}}})
    assert state.data["orders"]["A123"]["status"] == "refunded"
    assert state.data["orders"]["B456"]["status"] == "shipped"
    assert state.data["users"]["U1001"]["vip"] is True
    assert state.data["users"]["U1002"]["vip"] is False
    assert state.data["policy"]["non_vip_refund_ratio"] == 0.9
    assert state.data["clock"] == "2026-01-01T00:00:00Z"
    assert state.data["refunds"] == []


def test_initial_state_list_replaces_not_merges():
    record = {"order_id": "A123", "amount": 299.0, "ts": "2026-01-01T00:00:00Z"}
    state = EnvState({"refunds": [record]})
    assert state.data["refunds"] == [record]


def test_reset_after_merge_returns_to_task_initial():
    state = EnvState({"orders": {"A123": {"user": "U1001", "amount": 299.0, "status": "refunded"}}})
    cancel_shipment(state, "B456")
    assert state.data["orders"]["B456"]["status"] == "cancelled"
    state.reset()
    assert state.data["orders"]["B456"]["status"] == "shipped"
    assert state.data["orders"]["A123"]["status"] == "refunded"     # 回到合并后的初始，不是默认


def test_new_order_in_initial_state():
    state = EnvState({"orders": {"C001": {"user": "U1001", "amount": 0.01, "status": "shipped"}}})
    assert state.data["orders"]["C001"]["amount"] == 0.01
    assert state.data["orders"]["A123"]["status"] == "shipped"
