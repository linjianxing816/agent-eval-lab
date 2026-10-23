"""D1 sandbox 测试：轨迹记录 / 两种 error 分开 / 失败也留痕 / 步数口径。

核心是那条最容易写错的：**业务失败不能写进 TrajectoryStep.error**，
否则 D3 会把「模型操作不对」误判成「环境不稳定」。
"""
from __future__ import annotations

import pytest

from evalkit.env.sandbox import Sandbox
from evalkit.env.state import EnvState
from evalkit.env.tools import TOOL_REGISTRY


def test_success_records_trajectory():
    sb = Sandbox(EnvState())
    result = sb.call("get_order", {"order_id": "A123"}, step=1)
    assert result["status"] == "shipped"
    step = sb.trajectory()[0]
    assert step.step == 1 and step.tool == "get_order"
    assert step.error is None
    assert step.elapsed_ms > 0


def test_business_failure_has_result_error_but_no_trajectory_error():
    sb = Sandbox(EnvState())
    result = sb.call("submit_refund", {"order_id": "A123", "amount": 299.0}, step=1)
    assert "error" in result              # 业务失败：模型操作不对
    step = sb.trajectory()[0]
    assert step.error is None             # 但**不是**基建失败
    assert "cancel_shipment" in result["error"]


def test_tool_exception_sets_trajectory_error(monkeypatch):
    def boom(state, order_id):
        raise RuntimeError("环境炸了")

    monkeypatch.setitem(TOOL_REGISTRY, "boom", boom)
    sb = Sandbox(EnvState())
    result = sb.call("boom", {"order_id": "A123"}, step=1)
    assert "error" in result
    step = sb.trajectory()[0]
    assert step.error is not None and "RuntimeError" in step.error


def test_unknown_tool_is_recorded_not_raised():
    sb = Sandbox(EnvState())
    result = sb.call("drop_table", {}, step=1)
    assert "不存在" in result["error"]
    assert sb.tool_calls_total == 1
    assert sb.trajectory()[0].error is None


def test_missing_argument_message_is_human_readable():
    sb = Sandbox(EnvState())
    result = sb.call("get_order", {}, step=1)
    assert "order_id" in result["error"]
    assert "缺少必填参数" in result["error"]


def test_extra_argument_is_reported():
    sb = Sandbox(EnvState())
    result = sb.call("get_order", {"order_id": "A123", "user_id": "U1001"}, step=1)
    assert "user_id" in result["error"]
    assert "不接受参数" in result["error"]


def test_args_not_dict_does_not_crash():
    # D2 可能把 json.loads 失败的字符串/None 直接传进来
    sb = Sandbox(EnvState())
    for bad in ("A123", None):
        result = sb.call("get_order", bad, step=1)
        assert "参数必须是 JSON 对象" in result["error"]
    assert sb.tool_calls_total == 2
    assert all(step.error is None for step in sb.trajectory())


def test_parallel_calls_share_one_step():
    # 一次 LLM 往返返回 3 个 tool_call：步数 = 1，tool_calls_total = 3
    sb = Sandbox(EnvState())
    for order_id in ("A123", "B456", "ZZZ"):
        sb.call("get_order", {"order_id": order_id}, step=1)
    assert {step.step for step in sb.trajectory()} == {1}
    assert sb.tool_calls_total == 3


def test_trajectory_args_are_snapshots():
    sb = Sandbox(EnvState())
    args = {"order_id": "A123"}
    sb.call("get_order", args, step=1)
    args["order_id"] = "B456"                     # 调用方后续修改
    assert sb.trajectory()[0].args["order_id"] == "A123"


def test_trajectory_returns_new_list():
    sb = Sandbox(EnvState())
    sb.call("get_order", {"order_id": "A123"}, step=1)
    snapshot = sb.trajectory()
    snapshot.clear()
    assert sb.tool_calls_total == 1


def test_reset_clears_state_and_trajectory():
    sb = Sandbox(EnvState())
    sb.call("cancel_shipment", {"order_id": "A123"}, step=1)
    sb.reset()
    assert sb.trajectory() == []
    assert sb.state.data["orders"]["A123"]["status"] == "shipped"


def test_state_is_shared_with_caller():
    state = EnvState()
    sb = Sandbox(state)
    sb.call("cancel_shipment", {"order_id": "A123"}, step=1)
    assert state.data["orders"]["A123"]["status"] == "cancelled"


@pytest.mark.parametrize("tool", ["get_user", "get_order", "calc_refund", "cancel_shipment"])
def test_single_arg_tools_reject_empty_args(tool):
    sb = Sandbox(EnvState())
    assert "缺少必填参数" in sb.call(tool, {}, step=1)["error"]
