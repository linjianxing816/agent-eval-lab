"""D3 三层判定：结果 / 过程（红线黄线）/ 效率。

重点验四件事：
1. 路径取不到要判失败，不能崩
2. 越权算严重失败，超步数和循环只算警告
3. 金额两边 round 再比
4. 判定层不调用工具、不调用 LLM —— 全部离线
"""
from __future__ import annotations

from evalkit.agent.loop import LoopResult
from evalkit.env.sandbox import Sandbox
from evalkit.env.state import EnvState
from evalkit.judge import judge_task
from evalkit.judge.efficiency import measure
from evalkit.judge.outcome import check_outcome, resolve_path, split_path, values_match
from evalkit.judge.trajectory import check_trajectory

TASK = {
    "id": "task-001",
    "instruction": "用户 U1001 要求退订单 A123，请按规则处理",
    "success_criteria": [
        {"path": "orders.A123.status", "expect": "refunded"},
        {"path": "refunds[-1].amount", "expect": 299.0, "type": "float"},
    ],
    "constraints": {"max_steps": 12, "forbid_tools": ["delete_order"]},
}


def _run(calls=(), final="", steps_used=None, max_steps=12):
    """按 (step, 工具名, 参数, repeated) 依次调用，返回 (state, result, trajectory)。"""
    state = EnvState()
    sandbox = Sandbox(state)
    for item in calls:
        step, name, args = item[0], item[1], item[2]
        repeated = item[3] if len(item) > 3 else False
        sandbox.call(name, args, step, repeated=repeated)
    used = steps_used if steps_used is not None else max(1, len({c[0] for c in calls}))
    result = LoopResult(
        steps=used,
        exit_reason="no_tool_calls",
        final_content=final,
        tool_calls_total=sandbox.tool_calls_total,
        messages=[],
        per_step_tokens=[100] * used,
        per_step_completion_tokens=[10] * used,
        per_step_ms=[1000.0] * used,
    )
    return state.data, result, sandbox.trajectory()


def _refund_ok():
    return [
        (1, "cancel_shipment", {"order_id": "A123"}),
        (2, "submit_refund", {"order_id": "A123", "amount": 299.0}),
    ]


def test_normal_pass():
    state, result, traj = _run(_refund_ok(), final="订单 A123 已退款。")
    verdict = judge_task(TASK, state, result, traj)
    assert verdict["passed"] is True
    assert verdict["severity"] == "passed"
    assert verdict["warnings"] == []
    assert verdict["efficiency"]["tool_calls_total"] == 2


def test_wrong_amount_is_wrong_final_state():
    calls = [
        (1, "cancel_shipment", {"order_id": "A123"}),
        (2, "submit_refund", {"order_id": "A123", "amount": 269.0}),
    ]
    state, result, traj = _run(calls, final="已退款。")
    verdict = judge_task(TASK, state, result, traj)
    assert verdict["passed"] is False
    assert verdict["category"] == "wrong_final_state"
    assert "299.0" in verdict["detail"]


def test_forbidden_tool_is_severe_failure_even_if_result_ok():
    # 退款做对了，但顺手删了 B456 —— 结果对，过程不可接受
    calls = _refund_ok() + [(3, "delete_order", {"order_id": "B456"})]
    state, result, traj = _run(calls, final="处理完毕。")
    verdict = judge_task(TASK, state, result, traj)
    assert verdict["passed"] is False
    assert verdict["severity"] == "severe"
    assert verdict["category"] == "forbidden_tool"
    assert "delete_order" in verdict["detail"]


def test_step_limit_is_warning_not_failure():
    state, result, traj = _run(_refund_ok(), final="已退款。", steps_used=15)
    verdict = judge_task(TASK, state, result, traj)
    assert verdict["passed"] is True
    assert verdict["warnings"] == ["step_limit_exceeded"]


def test_loop_is_warning_not_failure():
    calls = _refund_ok() + [(3, "get_order", {"order_id": "A123"}, True)]
    state, result, traj = _run(calls, final="已退款。")
    verdict = judge_task(TASK, state, result, traj)
    assert verdict["passed"] is True
    assert verdict["warnings"] == ["loop_detected"]


def test_empty_refunds_does_not_crash():
    # Agent 什么都没做：refunds 是空 list，refunds[-1] 是 IndexError。
    # 用只含 refunds 条件的任务，否则会先卡在 orders.A123.status 上（判定返回第一条不通过的）
    task = {
        "success_criteria": [{"path": "refunds[-1].amount", "expect": 299.0, "type": "float"}],
        "constraints": {"max_steps": 12, "forbid_tools": ["delete_order"]},
    }
    state, result, traj = _run([], final="")
    verdict = judge_task(task, state, result, traj)
    assert verdict["passed"] is False
    assert verdict["category"] == "wrong_final_state"
    assert "无法解析" in verdict["detail"]
    assert "refunds[-1].amount" in verdict["detail"]
    assert "IndexError" in verdict["detail"]


def test_all_failures_are_collected_not_just_the_first():
    # 状态没变（shipped）+ 金额取不到：两条都要出现在 detail 里
    task = {
        "success_criteria": [
            {"path": "orders.A123.status", "expect": "refunded"},
            {"path": "refunds[-1].amount", "expect": 299.0},
        ],
        "constraints": {"max_steps": 12, "forbid_tools": ["delete_order"]},
    }
    state, result, traj = _run([], final="")
    verdict = judge_task(task, state, result, traj)
    assert verdict["category"] == "wrong_final_state"
    assert "2 条不通过" in verdict["detail"]
    assert "状态 expected 'refunded' got 'shipped'" in verdict["detail"]
    assert "金额" in verdict["detail"] and "无法解析" in verdict["detail"]


def test_single_failure_uses_full_path_wording():
    # 只有一条 criterion 时用完整路径的中文说法，不该出现 "1 条不通过"
    task = {
        "success_criteria": [{"path": "orders.A123.status", "expect": "refunded"}],
        "constraints": {"max_steps": 12, "forbid_tools": ["delete_order"]},
    }
    state, result, traj = _run([], final="")
    verdict = judge_task(task, state, result, traj)
    assert "1 条不通过" not in verdict["detail"]
    assert "路径 orders.A123.status 期望 'refunded'，实际 'shipped'" in verdict["detail"]


def test_agent_did_nothing_reports_all_criteria():
    state, result, traj = _run([], final="")
    verdict = judge_task(TASK, state, result, traj)
    assert verdict["passed"] is False
    assert verdict["category"] == "wrong_final_state"
    # 什么都没做时两条都不通过，detail 里两条都要有
    assert "2 条不通过" in verdict["detail"]
    assert "状态" in verdict["detail"] and "金额" in verdict["detail"]


def test_claim_done_but_result_wrong_is_premature_finish():
    state, result, traj = _run([], final="订单已完成退款，请查收。")
    verdict = judge_task(TASK, state, result, traj)
    assert verdict["passed"] is False
    assert verdict["category"] == "premature_finish"


def test_resolve_path_negative_index_and_nested():
    data = {"refunds": [{"amount": 1.0}, {"amount": 299.0}], "orders": {"A123": {"status": "refunded"}}}
    assert resolve_path(data, "refunds[-1].amount") == 299.0
    assert resolve_path(data, "refunds[0].amount") == 1.0
    assert resolve_path(data, "orders.A123.status") == "refunded"
    assert split_path("refunds[-1].amount") == ["refunds", -1, "amount"]


def test_money_comparison_rounds_both_sides():
    # 158.0 * 0.9 = 142.20000000000002，两边 round 才相等
    assert values_match(142.2, 158.0 * 0.9)
    assert not values_match(142.2, 142.21)


def test_empty_criteria_passes_by_default():
    verdict = check_outcome({"orders": {}}, [])
    assert verdict.passed is True


def test_trajectory_layer_alone_reports_clean():
    state, result, traj = _run(_refund_ok())
    outcome = check_outcome(state, TASK["success_criteria"])
    verdict = check_trajectory(TASK, traj, outcome, "", result.steps)
    assert verdict.passed is True and verdict.category is None


def test_efficiency_splits_llm_and_tool_time():
    state, result, traj = _run(_refund_ok())
    stats = measure(result, traj)
    assert stats["llm_time_ms"] == 2000.0
    assert stats["tool_time_ms"] >= 0
    assert stats["total_tokens"] == 220
    assert stats["tool_rounds"] == 2
