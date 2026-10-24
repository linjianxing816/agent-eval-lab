"""D10 区分度：弱 Agent 定义 + 批量跑 + 统计。全部离线。"""
from __future__ import annotations

from evalkit.agent.strong import STRONG_AGENT
from evalkit.client import ChatError, FakeClient
from evalkit.discriminate import (
    AGENTS,
    WEAK_AGENT,
    Outcome,
    compare,
    pass_rate,
    per_task_diff,
    run_batch,
    summarize_agent,
)

TASK = {
    "id": "t1",
    "instruction": "退 A123",
    "initial_state": {"orders": {"A123": {"user": "U1001", "amount": 299.0, "status": "shipped"}}},
    "tools_allowed": ["get_user", "get_order", "calc_refund", "cancel_shipment", "submit_refund"],
    "success_criteria": [{"path": "orders.A123.status", "expect": "refunded"},
                         {"path": "refunds[-1].amount", "expect": 299.0}],
    "constraints": {"max_steps": 12, "forbid_tools": ["delete_order"]},
}

CALL_CANCEL = {"id": "c1", "type": "function", "function": {"name": "cancel_shipment", "arguments": '{"order_id": "A123"}'}}
CALL_REFUND = {"id": "c2", "type": "function", "function": {"name": "submit_refund", "arguments": '{"order_id": "A123", "amount": 299.0}'}}


def _success_replies():
    return [
        {"role": "assistant", "content": None, "tool_calls": [CALL_CANCEL]},
        {"role": "assistant", "content": None, "tool_calls": [CALL_REFUND]},
        {"role": "assistant", "content": "已退款。"},
    ]


def test_weak_agent_only_lacks_get_order():
    strong_tools = {t["function"]["name"] for t in STRONG_AGENT["tools"]}
    weak_tools = {t["function"]["name"] for t in WEAK_AGENT["tools"]}
    assert strong_tools - weak_tools == {"get_order"}
    # 只改一处：prompt 一个字不改
    assert WEAK_AGENT["system_prompt"] == STRONG_AGENT["system_prompt"]


def test_agents_registry_has_both():
    assert set(AGENTS) == {"strong", "weak"}


def test_run_batch_collects_outcomes():
    client = FakeClient(_success_replies() * 2)
    outcomes = run_batch([TASK], AGENTS["strong"], client, repeats=2)
    assert len(outcomes) == 2
    assert all(o.passed for o in outcomes)
    assert all(o.tokens > 0 for o in outcomes)


def test_run_batch_records_crash_without_raising():
    class Boom:
        def chat(self, messages, tools):
            raise ChatError("HTTP 500")

    outcomes = run_batch([TASK], AGENTS["weak"], Boom(), repeats=1)
    assert len(outcomes) == 1
    assert outcomes[0].ok is False and outcomes[0].passed is False


def test_pass_rate_and_summary():
    rows = [
        Outcome("t1", "strong", 0, True, True, None, 3, 100, 10),
        Outcome("t1", "strong", 1, True, False, "wrong_final_state", 4, 100, 10),
        Outcome("t2", "strong", 0, True, False, "forbidden_tool", 2, 100, 10),
    ]
    assert pass_rate(rows) == 0.3333
    summary = summarize_agent(rows)
    assert summary["passed"] == 1
    assert summary["failure_distribution"] == {"forbidden_tool": 1, "wrong_final_state": 1}
    assert summary["tokens"] == 330


def test_per_task_diff_marks_effective_tasks():
    strong = [Outcome("t1", "strong", 0, True, True, None, 3, 0, 0)]
    weak = [Outcome("t1", "weak", 0, True, False, "wrong_final_state", 3, 0, 0)]
    rows = per_task_diff(strong, weak)
    assert rows[0]["diff"] == 1.0


def test_compare_reports_gap():
    strong = [Outcome("t1", "strong", 0, True, True, None, 3, 10, 1)]
    weak = [Outcome("t1", "weak", 0, True, False, "wrong_final_state", 3, 10, 1)]
    report = compare({"set": {"strong": strong, "weak": weak}})
    assert report["set"]["gap"] == 1.0
    assert report["set"]["effective_tasks"] == 1
