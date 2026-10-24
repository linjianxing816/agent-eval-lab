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


def test_agents_registry_has_all_four():
    assert set(AGENTS) == {"strong", "weak", "weak_no_calc", "weak_no_user"}


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


def test_round3_weakenings_each_remove_one_tool():
    from evalkit.discriminate import WEAK_NO_CALC, WEAK_NO_USER
    strong_tools = {t["function"]["name"] for t in STRONG_AGENT["tools"]}
    assert strong_tools - {t["function"]["name"] for t in WEAK_NO_CALC["tools"]} == {"calc_refund"}
    assert strong_tools - {t["function"]["name"] for t in WEAK_NO_USER["tools"]} == {"get_user"}
    # 只改一处：prompt 都不动
    assert WEAK_NO_CALC["system_prompt"] == STRONG_AGENT["system_prompt"]
    assert WEAK_NO_USER["system_prompt"] == STRONG_AGENT["system_prompt"]


def test_compare_by_type_uses_reversed_formula_for_unchanged():
    from evalkit.discriminate import compare_by_type
    tasks = [{"id": "a", "scenario": "正常"}, {"id": "b", "scenario": "诱导越权"}]
    strong = [Outcome("a", "strong", 0, True, True, None, 3, 0, 0),
              Outcome("b", "strong", 0, True, True, None, 2, 0, 0)]
    # 要做事组弱 Agent 失败；不该动组弱 Agent 反而通过
    weak = [Outcome("a", "weak", 0, True, False, "wrong_final_state", 3, 0, 0),
            Outcome("b", "weak", 0, True, True, None, 1, 0, 0)]
    report = compare_by_type(strong, weak, tasks)
    assert report["要做事"]["gap"] == 1.0      # strong 1.0 - weak 0.0
    assert report["不该动"]["gap"] == 0.0      # weak 1.0 - strong 1.0
    assert report["要做事"]["tasks"] == 1


def test_compare_by_type_marks_conservative_gap():
    from evalkit.discriminate import compare_by_type
    tasks = [{"id": "b", "scenario": "状态陷阱"}]
    strong = [Outcome("b", "strong", 0, True, False, "forbidden_tool", 3, 0, 0)]
    weak = [Outcome("b", "weak", 0, True, True, None, 1, 0, 0)]
    report = compare_by_type(strong, weak, tasks)
    assert report["不该动"]["gap"] == 1.0      # weak 1.0 - strong 0.0
    assert "更保守" in report["不该动"]["direction"]


def test_cost_is_none_without_price(monkeypatch):
    from evalkit.discriminate import cost_of, token_totals
    monkeypatch.delenv("PRICE_INPUT_PER_1M", raising=False)
    monkeypatch.delenv("PRICE_OUTPUT_PER_1M", raising=False)
    totals = token_totals([Outcome("t", "strong", 0, True, True, None, 3, 1000, 100)])
    assert totals == {"runs": 1, "input": 1000, "output": 100}
    assert cost_of(totals) is None          # 单价没配就返回 None，不估


def test_cost_uses_configured_price(monkeypatch):
    from evalkit.discriminate import cost_of
    monkeypatch.setenv("PRICE_INPUT_PER_1M", "1")
    monkeypatch.setenv("PRICE_OUTPUT_PER_1M", "8")
    result = cost_of({"runs": 1, "input": 1_000_000, "output": 1_000_000})
    assert result["input_cost"] == 1.0
    assert result["output_cost"] == 8.0
    assert result["total"] == 9.0


def test_cost_of_loads_dotenv_from_file(tmp_path):
    # 真 bug 回归：cost_of 只读 os.environ 而不 load_dotenv，填了单价也永远是 None
    import os
    from evalkit.discriminate import cost_of
    env_file = tmp_path / ".env"
    env_file.write_text("PRICE_INPUT_PER_1M=2\nPRICE_OUTPUT_PER_1M=9\n", encoding="utf-8")
    os.environ.pop("PRICE_INPUT_PER_1M", None)
    os.environ.pop("PRICE_OUTPUT_PER_1M", None)
    result = cost_of({"runs": 1, "input": 1_000_000, "output": 1_000_000}, env_path=env_file)
    assert result is not None
    assert result["input_cost"] == 2.0 and result["output_cost"] == 9.0


def test_small_gap_inside_noise_is_not_reported_as_direction():
    # 人工集要做事组 22 个观测、门槛 0.213，-0.045 完全在噪声里，不能说"方向反了"
    from evalkit.discriminate import compare_by_type
    tasks = [{"id": "task-0{0:02d}".format(i), "scenario": "正常"} for i in range(1, 23)]
    strong = [Outcome(t["id"], "strong", 0, True, i % 2 == 0, None, 3, 10, 1) for i, t in enumerate(tasks)]
    weak = [Outcome(t["id"], "weak", 0, True, i % 2 == 0, None, 3, 10, 1) for i, t in enumerate(tasks)]
    # 只让一条不同 → gap 约 0.045，远小于门槛
    weak[0] = Outcome(tasks[0]["id"], "weak", 0, True, not strong[0].passed, None, 3, 10, 1)
    report = compare_by_type(strong, weak, tasks)
    row = report["要做事"]
    assert row["runs"] == 22
    assert row["threshold"] == 0.213
    assert abs(row["gap"]) < row["threshold"]
    assert "噪声" in row["direction"]
