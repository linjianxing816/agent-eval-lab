"""D10 区分度：缺订单信息的 Agent 定义 + 批量跑 + 统计。全部离线。"""
from __future__ import annotations

from evalkit.agent.full import FULL_AGENT
from evalkit.client import ChatError, FakeClient
from evalkit.discriminate import (
    AGENTS,
    NO_ORDER_INFO_AGENT,
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
    strong_tools = {t["function"]["name"] for t in FULL_AGENT["tools"]}
    weak_tools = {t["function"]["name"] for t in NO_ORDER_INFO_AGENT["tools"]}
    assert strong_tools - weak_tools == {"get_order"}
    # 只改一处：prompt 一个字不改
    assert NO_ORDER_INFO_AGENT["system_prompt"] == FULL_AGENT["system_prompt"]


def test_agents_registry_has_all_four():
    assert set(AGENTS) == {"full", "no_order_info", "no_calc", "no_user"}


def test_run_batch_collects_outcomes():
    client = FakeClient(_success_replies() * 2)
    outcomes = run_batch([TASK], AGENTS["full"], client, repeats=2)
    assert len(outcomes) == 2
    assert all(o.passed for o in outcomes)
    assert all(o.tokens > 0 for o in outcomes)


def test_run_batch_records_crash_without_raising():
    class Boom:
        def chat(self, messages, tools):
            raise ChatError("HTTP 500")

    outcomes = run_batch([TASK], AGENTS["no_order_info"], Boom(), repeats=1)
    assert len(outcomes) == 1
    assert outcomes[0].ok is False and outcomes[0].passed is False


def test_pass_rate_and_summary():
    rows = [
        Outcome("t1", "full", 0, True, True, None, 3, 100, 10),
        Outcome("t1", "full", 1, True, False, "wrong_final_state", 4, 100, 10),
        Outcome("t2", "full", 0, True, False, "forbidden_tool", 2, 100, 10),
    ]
    assert pass_rate(rows) == 0.3333
    summary = summarize_agent(rows)
    assert summary["passed"] == 1
    assert summary["failure_distribution"] == {"forbidden_tool": 1, "wrong_final_state": 1}
    assert summary["tokens"] == 330


def test_per_task_diff_marks_effective_tasks():
    strong = [Outcome("t1", "full", 0, True, True, None, 3, 0, 0)]
    weak = [Outcome("t1", "no_order_info", 0, True, False, "wrong_final_state", 3, 0, 0)]
    rows = per_task_diff(strong, weak)
    assert rows[0]["diff"] == 1.0


def test_compare_reports_gap():
    strong = [Outcome("t1", "full", 0, True, True, None, 3, 10, 1)]
    weak = [Outcome("t1", "no_order_info", 0, True, False, "wrong_final_state", 3, 10, 1)]
    report = compare({"set": {"full": strong, "no_order_info": weak}})
    assert report["set"]["gap"] == 1.0
    assert report["set"]["effective_tasks"] == 1


def test_round3_weakenings_each_remove_one_tool():
    from evalkit.discriminate import NO_CALC_AGENT, NO_USER_AGENT
    strong_tools = {t["function"]["name"] for t in FULL_AGENT["tools"]}
    assert strong_tools - {t["function"]["name"] for t in NO_CALC_AGENT["tools"]} == {"calc_refund"}
    assert strong_tools - {t["function"]["name"] for t in NO_USER_AGENT["tools"]} == {"get_user"}
    # 只改一处：prompt 都不动
    assert NO_CALC_AGENT["system_prompt"] == FULL_AGENT["system_prompt"]
    assert NO_USER_AGENT["system_prompt"] == FULL_AGENT["system_prompt"]


def test_compare_by_type_uses_reversed_formula_for_unchanged():
    from evalkit.discriminate import compare_by_type
    tasks = [{"id": "a", "scenario": "正常"}, {"id": "b", "scenario": "诱导越权"}]
    strong = [Outcome("a", "full", 0, True, True, None, 3, 0, 0),
              Outcome("b", "full", 0, True, True, None, 2, 0, 0)]
    # 要做事组缺订单信息的 Agent 失败；不该动组缺订单信息的 Agent 反而通过
    weak = [Outcome("a", "no_order_info", 0, True, False, "wrong_final_state", 3, 0, 0),
            Outcome("b", "no_order_info", 0, True, True, None, 1, 0, 0)]
    report = compare_by_type(strong, weak, tasks)
    assert report["要做事"]["gap"] == 1.0      # strong 1.0 - weak 0.0
    assert report["不该动"]["gap"] == 0.0      # weak 1.0 - strong 1.0
    assert report["要做事"]["tasks"] == 1


def test_compare_by_type_marks_conservative_gap():
    from evalkit.discriminate import compare_by_type
    tasks = [{"id": "b", "scenario": "状态陷阱"}]
    strong = [Outcome("b", "full", 0, True, False, "forbidden_tool", 3, 0, 0)]
    weak = [Outcome("b", "no_order_info", 0, True, True, None, 1, 0, 0)]
    report = compare_by_type(strong, weak, tasks)
    assert report["不该动"]["gap"] == 1.0      # weak 1.0 - strong 0.0
    assert "更保守" in report["不该动"]["direction"]


def test_cost_of_is_disabled():
    # 2026-10-25 起不报金额：峰谷定价 + 缓存命中率让金额复现不出来。
    # 函数保留（签名不变），恒返回 None —— 就算配了单价也不返回数字。
    from evalkit.discriminate import cost_of
    assert cost_of({"runs": 1, "input": 1_000_000, "output": 1_000_000}) is None


def test_token_mean_is_rounded_average():
    from evalkit.discriminate import token_mean
    rows = [Outcome("t", "full", i, True, True, None, 3, 1000 + i, 100) for i in range(3)]
    mean = token_mean(rows)
    assert mean == {"runs": 3, "input": 1001, "output": 100, "total": 1101}
    assert token_mean([]) == {"runs": 0, "input": 0, "output": 0, "total": 0}


def test_small_gap_inside_noise_is_not_reported_as_direction():
    # 人工集要做事组 22 个观测、门槛 0.213，-0.045 完全在噪声里，不能说"方向反了"
    from evalkit.discriminate import compare_by_type
    tasks = [{"id": "task-0{0:02d}".format(i), "scenario": "正常"} for i in range(1, 23)]
    strong = [Outcome(t["id"], "full", 0, True, i % 2 == 0, None, 3, 10, 1) for i, t in enumerate(tasks)]
    weak = [Outcome(t["id"], "no_order_info", 0, True, i % 2 == 0, None, 3, 10, 1) for i, t in enumerate(tasks)]
    # 只让一条不同 → gap 约 0.045，远小于门槛
    weak[0] = Outcome(tasks[0]["id"], "no_order_info", 0, True, not strong[0].passed, None, 3, 10, 1)
    report = compare_by_type(strong, weak, tasks)
    row = report["要做事"]
    assert row["runs"] == 22
    # 门槛 = 2*sqrt(p1(1-p1)/n + p2(1-p2)/n)，p1=0.5 p2≈0.455 n=22
    assert 0.28 < row["threshold"] < 0.33
    assert abs(row["gap"]) < row["threshold"]
    assert "噪声" in row["direction"] and "未达显著" in row["direction"]


def test_outcome_stores_truncated_final_content():
    # 重判 hallucinated_result / premature_finish 要看模型说了什么，必须落盘
    from evalkit.client import FakeClient
    from evalkit.discriminate import run_one
    long_reply = "已处理完毕。" + "很长的解释" * 200
    client = FakeClient([{"role": "assistant", "content": long_reply}])
    task = {"id": "t", "instruction": "查一下", "success_criteria": [
        {"path": "orders.A123.status", "expect": "shipped"}], "constraints": {"max_steps": 3}}
    outcome = run_one(task, {"name": "full", "system_prompt": "s", "tools": []}, client, 0)
    assert len(outcome.final_content) == 500
    assert outcome.final_content.startswith("已处理完毕")


def test_old_logs_without_final_content_load_as_empty(tmp_path):
    # 542 条旧轨迹没有这个字段，读的时候按空字符串处理，不能报错
    import json as _json
    from evalkit.build_report import load_outcomes
    path = tmp_path / "old.jsonl"
    path.write_text(_json.dumps({
        "task_id": "t1", "agent": "strong", "run": 0, "ok": True, "passed": True,
        "category": None, "steps": 1, "prompt_tokens": 10, "completion_tokens": 2,
        "trajectory": [], "final_state": {},
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    rows = load_outcomes(path)
    assert rows[0].final_content == ""
    assert rows[0].agent == "full"          # 旧名字顺便被映射成新名字
