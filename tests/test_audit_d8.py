"""D8 质检：可执行性 + 可判定性。全部离线（FakeClient）。"""
from __future__ import annotations

import json

from evalkit.audit import SAMPLE_PLAN, audit_tasks, run_once, sample_tasks, state_digest
from evalkit.client import ChatError, FakeClient

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


def _replies_for_success():
    return [
        {"role": "assistant", "content": None, "tool_calls": [CALL_CANCEL]},
        {"role": "assistant", "content": None, "tool_calls": [CALL_REFUND]},
        {"role": "assistant", "content": "已退款。"},
    ]


def test_sample_respects_quota_and_seed():
    tasks = [dict(TASK, id="g{0}".format(i), scenario=g)
             for g in SAMPLE_PLAN for i in range(20)]
    picked = sample_tasks(tasks, seed=1)
    assert len(picked) == sum(SAMPLE_PLAN.values()) == 30
    counts = {}
    for task in picked:
        counts[task["scenario"]] = counts.get(task["scenario"], 0) + 1
    assert counts == SAMPLE_PLAN


def test_sample_is_reproducible():
    tasks = [dict(TASK, id="g{0}".format(i), scenario="正常") for i in range(20)]
    a = [t["id"] for t in sample_tasks(tasks, {"正常": 5}, seed=7)]
    b = [t["id"] for t in sample_tasks(tasks, {"正常": 5}, seed=7)]
    assert a == b


def test_run_once_records_success():
    record = run_once(TASK, FakeClient(_replies_for_success()))
    assert record.ok is True and record.passed is True
    assert record.category is None
    assert record.state_digest and len(record.state_digest) == 16


def test_run_once_records_chat_error_without_raising():
    class Boom:
        def chat(self, messages, tools):
            raise ChatError("HTTP 429: 限流")

    record = run_once(TASK, Boom())
    assert record.ok is False
    assert "429" in (record.error or "")


def test_state_digest_is_stable_and_sensitive():
    state = {"orders": {"A123": {"status": "shipped"}}, "refunds": []}
    assert state_digest(state) == state_digest(dict(state))
    assert state_digest(state) != state_digest({"orders": {"A123": {"status": "refunded"}}, "refunds": []})


def test_audit_all_consistent():
    client = FakeClient(_replies_for_success() * 3)
    report = audit_tasks([TASK], client, repeats=3)
    assert report["executable"]["all_ok"] == 1
    assert report["executable"]["executable_rate"] == 1.0
    assert report["deterministic"]["consistency_rate"] == 1.0
    assert report["inconsistent_detail"] == []


def test_audit_detects_inconsistency_with_detail():
    # 第一次成功，第二次什么都没做（判定变成失败）→ 不一致，要能说清哪次不同
    replies = _replies_for_success() + [{"role": "assistant", "content": "我处理好了。"}]
    client = FakeClient(replies)
    report = audit_tasks([TASK], client, repeats=2)
    assert report["deterministic"]["inconsistent"] == 1
    detail = report["inconsistent_detail"][0]
    assert detail["task_id"] == "t1"
    assert len(detail["verdicts"]) == 2
    assert detail["diffs"][0]["run"] == 2
    assert detail["diffs"][0]["passed"] is False


def test_audit_splits_flaky_from_crashed():
    class Flaky:
        def __init__(self):
            self.calls = 0

        def chat(self, messages, tools):
            self.calls += 1
            if self.calls == 1:
                raise ChatError("HTTP 500: 服务端错误")
            from evalkit.client import ChatReply
            message = _replies_for_success()[0]
            return ChatReply(message=message, elapsed_ms=0.0, usage={"prompt_tokens": 10})

    report = audit_tasks([TASK], Flaky(), repeats=2)
    assert report["executable"]["flaky"] == 1
    assert report["executable"]["crashed"] == 0
    assert report["executable"]["flaky_ids"] == ["t1"]
