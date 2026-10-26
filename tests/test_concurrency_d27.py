"""D27 并发跑批：线程池 + Semaphore + 判定一致率。全部离线，不花钱。

三个必测的点：
1. **顺序不能变** —— 一致性验证按下标逐条比，gather 一乱序就全对不上
2. **Semaphore 真的限流** —— 不限的话并发 16 会变成"30 个一起冲"，429 怪不到并发度头上
3. **一致率要能分辨"并发的问题"和"模型随机"** —— 所以必须有一条噪声基线
"""
from __future__ import annotations

import asyncio
import threading
import time

import pytest

from evalkit.client import ChatError, ChatReply
from evalkit.discriminate import (
    Outcome,
    agreement,
    run_batch_async,
    verdict_key,
    within_task_consistency,
)
from evalkit.perf import pick_best_concurrency


class SlowClient:
    """线程安全的假模型：每次 chat 睡一会儿，同时记录"同时在飞"的峰值。

    `FakeClient` 不能用来测并发——它的 `_replies.pop(0)` 和 `calls.append`
    在多线程下会漏记，测出来的峰值是假的。
    """

    def __init__(self, delay: float = 0.03) -> None:
        self.delay = delay
        self._lock = threading.Lock()
        self.inflight = 0
        self.peak = 0
        self.calls = 0

    def chat(self, messages: list[dict], tools: list[dict]) -> ChatReply:
        with self._lock:
            self.inflight += 1
            self.calls += 1
            self.peak = max(self.peak, self.inflight)
        try:
            time.sleep(self.delay)
            return ChatReply(
                message={"role": "assistant", "content": "好的"},
                elapsed_ms=self.delay * 1000,
                usage={"prompt_tokens": 10, "completion_tokens": 2},
            )
        finally:
            with self._lock:
                self.inflight -= 1


class Boom429:
    """每次都 429。用来验证 status 真的从 ChatError 传到了 Outcome。"""

    def chat(self, messages: list[dict], tools: list[dict]) -> ChatReply:
        raise ChatError("HTTP 429: 限流", status=429)


AGENT = {"name": "full", "system_prompt": "你是客服", "tools": []}


def _tasks(n: int = 5) -> list[dict]:
    return [{
        "id": "task-{0:03d}".format(i),
        "instruction": "退 A123",
        "initial_state": {"orders": {"A123": {"user": "U1001", "amount": 299.0, "status": "shipped"}}},
        "success_criteria": [{"path": "orders.A123.status", "expect": "refunded"}],
        "constraints": {"max_steps": 2},
    } for i in range(1, n + 1)]


def _outcome(task_id: str, run: int = 0, passed: bool = True, category=None,
             state=None, ok: bool = True, status=None) -> Outcome:
    return Outcome(task_id, "full", run, ok, passed, category, 1, 1, 1,
                   final_state=state, error_status=status)


# ---------- 顺序与并发度 ----------

def test_result_order_matches_serial_layout():
    """结果必须按 (任务, 第几次) 排好，和串行 run_batch 的落盘顺序一致。"""
    tasks = _tasks(4)
    outcomes = asyncio.run(run_batch_async(tasks, AGENT, SlowClient(0.005), repeats=2, concurrency=3))
    assert [o.task_id for o in outcomes] == [t["id"] for t in tasks for _ in range(2)]
    assert [o.run for o in outcomes] == [0, 1] * 4


def test_semaphore_caps_inflight_requests():
    """并发 3 → 同时在飞的请求数不能超过 3。"""
    client = SlowClient(0.03)
    asyncio.run(run_batch_async(_tasks(6), AGENT, client, repeats=2, concurrency=3))
    assert client.calls == 12
    assert client.peak <= 3
    # 上限没生效的话会是 12；真的并发了就不会是 1
    assert client.peak >= 2


def test_concurrency_one_is_serial():
    client = SlowClient(0.005)
    asyncio.run(run_batch_async(_tasks(3), AGENT, client, repeats=2, concurrency=1))
    assert client.peak == 1


def test_threads_override_caps_inflight():
    """`threads` 小于 concurrency 时，瓶颈是线程池——这时候测出来的上限不是服务端的。"""
    client = SlowClient(0.03)
    asyncio.run(run_batch_async(_tasks(6), AGENT, client, repeats=1, concurrency=6, threads=2))
    assert client.peak <= 2


def test_concurrency_zero_rejected():
    with pytest.raises(ValueError):
        asyncio.run(run_batch_async(_tasks(1), AGENT, SlowClient(), concurrency=0))


def test_timings_filled_per_execution():
    timings: dict = {}
    asyncio.run(run_batch_async(_tasks(3), AGENT, SlowClient(0.01), repeats=2,
                                concurrency=2, timings=timings))
    assert set(timings) == {(t["id"], r) for t in _tasks(3) for r in range(2)}
    assert all(v >= 9 for v in timings.values())


# ---------- 失败也要落在 Outcome 里 ----------

def test_429_status_recorded():
    """category 只有 request_error 一个值，429 和超时分不开就判不了"最优并发"。"""
    outcomes = asyncio.run(run_batch_async(_tasks(2), AGENT, Boom429(), repeats=1, concurrency=2))
    assert [o.category for o in outcomes] == ["request_error", "request_error"]
    assert [o.error_status for o in outcomes] == [429, 429]
    assert not any(o.ok for o in outcomes)


# ---------- 判定签名与一致率 ----------

def test_verdict_key_covers_three_items():
    base = _outcome("t1", state={"orders": {"A123": {"status": "refunded"}}})
    assert verdict_key(base) == verdict_key(_outcome("t1", state={"orders": {"A123": {"status": "refunded"}}}))
    # 只变 passed
    assert verdict_key(base) != verdict_key(_outcome("t1", passed=False, state=base.final_state))
    # 只变 category
    assert verdict_key(base) != verdict_key(_outcome("t1", category="premature_finish", state=base.final_state))
    # 只变最终状态（**这条最容易漏**：两个都 passed 但退的金额不同）
    assert verdict_key(base) != verdict_key(_outcome("t1", state={"orders": {"A123": {"status": "shipped"}}}))


def test_state_hash_ignores_key_order():
    a = _outcome("t1", state={"a": 1, "b": 2})
    b = _outcome("t1", state={"b": 2, "a": 1})
    assert verdict_key(a) == verdict_key(b)


def test_agreement_all_same():
    rows = [_outcome("t1"), _outcome("t2")]
    report = agreement(rows, list(rows))
    assert report["rate"] == 1.0 and report["valid_rate"] == 1.0
    assert report["diffs"] == []


def test_agreement_reports_which_field_changed():
    base = [_outcome("t1", state={"s": 1}), _outcome("t2", state={"s": 2})]
    cur = [_outcome("t1", state={"s": 9}), _outcome("t2", state={"s": 2}, passed=False)]
    report = agreement(base, cur)
    assert report["same"] == 0
    assert report["diffs"][0]["changed"] == ["state"]
    assert report["diffs"][1]["changed"] == ["passed"]


def test_agreement_rejects_length_mismatch():
    with pytest.raises(ValueError):
        agreement([_outcome("t1")], [_outcome("t1"), _outcome("t1")])


def test_pseudo_agreement_of_two_request_errors_excluded():
    """两边都失败时三项全一样，会被算成"一致"——这是伪一致，valid_rate 要把它剔掉。"""
    base = [_outcome("t1", passed=False, category="request_error", ok=False, state=None, status=429),
            _outcome("t2")]
    cur = [_outcome("t1", passed=False, category="request_error", ok=False, state=None, status=429),
           _outcome("t2")]
    report = agreement(base, cur)
    assert report["rate"] == 1.0
    assert report["valid_total"] == 1          # 只有 t2 两边都有效
    assert report["valid_same"] == 1


# ---------- 任务内一致率（D8 口径） ----------

def test_within_task_all_stable():
    rows = [_outcome("t1", run=r, state={"s": 1}) for r in range(3)]
    report = within_task_consistency(rows)
    assert report == {"tasks": 1, "stable": 1, "rate": 1.0, "unstable_tasks": []}


def test_within_task_counts_tasks_not_runs():
    """分母必须是任务数。30 条 × 3 次 → 30，不是 90。"""
    rows = [_outcome("t{0}".format(i), run=r, state={"s": i}) for i in range(4) for r in range(3)]
    report = within_task_consistency(rows)
    assert report["tasks"] == 4 and report["stable"] == 4


def test_within_task_flags_unstable_task():
    rows = ([_outcome("t1", run=r, state={"s": 1}) for r in range(3)]
            + [_outcome("t2", run=0, state={"s": 2}), _outcome("t2", run=1, state={"s": 3}),
               _outcome("t2", run=2, state={"s": 2})])
    report = within_task_consistency(rows)
    assert report["tasks"] == 2 and report["stable"] == 1
    assert report["rate"] == 0.5
    assert report["unstable_tasks"] == ["t2"]


def test_within_task_empty():
    assert within_task_consistency([])["rate"] == 0.0


# ---------- 最优并发 ----------

def _row(level, wall, throughput, n429=0, err=0):
    return {"concurrency": level, "wall_s": wall, "throughput": throughput,
            "status_429": n429, "request_error": err}


def test_best_picks_fastest_clean_level():
    rows = [_row(1, 100, 0.3), _row(2, 55, 0.55), _row(4, 32, 0.94), _row(8, 30, 1.0, n429=3)]
    best = pick_best_concurrency(rows)
    assert best["best"] == 4                 # 8 更快但 429 了，不算
    assert best["first_429"] == 8
    assert best["strictly_below_429"] is True


def test_best_skips_level_with_request_errors():
    rows = [_row(1, 100, 0.3), _row(2, 50, 0.6), _row(4, 40, 0.75, err=2)]
    best = pick_best_concurrency(rows)
    assert best["best"] == 2                 # 4 吞吐更高但 2 条失败


def test_best_falls_back_when_all_have_429():
    rows = [_row(1, 100, 0.3, n429=1), _row(2, 50, 0.6, n429=5)]
    best = pick_best_concurrency(rows)
    assert best["best"] == 1
    assert best["first_429"] == 1
    assert "只能当参考" in best["why"]


def test_best_no_429_anywhere():
    rows = [_row(1, 100, 0.3), _row(2, 55, 0.55), _row(4, 33, 0.9)]
    best = pick_best_concurrency(rows)
    assert best["best"] == 4
    assert best["first_429"] is None
    assert best["strictly_below_429"] is True


def test_best_empty_rows():
    assert pick_best_concurrency([])["best"] is None
