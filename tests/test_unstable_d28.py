"""D28 不稳定任务：标记 + 统计剔除。全部离线。

三个必测的点：
1. `split_unstable` 真的把标了 `unstable` 的摘出去（不是"看着像"）
2. 剔除后主结论的 N 变小、通过率变高（抖动的失败被拿掉了）
3. 没标的任务一条都不能少——**误剔比不剔更危险**，会凭空美化通过率
"""
from __future__ import annotations

from evalkit.build_report import build_unstable, split_unstable
from evalkit.discriminate import Outcome


def _task(tid: str, unstable: bool = False, scenario: str = "正常") -> dict:
    task = {"id": tid, "instruction": "退 A123", "scenario": scenario,
            "success_criteria": [{"path": "orders.A123.status", "expect": "refunded"}]}
    if unstable:
        task["unstable"] = True
        task["unstable_note"] = "模型有时做有时不做"
    return task


def _outcome(tid: str, passed: bool, category=None) -> Outcome:
    return Outcome(tid, "full", 0, True, passed, category, 3, 100, 20)


def test_split_separates_marked_tasks():
    tasks = [_task("t1"), _task("t2", unstable=True), _task("t3")]
    stable, unstable = split_unstable(tasks)
    assert [t["id"] for t in stable] == ["t1", "t3"]
    assert [t["id"] for t in unstable] == ["t2"]


def test_split_keeps_everything_when_nothing_marked():
    """没标就一条都不能少——误剔会凭空美化通过率。"""
    tasks = [_task("t1"), _task("t2")]
    stable, unstable = split_unstable(tasks)
    assert len(stable) == 2 and unstable == []


def test_split_ignores_false_value():
    """`unstable: false` 是"明确标了没问题"，不能当不稳剔除。"""
    task = _task("t1")
    task["unstable"] = False
    stable, unstable = split_unstable([task])
    assert len(stable) == 1 and unstable == []


def test_unstable_block_counts_and_share():
    human_all = [_task("t1", unstable=True), _task("t2"), _task("t3")]
    auto_all = [_task("a1"), _task("a2")]
    pools = {
        "人工": {"要做事": [_outcome("t1", False, "forbidden_tool"), _outcome("t1", False, "forbidden_tool")],
                 "不该动": []},
        "自动": {"要做事": [], "不该动": []},
    }
    block = build_unstable([human_all[0]], [], human_all, auto_all, pools)
    assert block["count"] == 1 and block["total"] == 5
    assert block["share"] == 0.2
    row = block["rows"][0]
    assert row["task_id"] == "t1" and row["runs"] == 2 and row["passed"] == 0
    assert row["rate"] == 0.0
    assert "forbidden_tool×2" in row["failure"]
    assert block["human_before"] == 3 and block["human_after"] == 2


def test_unstable_block_mixed_verdicts():
    human_all = [_task("t1", unstable=True)]
    pools = {"人工": {"要做事": [_outcome("t1", True), _outcome("t1", False, "forbidden_tool")],
                      "不该动": []},
             "自动": {"要做事": [], "不该动": []}}
    row = build_unstable([human_all[0]], [], human_all, [], pools)["rows"][0]
    assert row["passed"] == 1 and row["rate"] == 0.5
    assert row["failure"] == "forbidden_tool×1"


def test_removing_unstable_raises_pass_rate():
    """抖动的失败被拿掉后，通过率应该变高——这是剔除的动机，也要能被测出来。"""
    tasks = [_task("t1"), _task("t2", unstable=True)]
    stable, unstable = split_unstable(tasks)
    all_rows = [_outcome("t1", True), _outcome("t2", False, "forbidden_tool"),
                _outcome("t2", True)]
    before = sum(1 for o in all_rows if o.passed) / len(all_rows)
    keep = {t["id"] for t in stable}
    after_rows = [o for o in all_rows if o.task_id in keep]
    after = sum(1 for o in after_rows if o.passed) / len(after_rows)
    assert before == 0.6667 or abs(before - 2 / 3) < 1e-4
    assert after == 1.0
