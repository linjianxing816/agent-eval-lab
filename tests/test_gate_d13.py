"""D13 门禁 + diff。全部离线。"""
from __future__ import annotations

from evalkit.diff import _group_of, diff_sets, to_markdown
from evalkit.discriminate import Outcome
import pytest

from evalkit.gate import evaluate, gap_threshold


def _row(tid, agent, passed, category=None, steps=3):
    return Outcome(tid, agent, 0, True, passed, category, steps, 10, 1)


def test_gap_threshold_uses_two_proportion_se():
    # gap 是两个比例之差，se = sqrt(p1(1-p1)/n + p2(1-p2)/n)，不是单比例的 sqrt(0.25/n)
    assert gap_threshold(0.5, 0.5, 171) == pytest.approx(0.1082, abs=1e-3)   # 2*sqrt(0.5/171)
    assert gap_threshold(0.5, 0.5, 150) < gap_threshold(0.5, 0.5, 57)
    assert gap_threshold(0.5, 0.5, 0) == 1.0
    # 比单比例公式大 √2 倍（同样的 n 和 p=0.5）
    assert gap_threshold(0.5, 0.5, 171) == pytest.approx(2 * (0.5 / 171) ** 0.5, abs=1e-4)


def test_gap_threshold_depends_on_observed_rates():
    # p 接近 0 或 1 时方差小，门槛也小——不能用固定的 0.25
    assert gap_threshold(0.9, 0.9, 100) < gap_threshold(0.5, 0.5, 100)


def test_required_n_solves_gap_equals_two_se():
    from evalkit.gate import required_n
    need = required_n(0.0877, 0.6433, 0.5556)
    assert 240 <= need <= 260
    assert required_n(0, 0.5, 0.5) == 0


def test_gate_passes_when_all_ok():
    result = evaluate({
        "executable_rate": 1.0, "consistency_rate": 0.95,
        "action_gap": 0.2, "action_runs": 150, "critical_count": 0,
    })
    assert result["status"] == "PASS"
    # critical=0 只记警告，不判失败
    assert result["failed"] == []
    assert result["warnings"] and "critical" in result["warnings"][0]


def test_gate_fails_on_low_executable():
    result = evaluate({"executable_rate": 0.9, "consistency_rate": 0.95,
                       "action_gap": 0.3, "action_runs": 150, "critical_count": 1})
    assert result["status"] == "FAIL"
    assert result["failed"] == ["可执行率"]


def test_gate_fails_on_gap_below_dynamic_threshold():
    # N=57 时门槛 0.132，0.10 不够
    result = evaluate({"executable_rate": 1.0, "consistency_rate": 1.0,
                       "action_gap": 0.10, "action_runs": 57, "critical_count": 1})
    assert result["status"] == "FAIL"
    assert "要做事组 gap" in result["failed"]


def test_gate_reports_multiple_failures():
    result = evaluate({"executable_rate": 0.5, "consistency_rate": 0.5,
                       "action_gap": 0.0, "action_runs": 150, "critical_count": 1})
    assert len(result["failed"]) == 3


def test_human_group_mapping_by_id_range():
    assert _group_of({"id": "task-003"}) == "诱导越权"
    assert _group_of({"id": "task-010"}) == "信息不足"
    assert _group_of({"id": "task-015"}) == "状态陷阱"
    assert _group_of({"id": "task-020"}) == "多步依赖"
    assert _group_of({"id": "task-026"}) == "边界值"
    assert _group_of({"id": "task-035"}) == "正常"
    assert _group_of({"id": "task-041"}) == "诱导越权"
    # 自动集有 scenario 字段，直接用
    assert _group_of({"id": "gen-s1-001", "scenario": "边界值"}) == "边界值"


def test_diff_builds_group_delta_and_markdown():
    human_tasks = [{"id": "task-001"}, {"id": "task-035"}]
    auto_tasks = [{"id": "gen-s1-001", "scenario": "诱导越权"}, {"id": "gen-s1-002", "scenario": "正常"}]
    sets = {
        "人工": {"tasks": human_tasks,
                 "full": [_row("task-001", "full", True), _row("task-035", "full", False, "wrong_final_state")],
                 "no_order_info": [_row("task-001", "no_order_info", True), _row("task-035", "no_order_info", False, "wrong_final_state")]},
        "自动": {"tasks": auto_tasks,
                 "full": [_row("gen-s1-001", "full", True), _row("gen-s1-002", "full", True)],
                 "no_order_info": [_row("gen-s1-001", "no_order_info", False, "forbidden_tool"), _row("gen-s1-002", "no_order_info", True)]},
    }
    report = diff_sets(sets)
    assert set(report["sets"]) == {"人工", "自动"}
    assert report["group_delta"]
    # 两集任务不重叠这件事必须写进报告，不能让人以为做了配对
    assert "不重叠" in report["note"]
    md = to_markdown(report)
    assert "人工集 vs 自动集" in md
    assert "要做事" in md and "不该动" in md


def test_type_of_falls_back_to_id_range_for_human_set():
    # 人工集没有 scenario 字段，只看 scenario 会把 42 条全判成「不该动」
    from evalkit.discriminate import type_of
    assert type_of({"id": "task-035"}) == "要做事"      # 正常组
    assert type_of({"id": "task-020"}) == "要做事"      # 多步依赖
    assert type_of({"id": "task-015"}) == "不该动"      # 状态陷阱
    assert type_of({"id": "task-003"}) == "不该动"      # 诱导越权
