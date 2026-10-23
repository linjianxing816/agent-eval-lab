"""D4 指标：手算对账。

5 条假样本覆盖：通过 / 通过但有黄线 / 结果错 / 越权 + 结果错 / 什么都没做。
**先手算，再和程序比。对不上就是程序错，不许改手算值去迁就程序。**

手算过程（分母一律 5）：
  total=5  passed=2  failed=3  critical=1（只有越权那条有红线）
  pass_rate=2/5=0.4   critical_fail_rate=1/5=0.2   forbidden_tool_rate=1/5=0.2
  loop_rate=1/5=0.2   step_limit_rate=0   premature_finish_rate=0
  steps: [4,6,3,5,2] → min=2 max=6 mean=20/5=4.0
  成功步数: [4,6] → min=4 max=6 mean=5.0
  失败步数: [3,5,2] → min=2 max=5 mean=10/3≈3.3333
  tool_rounds_mean=(3+5+3+4+2)/5=17/5=3.4
  tool_calls_total_mean=(4+7+3+5+2)/5=21/5=4.2
  prompt_tokens_mean=5700/5=1140.0   completion=570/5=114.0   total=1254.0
  llm_time_mean=10800/5=2160.0   tool_time_mean=5/5=1.0
  llm_time_ratio=10800/10805≈0.9995
  cost（单价 in=0.15 out=0.60 / 1M）: 每条 = prompt/1e6*0.15 + completion/1e6*0.6
      0.00021 + 0.00042 + 0.000189 + 0.000252 + 0.000126 = 0.001197 → 均值 0.0002394
"""
from __future__ import annotations

import pytest

from evalkit.metrics import FAILURE_CATEGORIES, TaskResult, compute, from_verdict

SAMPLES = [
    TaskResult("t1", passed=True, steps=4, tool_rounds=3, tool_calls_total=4,
               prompt_tokens=1000, completion_tokens=100, llm_time_ms=2000, tool_time_ms=1, wall_time_ms=2001),
    TaskResult("t2", passed=True, warnings=("loop_detected",), steps=6, tool_rounds=5, tool_calls_total=7,
               prompt_tokens=2000, completion_tokens=200, llm_time_ms=4000, tool_time_ms=2, wall_time_ms=4002),
    TaskResult("t3", passed=False, category="wrong_final_state", severity="failed", steps=3, tool_rounds=3,
               tool_calls_total=3, prompt_tokens=900, completion_tokens=90,
               llm_time_ms=1500, tool_time_ms=1, wall_time_ms=1501),
    TaskResult("t4", passed=False, category="forbidden_tool", severity="severe", steps=5, tool_rounds=4,
               tool_calls_total=5, prompt_tokens=1200, completion_tokens=120,
               llm_time_ms=2500, tool_time_ms=1, wall_time_ms=2501),
    TaskResult("t5", passed=False, category="wrong_final_state", severity="failed", steps=2, tool_rounds=2,
               tool_calls_total=2, prompt_tokens=600, completion_tokens=60,
               llm_time_ms=800, tool_time_ms=0, wall_time_ms=800),
]
PRICE = {"input_per_1m": 0.15, "output_per_1m": 0.60}


@pytest.fixture()
def stats():
    return compute(SAMPLES, price=PRICE)


def test_counts(stats):
    assert stats["total"] == 5
    assert stats["passed"] == 2
    assert stats["failed"] == 3
    assert stats["critical_failed"] == 1


def test_rates(stats):
    assert stats["pass_rate"] == 0.4
    assert stats["critical_fail_rate"] == 0.2
    assert stats["forbidden_tool_rate"] == 0.2
    assert stats["loop_rate"] == 0.2
    assert stats["step_limit_rate"] == 0.0
    assert stats["premature_finish_rate"] == 0.0


def test_steps_need_range_not_just_mean(stats):
    assert stats["steps_min"] == 2
    assert stats["steps_max"] == 6
    assert stats["steps_mean"] == 4.0


def test_passed_and_failed_steps_are_split(stats):
    assert stats["passed_steps"] == {"min": 4, "max": 6, "mean": 5.0}
    assert stats["failed_steps"]["min"] == 2
    assert stats["failed_steps"]["max"] == 5
    assert stats["failed_steps"]["mean"] == pytest.approx(3.3333, abs=1e-4)


def test_tool_means(stats):
    assert stats["tool_rounds_mean"] == 3.4
    assert stats["tool_calls_total_mean"] == 4.2


def test_tokens_and_time(stats):
    assert stats["prompt_tokens_mean"] == 1140.0
    assert stats["completion_tokens_mean"] == 114.0
    assert stats["total_tokens_mean"] == 1254.0
    assert stats["llm_time_ms_mean"] == 2160.0
    assert stats["tool_time_ms_mean"] == 1.0
    assert stats["llm_time_ratio"] == pytest.approx(0.9995, abs=1e-4)


def test_cost_per_task(stats):
    assert stats["cost_per_task"] == pytest.approx(0.0002394, abs=1e-9)


def test_failure_distribution_has_all_ten_keys(stats):
    dist = stats["failure_distribution"]
    assert set(dist) == set(FAILURE_CATEGORIES)
    assert len(dist) == 10
    assert dist["wrong_final_state"] == 2
    assert dist["forbidden_tool"] == 1
    assert dist["request_error"] == 0


def test_empty_list_returns_zeros():
    stats = compute([])
    assert stats["total"] == 0
    assert stats["pass_rate"] == 0.0
    assert stats["steps_min"] == 0 and stats["steps_max"] == 0
    assert stats["llm_time_ratio"] == 0.0
    assert stats["cost_per_task"] is None
    assert sum(stats["failure_distribution"].values()) == 0


def test_cost_is_none_without_price():
    # 没给单价就报"暂缺"，不填 0 冒充
    assert compute(SAMPLES)["cost_per_task"] is None


def test_from_verdict_flattens_judge_output():
    verdict = {
        "passed": False, "category": "forbidden_tool", "severity": "severe", "warnings": [],
        "efficiency": {"steps": 5, "tool_rounds": 4, "tool_calls_total": 5,
                       "prompt_tokens": 1200, "completion_tokens": 120,
                       "llm_time_ms": 2500.0, "tool_time_ms": 1.0, "wall_time_ms": 2501.0},
    }
    row = from_verdict("t4", verdict, source="human")
    assert row.task_id == "t4" and row.source == "human"
    assert row.steps == 5 and row.prompt_tokens == 1200 and row.wall_time_ms == 2501.0
    assert row.category == "forbidden_tool"
