"""D4 指标统计：纯函数。输入 list[TaskResult]，输出 dict。

**不碰 IO、不调工具、不调 LLM** —— 所以能手算对账。这一条是 D4 验收的全部意义：
程序算出来的数和手算一致，后面所有报告里的数字才可信。

【四条口径，写错就出假数字】
1. **分母一律 `total`**，不许剔除任何样本。剔掉失败请求就是作弊。
2. **步数必须给区间**，只给均值算错误 —— 同一任务跑 5 次，`steps` 在 4 和 5 之间跳
   （见踩坑记录 #9），只报 4.4 会让读者以为它稳定。
3. **成功任务和失败任务的步数分开统计** —— 混在一起看不出"失败是不是因为走太多步"。
4. **空列表返回全 0**，不报错。

【红线 / 黄线在这里也要分开】
`critical_failed` 只数红线（`forbidden_tool` / `premature_finish` / `hallucinated_result` /
`tool_call_malformed`）；`loop_rate` / `step_limit_rate` 数黄线，它们**不进**
`failure_distribution`——因为黄线的任务 `passed=True`，本来就不算失败。

【10 类失败里，哪几类目前产生不了（看到 0 别当成"没发生"）】
| 类别 | 现状 | 什么时候接上 |
|---|---|---|
| `request_error` | **产生不了** | D12 批量跑的时候，runner 接住 `ChatError` |
| `rate_limited` | **产生不了** | 同上（429 单独一类） |
| `wrong_tool` | **产生不了** | D5 模板定了之后——"调了无关工具"要先有任务语义才能判 |
| 其余 7 类 | 已能产生 | — |

所以报告里这 3 个键恒为 0 时，必须写"未接入"，不能写"未发生"。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# 契约里定死的 10 类。没出现的记 0，不要漏键——漏了图表的图例就会跳。
FAILURE_CATEGORIES: tuple[str, ...] = (
    "request_error",
    "rate_limited",
    "tool_call_malformed",
    "forbidden_tool",
    "wrong_tool",
    "step_limit_exceeded",
    "loop_detected",
    "premature_finish",
    "wrong_final_state",
    "hallucinated_result",
)
RED_CATEGORIES = ("forbidden_tool", "premature_finish", "hallucinated_result", "tool_call_malformed")
YELLOW_CATEGORIES = ("loop_detected", "step_limit_exceeded")


@dataclass(frozen=True)
class TaskResult:
    """一个任务的判定结果 + 效率数字。runner（D12）负责从 judge 结果拼出来。"""

    task_id: str
    passed: bool
    category: str | None = None
    severity: str = "passed"
    warnings: tuple[str, ...] = ()
    source: str = "generated"          # human / generated，契约要求能按 source 分组
    steps: int = 0
    tool_rounds: int = 0
    tool_calls_total: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_time_ms: float = 0.0
    tool_time_ms: float = 0.0
    wall_time_ms: float = 0.0


def from_verdict(task_id: str, verdict: dict, source: str = "generated") -> TaskResult:
    """把 `judge_task()` 的输出摊平成 TaskResult。"""
    eff = verdict.get("efficiency") or {}
    return TaskResult(
        task_id=task_id,
        passed=bool(verdict.get("passed")),
        category=verdict.get("category"),
        severity=str(verdict.get("severity") or "passed"),
        warnings=tuple(verdict.get("warnings") or ()),
        source=source,
        steps=int(eff.get("steps", 0)),
        tool_rounds=int(eff.get("tool_rounds", 0)),
        tool_calls_total=int(eff.get("tool_calls_total", 0)),
        prompt_tokens=int(eff.get("prompt_tokens", 0)),
        completion_tokens=int(eff.get("completion_tokens", 0)),
        llm_time_ms=float(eff.get("llm_time_ms", 0.0)),
        tool_time_ms=float(eff.get("tool_time_ms", 0.0)),
        wall_time_ms=float(eff.get("wall_time_ms", 0.0)),
    )


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _rate(count: int, total: int) -> float:
    """分母永远是 total。空列表时返回 0.0，不是 ZeroDivisionError。"""
    return round(count / total, 4) if total else 0.0


def _step_range(rows: list[TaskResult]) -> dict:
    if not rows:
        return {"min": 0, "max": 0, "mean": 0.0}
    steps = [row.steps for row in rows]
    return {"min": min(steps), "max": max(steps), "mean": _mean([float(s) for s in steps])}


def compute(results: list[TaskResult], price: dict | None = None) -> dict[str, Any]:
    """汇总所有指标。`price` 形如 {"input_per_1m": 0.15, "output_per_1m": 0.6}。

    单价**从外部传**，不写死在代码里——写死的话改价就得改代码，而且会和 results/ 里的
    历史记录对不上。没传单价时 `cost_per_task` 返回 None（暂缺），不填 0 冒充。
    """
    total = len(results)
    passed_rows = [r for r in results if r.passed]
    failed_rows = [r for r in results if not r.passed]

    def has(cat: str, rows: list[TaskResult]) -> int:
        return sum(1 for r in rows if r.category == cat or cat in r.warnings)

    distribution = {cat: 0 for cat in FAILURE_CATEGORIES}
    for row in failed_rows:
        if row.category in distribution:
            distribution[row.category] += 1

    prompt = [r.prompt_tokens for r in results]
    completion = [r.completion_tokens for r in results]
    llm_ms = [r.llm_time_ms for r in results]
    tool_ms = [r.tool_time_ms for r in results]
    wall = sum(r.wall_time_ms for r in results)

    cost_per_task = None
    if price is not None:
        unit_in = float(price.get("input_per_1m", 0.0))
        unit_out = float(price.get("output_per_1m", 0.0))
        per_task = [
            r.prompt_tokens / 1_000_000 * unit_in + r.completion_tokens / 1_000_000 * unit_out
            for r in results
        ]
        cost_per_task = round(sum(per_task) / total, 8) if total else None

    return {
        "total": total,
        "passed": len(passed_rows),
        "failed": len(failed_rows),
        "critical_failed": sum(1 for r in results if not r.passed and r.category in RED_CATEGORIES),
        "pass_rate": _rate(len(passed_rows), total),
        "critical_fail_rate": _rate(
            sum(1 for r in results if not r.passed and r.category in RED_CATEGORIES), total
        ),
        "forbidden_tool_rate": _rate(has("forbidden_tool", results), total),
        "loop_rate": _rate(has("loop_detected", results), total),
        "step_limit_rate": _rate(has("step_limit_exceeded", results), total),
        "premature_finish_rate": _rate(has("premature_finish", results), total),
        "steps_min": _step_range(results)["min"],
        "steps_max": _step_range(results)["max"],
        "steps_mean": _step_range(results)["mean"],
        "passed_steps": _step_range(passed_rows),
        "failed_steps": _step_range(failed_rows),
        "tool_rounds_mean": _mean([float(r.tool_rounds) for r in results]),
        "tool_calls_total_mean": _mean([float(r.tool_calls_total) for r in results]),
        "prompt_tokens_mean": _mean([float(x) for x in prompt]),
        "completion_tokens_mean": _mean([float(x) for x in completion]),
        "total_tokens_mean": _mean([float(a + b) for a, b in zip(prompt, completion)]),
        "llm_time_ms_mean": _mean(llm_ms),
        "tool_time_ms_mean": _mean(tool_ms),
        "llm_time_ratio": round(sum(llm_ms) / wall, 4) if wall else 0.0,
        "cost_per_task": cost_per_task,
        "failure_distribution": distribution,
    }
