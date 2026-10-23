"""效率层：只统计，不判定。

`passed` 恒为 True —— 慢不是错。要看"慢"和"乱来"的区别，去看 `trajectory` 层。

两个容易混的口径：
- `steps`：LLM 往返次数（一步 = 一次往返，并行 tool_calls 算一步）
- `tool_rounds`：轨迹里出现过几个不同的 step，和 `steps` 在"最后一步没有工具调用"时会差 1
- `tool_calls_total`：工具被调了几次，和步数不是一个东西

`wall_time_ms` = LLM 往返耗时 + 工具执行耗时。两者分开给，因为它们的量级差三个数量级
（LLM 一次 1–2 秒，内存工具 0.01 毫秒）——混在一起看不出瓶颈在哪。
"""
from __future__ import annotations

from evalkit.agent.loop import LoopResult
from evalkit.env.sandbox import TrajectoryStep
from evalkit.judge.outcome import Verdict


def measure(result: LoopResult, trajectory: list[TrajectoryStep]) -> dict:
    llm_ms = sum(result.per_step_ms)
    tool_ms = sum(step.elapsed_ms for step in trajectory)
    prompt = sum(result.per_step_tokens)
    completion = sum(result.per_step_completion_tokens)
    return {
        "steps": result.steps,
        "tool_rounds": len({step.step for step in trajectory}),
        "tool_calls_total": result.tool_calls_total,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "llm_time_ms": round(llm_ms, 3),
        "tool_time_ms": round(tool_ms, 3),
        "wall_time_ms": round(llm_ms + tool_ms, 3),
        "avg_step_ms": round(llm_ms / len(result.per_step_ms), 3) if result.per_step_ms else 0.0,
    }


def check_efficiency(result: LoopResult, trajectory: list[TrajectoryStep]) -> Verdict:
    """占位语义：效率层永远"通过"，数字在 `measure()` 里。"""
    stats = measure(result, trajectory)
    return Verdict(
        True,
        None,
        "{0} 步 / {1} 次工具 / {2} token / {3} ms".format(
            stats["steps"], stats["tool_calls_total"], stats["total_tokens"], stats["wall_time_ms"]
        ),
    )
