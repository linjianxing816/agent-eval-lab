"""弱 Agent + 区分度测量（D10 / D11）。

【弱 Agent 怎么削弱：删掉 `get_order`】
三条原则逐条对照：

1. **绕不过去**：`get_order` 是拿到 `amount` 和 `status` 的唯一入口。
   `get_user` 只给 `name` / `vip`，`calc_refund` 必须先知道订单号且内部会读状态——
   但算出来的金额也拿不回来（它返回 refund_amount，嗯，这条要单独说，见下）。
   所以拿不到金额和状态，就没有替代路径。

2. **影响能力不是运气**：金额只能从五个值里猜（0.01 / 88.8 / 158.0 / 299.0 / 999999.0），
   猜中概率接近 0；状态不知道就没法判断该不该先取消发货。这是**信息缺失**导致的系统性错误，
   不是加温度那种随机波动。

3. **只改一处**：工具集少一个，system prompt 一个字没改。归因干净。

**已知代价**（报告里要写）：有 40% 的任务断言是"状态没变"，弱 Agent 什么都不做也能过。
所以它的通过率会被这类任务抬高，区分度被低估——这是任务设计带来的，不是测量误差。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evalkit.agent.loop import run_task
from evalkit.agent.strong import STRONG_AGENT, SYSTEM_PROMPT
from evalkit.client import ChatError, ModelClient
from evalkit.env.sandbox import Sandbox
from evalkit.env.state import EnvState
from evalkit.env.tools import TOOL_SCHEMAS
from evalkit.gate import gap_threshold
from evalkit.judge import judge_task

def _without(tool_name: str) -> list[dict]:
    return [t for t in TOOL_SCHEMAS if t["function"]["name"] != tool_name]


WEAK_AGENT: dict[str, Any] = {          # 第 1/2 轮用的：删 get_order
    "name": "weak_no_order",
    "system_prompt": SYSTEM_PROMPT,
    "tools": _without("get_order"),
}
# 第 3 轮的两个候选。删 get_order 绕得过去，因为 calc_refund 是 oracle
# （返回 refund_amount + basis），所以这轮直接打 oracle 和 VIP 信息来源。
WEAK_NO_CALC: dict[str, Any] = {
    "name": "weak_no_calc",
    "system_prompt": SYSTEM_PROMPT,
    "tools": _without("calc_refund"),
}
WEAK_NO_USER: dict[str, Any] = {
    "name": "weak_no_user",
    "system_prompt": SYSTEM_PROMPT,
    "tools": _without("get_user"),
}
# 契约第七节：区分度必须按这两组分开算，不许合并。
# 「不该动」组的正确行为是"什么都不做"，所以公式要反过来——
# 少给信息 → 更保守 → 通过率更高，用同一方向减会读反。
ACTION_GROUPS = ("多步依赖", "边界值", "正常")
UNCHANGED_GROUPS = ("诱导越权", "信息不足", "状态陷阱")


def group_of(task: dict) -> str:
    """取任务类别。

    自动集有 `scenario` 字段直接用；**人工集没有**，按 id 段推
    （1-7 越权 / 8-12 信息不足 / 13-18 状态陷阱 / 19-23 多步 / 24-28 边界 /
    29-40 正常 / 41-42 越权升级）。
    踩过：只看 `scenario` 会把人工集 42 条全判成「不该动」，要做事组通过率变成 0。
    """
    if task.get("scenario"):
        return str(task["scenario"])
    number = int(str(task["id"]).split("-")[1])
    if number <= 7 or number >= 41:
        return "诱导越权"
    if number <= 12:
        return "信息不足"
    if number <= 18:
        return "状态陷阱"
    if number <= 23:
        return "多步依赖"
    if number <= 28:
        return "边界值"
    return "正常"


def type_of(task: dict) -> str:
    return "要做事" if group_of(task) in ACTION_GROUPS else "不该动"


AGENTS = {
    "strong": STRONG_AGENT,
    "weak": WEAK_AGENT,
    "weak_no_calc": WEAK_NO_CALC,
    "weak_no_user": WEAK_NO_USER,
}


@dataclass(frozen=True)
class Outcome:
    task_id: str
    agent: str
    run: int
    ok: bool
    passed: bool
    category: str | None
    steps: int
    prompt_tokens: int
    completion_tokens: int
    # D14 轨迹查看器用。默认空，不影响已有调用。
    trajectory: tuple = ()
    final_state: dict | None = None

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def run_one(task: dict, agent: dict, client: ModelClient, run_index: int) -> Outcome:
    state = EnvState(task.get("initial_state"))
    sandbox = Sandbox(state)
    max_steps = int((task.get("constraints") or {}).get("max_steps") or 12)
    try:
        result = run_task(
            client, sandbox,
            instruction=task["instruction"],
            system_prompt=agent["system_prompt"],
            tools=agent["tools"],
            max_steps=max_steps,
        )
        verdict = judge_task(task, state.data, result, sandbox.trajectory())
    except ChatError as exc:
        return Outcome(task["id"], agent["name"], run_index, False, False, "request_error", 0, 0, 0)
    except Exception:                              # noqa: BLE001 - 崩了要记，不能中断整批
        return Outcome(task["id"], agent["name"], run_index, False, False, "crash", 0, 0, 0)
    return Outcome(
        task_id=task["id"], agent=agent["name"], run=run_index, ok=True,
        passed=bool(verdict["passed"]), category=verdict.get("category"),
        steps=result.steps,
        prompt_tokens=sum(result.per_step_tokens),
        completion_tokens=sum(result.per_step_completion_tokens),
        trajectory=tuple(step_as_dict(step) for step in sandbox.trajectory()),
        final_state=state.data,
    )


def step_as_dict(step) -> dict:
    """把 TrajectoryStep 摊平成可 JSON 的 dict。"""
    return {
        "step": step.step,
        "tool": step.tool,
        "args": step.args,
        "result": step.result,
        "error": step.error,
        "elapsed_ms": round(step.elapsed_ms, 1),
        "repeated": step.repeated,
        "malformed": step.malformed,
    }


def run_batch(tasks: list[dict], agent: dict, client: ModelClient,
              repeats: int = 1, log: Path | None = None) -> list[Outcome]:
    """串行跑。崩一条不中断，记 ok=False。"""
    outcomes: list[Outcome] = []
    for index, task in enumerate(tasks, 1):
        for run_index in range(repeats):
            outcomes.append(run_one(task, agent, client, run_index))
        if log is not None:
            log.write_text(
                "\n".join(json.dumps(o.__dict__, ensure_ascii=False) for o in outcomes) + "\n",
                encoding="utf-8",
            )
    return outcomes


def token_totals(outcomes: list[Outcome]) -> dict[str, int]:
    return {
        "runs": len(outcomes),
        "input": sum(o.prompt_tokens for o in outcomes),
        "output": sum(o.completion_tokens for o in outcomes),
    }


def token_mean(outcomes: list[Outcome]) -> dict[str, int]:
    """单任务 token 平均值，取整。报告只报这三个数（金额不报，见 cost_of）。"""
    if not outcomes:
        return {"runs": 0, "input": 0, "output": 0, "total": 0}
    n = len(outcomes)
    pin = sum(o.prompt_tokens for o in outcomes)
    pout = sum(o.completion_tokens for o in outcomes)
    return {"runs": n, "input": round(pin / n), "output": round(pout / n), "total": round((pin + pout) / n)}


def cost_of(totals: dict[str, int], env_path: Path | None = None) -> dict[str, Any] | None:
    """**暂不使用 —— 恒返回 None。** 2026-10-25 起报告只报 token，不报金额。

    为什么不报金额：
    1. **峰谷定价**：工作日 9-12 / 14-18 单价翻倍，同一批实验换个时段跑金额就不同
    2. **缓存命中**：未命中约 3 元/百万，命中约 0.025 元/百万，差 100 倍以上，
       而命中率取决于前面的调用顺序——同一任务跑两次都可能不一样

    所以金额**复现不出同一个数**，写进报告就是假精确。token 是精确的。
    函数和参数保留，将来要算的时候把 body 恢复即可（见 git 历史）。
    """
    return None


def pass_rate(outcomes: list[Outcome]) -> float:
    return round(sum(1 for o in outcomes if o.passed) / len(outcomes), 4) if outcomes else 0.0


def summarize_agent(outcomes: list[Outcome]) -> dict[str, Any]:
    total = len(outcomes)
    failed = [o for o in outcomes if not o.passed]
    distribution: dict[str, int] = {}
    for o in failed:
        key = o.category or ("request_error" if not o.ok else "unknown")
        distribution[key] = distribution.get(key, 0) + 1
    return {
        "runs": total,
        "passed": total - len(failed),
        "pass_rate": pass_rate(outcomes),
        "failure_distribution": dict(sorted(distribution.items(), key=lambda kv: -kv[1])),
        "avg_steps": round(sum(o.steps for o in outcomes) / total, 2) if total else 0.0,
        "tokens": sum(o.tokens for o in outcomes),
        "crashes": sum(1 for o in outcomes if not o.ok),
    }


def per_task_diff(strong: list[Outcome], weak: list[Outcome]) -> list[dict]:
    """单任务区分度。**只回答"多少条任务有效"，不设门槛**——
    repeat 小的时候单任务误差 ±0.38，比任何门槛都大，算出来是噪声。"""
    def by_task(outcomes: list[Outcome]) -> dict[str, float]:
        grouped: dict[str, list[Outcome]] = {}
        for o in outcomes:
            grouped.setdefault(o.task_id, []).append(o)
        return {tid: round(sum(1 for o in rs if o.passed) / len(rs), 4) for tid, rs in grouped.items()}

    s_map, w_map = by_task(strong), by_task(weak)
    rows = [
        {"task_id": tid, "strong": s_map[tid], "weak": w_map[tid],
         "diff": round(s_map[tid] - w_map[tid], 4)}
        for tid in s_map if tid in w_map
    ]
    rows.sort(key=lambda r: -r["diff"])
    return rows


def compare_by_type(strong: list[Outcome], weak: list[Outcome],
                    tasks: list[dict]) -> dict[str, Any]:
    """按契约第七节分两组算区分度。

    - 要做事：`p_strong - p_weak`，正数 = 弱 Agent 做不成事
    - 不该动：`p_weak - p_strong`，正数 = 弱 Agent 因为不敢动而"通过"
    """
    kind = {t["id"]: type_of(t) for t in tasks}

    def rate(rows: list[Outcome], bucket: str) -> tuple[float, int]:
        picked = [r for r in rows if kind.get(r.task_id) == bucket]
        if not picked:
            return 0.0, 0
        return round(sum(1 for r in picked if r.passed) / len(picked), 4), len(picked)

    report: dict[str, Any] = {}
    for bucket in ("要做事", "不该动"):
        s_rate, s_n = rate(strong, bucket)
        w_rate, w_n = rate(weak, bucket)
        gap = round(s_rate - w_rate, 4) if bucket == "要做事" else round(w_rate - s_rate, 4)
        # 门槛按观测数算（2 倍标准误）。|gap| 落在门槛内就是噪声，不能说"方向反了"——
        # 实测人工集要做事组 22 个观测、误差 ±0.21，-0.045 完全在噪声里。
        threshold = gap_threshold(s_n)
        if abs(gap) < threshold:
            direction = "差异在噪声内（未测出区分度，门槛 {0:.3f}）".format(threshold)
        elif bucket == "要做事":
            direction = "弱 Agent 做不成事"
        else:
            direction = "弱 Agent 更保守（不该动却动得更少）"
        report[bucket] = {
            "strong": s_rate, "weak": w_rate, "gap": gap,
            "runs": s_n, "tasks": len({r.task_id for r in strong if kind.get(r.task_id) == bucket}),
            "threshold": threshold, "direction": direction,
        }
    return report


def compare(sets: dict[str, dict[str, list[Outcome]]]) -> dict[str, Any]:
    """sets: {任务集名: {"strong": [...], "weak": [...]}}"""
    report: dict[str, Any] = {}
    for name, pair in sets.items():
        strong_sum = summarize_agent(pair["strong"])
        weak_sum = summarize_agent(pair["weak"])
        rows = per_task_diff(pair["strong"], pair["weak"])
        report[name] = {
            "strong": strong_sum,
            "weak": weak_sum,
            "gap": round(strong_sum["pass_rate"] - weak_sum["pass_rate"], 4),
            "effective_tasks": sum(1 for r in rows if r["diff"] > 0),
            "task_count": len(rows),
            "per_task": rows,
        }
    return report
