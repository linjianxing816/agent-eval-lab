"""D8 质检：可执行性 + 可判定性。

【两个检测的区别】
- **可执行性**：任务能不能跑完，**不管对错**。判据是"没崩"——
  `step_limit` / `loop_detected` 都算跑完了（它们是黄线，不是崩了）。
  只有 `ChatError`（请求失败）和未捕获异常算不可执行。
- **可判定性**：同一条跑 N 次，判定是否一致。

【一致性比什么】
比 `(passed, category)` + **最终状态哈希**。
- 只比 `passed` 太松：`wrong_final_state` 和 `premature_finish` 都算失败，但性质完全不同
- 连 `detail` 一起比太紧：detail 里有路径和浮点数，措辞会变
- **不比步数**：同一任务跑 5 次步数会在 4/5 之间跳（踩坑记录 #9），比它等于自找不稳定

【为什么要 inconsistent_detail】
只报"一致率 90%"看不出问题模式。要记下"哪一次不同、不同在哪"，
才能判断是"模型随机"还是"任务本身有歧义"。
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evalkit.agent.loop import run_task
from evalkit.agent.full import FULL_AGENT
from evalkit.client import ChatError, ModelClient
from evalkit.env.sandbox import Sandbox
from evalkit.env.state import EnvState
from evalkit.judge import judge_task

# 按实际配比抽（93~100 条时的比例），不按"每类 5 条"——
# 每类 5 条会把只占 12% 的信息不足组抬到 17%，样本没代表性。
SAMPLE_PLAN: dict[str, int] = {
    "诱导越权": 5, "信息不足": 4, "状态陷阱": 5, "多步依赖": 4, "边界值": 4, "正常": 8,
}


@dataclass(frozen=True)
class RunRecord:
    task_id: str
    ok: bool                    # 跑完了没崩
    error: str | None
    passed: bool
    category: str | None
    severity: str
    state_digest: str           # 最终状态的哈希
    steps: int

    @property
    def verdict(self) -> tuple:
        """一致性比的就是这个三元组。"""
        return (self.passed, self.category, self.state_digest)


def state_digest(state: dict) -> str:
    payload = json.dumps(state, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def load_tasks(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def sample_tasks(tasks: list[dict], plan: dict[str, int] | None = None,
                 seed: int = 0) -> list[dict]:
    """按类别配额抽样。每类内部随机，种子固定所以可复现。"""
    quota = plan or SAMPLE_PLAN
    rng = random.Random(seed)
    picked: list[dict] = []
    for group, count in quota.items():
        pool = [t for t in tasks if t.get("scenario") == group]
        rng.shuffle(pool)
        picked.extend(pool[:count])
    return picked


def run_once(task: dict, client: ModelClient, max_steps: int | None = None) -> RunRecord:
    """跑一次。**不抛异常**——崩了就记 ok=False。"""
    state = EnvState(task.get("initial_state"))
    sandbox = Sandbox(state)
    steps = int((task.get("constraints") or {}).get("max_steps") or 12)
    try:
        result = run_task(
            client, sandbox,
            instruction=task["instruction"],
            system_prompt=FULL_AGENT["system_prompt"],
            tools=FULL_AGENT["tools"],
            max_steps=max_steps or steps,
        )
        verdict = judge_task(task, state.data, result, sandbox.trajectory())
    except ChatError as exc:
        return RunRecord(task["id"], False, "ChatError: {0}".format(exc), False, None, "error", "", 0)
    except Exception as exc:                      # noqa: BLE001 - 要记录而不是吞掉
        return RunRecord(task["id"], False, "{0}: {1}".format(type(exc).__name__, exc),
                         False, None, "error", "", 0)
    return RunRecord(
        task_id=task["id"],
        ok=True,
        error=None,
        passed=bool(verdict["passed"]),
        category=verdict.get("category"),
        severity=str(verdict.get("severity")),
        state_digest=state_digest(state.data),
        steps=result.steps,
    )


def audit_tasks(tasks: list[dict], client: ModelClient, repeats: int = 3) -> dict[str, Any]:
    """每条跑 `repeats` 次，同时算可执行性和可判定性。"""
    records: dict[str, list[RunRecord]] = {}
    for task in tasks:
        records[task["id"]] = [run_once(task, client) for _ in range(repeats)]

    crashed = [rid for rid, rs in records.items() if all(not r.ok for r in rs)]
    flaky = [rid for rid, rs in records.items() if any(not r.ok for r in rs) and any(r.ok for r in rs)]
    finished = [rid for rid, rs in records.items() if all(r.ok for r in rs)]

    consistent, inconsistent = [], []
    for rid, rs in records.items():
        if not all(r.ok for r in rs):
            continue
        verdicts = [r.verdict for r in rs]
        if len(set(verdicts)) == 1:
            consistent.append(rid)
            continue
        base = verdicts[0]
        diffs = [
            {"run": index + 1,
             "passed": r.passed, "category": r.category, "state_digest": r.state_digest,
             "steps": r.steps}
            for index, r in enumerate(rs) if r.verdict != base
        ]
        inconsistent.append({
            "task_id": rid,
            "verdicts": [{"passed": r.passed, "category": r.category, "state_digest": r.state_digest}
                         for r in rs],
            "diffs": diffs,
            "steps": [r.steps for r in rs],
        })

    total = len(tasks)
    judged = len(consistent) + len(inconsistent)
    return {
        "tasks": total,
        "repeats": repeats,
        "executable": {
            "all_ok": len(finished),
            "crashed": len(crashed),
            "flaky": len(flaky),
            "executable_rate": round(len(finished) / total, 4) if total else 0.0,
            "crashed_ids": crashed,
            "flaky_ids": flaky,
        },
        "deterministic": {
            "judged": judged,
            "consistent": len(consistent),
            "inconsistent": len(inconsistent),
            "consistency_rate": round(len(consistent) / judged, 4) if judged else 0.0,
        },
        "inconsistent_detail": inconsistent,
    }
