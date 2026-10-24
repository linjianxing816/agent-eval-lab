"""用已存轨迹重判一遍，验证判定层没漂移。纯本地，0 成本。

**注意**：`judge_task` 不读 `instruction`，所以改 instruction 不会改变任何已存轨迹的判定。
这个脚本的用途是**一致性检查**（证明除改动点外没有别的漂移），不是"改完重判就生效"。

局限：日志里没存 `final_content`，所以 `premature_finish` / `hallucinated_result`
两类复现不出来——遇到这两类直接沿用原判，并单独计数。
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # 直接 `python scripts/rejudge_check.py` 也能跑

from evalkit.agent.loop import LoopResult
from evalkit.audit import load_tasks
from evalkit.env.sandbox import TrajectoryStep
from evalkit.judge import judge_task

RESULTS = Path("results")
CANNOT_REPRODUCE = ("premature_finish", "hallucinated_result")


def rebuild(record: dict) -> tuple[list[TrajectoryStep], LoopResult]:
    steps = [
        TrajectoryStep(
            step=s["step"], tool=s["tool"], args=s["args"], result=s["result"],
            elapsed_ms=s["elapsed_ms"], error=s["error"],
            repeated=s.get("repeated", False), malformed=s.get("malformed", False),
        )
        for s in record["trajectory"]
    ]
    result = LoopResult(
        steps=record["steps"], exit_reason="", final_content="",
        tool_calls_total=len(steps), messages=[], per_step_tokens=[],
        per_step_completion_tokens=[], per_step_ms=[],
    )
    return steps, result


def main() -> None:
    tasks = {t["id"]: t for t in load_tasks("tasks/human40.jsonl")}
    tasks.update({t["id"]: t for t in load_tasks("tasks/gen_s2.jsonl")})

    # 只取带 trajectory 的那几批（d10/d11 是早期日志，没有轨迹，重判不了）
    files = sorted(f for f in RESULTS.glob("d1*_*.jsonl")
                   if f.name.split("_")[0] in ("d13", "d13b", "d15", "d16"))
    total = same = 0
    skipped = 0
    diffs: list[tuple[str, str, str, str, str]] = []
    for path in files:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            task = tasks.get(record["task_id"])
            if task is None:
                continue
            total += 1
            if record["category"] in CANNOT_REPRODUCE:
                skipped += 1
                continue
            steps, result = rebuild(record)
            verdict = judge_task(task, record["final_state"], result, steps)
            if (bool(verdict["passed"]), verdict["category"]) == (record["passed"], record["category"]):
                same += 1
            else:
                diffs.append((path.name, record["task_id"], record["agent"],
                              str(record["category"]), str(verdict["category"])))

    checked = total - skipped
    print("日志文件:", len(files), "| 记录:", total)
    print("可复现判定:", checked, "| 一致:", same, "| 不一致:", len(diffs))
    print("跳过（缺 final_content，无法复现）:", skipped,
          "→", dict(Counter(r["category"] for r in
                            (json.loads(x) for f in files for x in
                             f.read_text(encoding="utf-8").splitlines() if x.strip())
                            if r["category"] in CANNOT_REPRODUCE)))
    if diffs:
        print("\n不一致明细（文件 / 任务 / Agent / 原判 / 重判）:")
        for row in diffs[:20]:
            print("  ", " | ".join(row))
    print("\n一致性: {0:.1%}".format(same / checked if checked else 0.0))


if __name__ == "__main__":
    main()
