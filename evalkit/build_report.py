"""把 results/ 里的原始日志装配成 D13 的 diff 和 D14 的报告页。

一条命令：
    python -m evalkit.build_report
产出：
    results/diff.json   人工集 vs 自动集
    results/diff.md     同上，人读版
    results/report.html 双击可打开（数据全内联，无 fetch）
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from evalkit.audit import load_tasks, sample_tasks
from evalkit.diff import diff_sets, to_markdown
from evalkit.discriminate import (
    Outcome,
    compare_by_type,
    cost_of,
    summarize_agent,
    token_totals,
    type_of,
)
from evalkit.gate import evaluate
from evalkit.report import write_report

RESULTS = Path("results")
COLOR_ACTION = "#1f5fbf"
COLOR_UNCHANGED = "#c77700"


def load_outcomes(path: Path) -> list[Outcome]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        rows.append(Outcome(
            task_id=raw["task_id"], agent=raw["agent"], run=raw["run"], ok=raw["ok"],
            passed=raw["passed"], category=raw["category"], steps=raw["steps"],
            prompt_tokens=raw["prompt_tokens"], completion_tokens=raw["completion_tokens"],
            trajectory=tuple(raw.get("trajectory") or ()),
            final_state=raw.get("final_state"),
        ))
    return rows


def build_diff() -> dict[str, Any]:
    human = load_tasks("tasks/human40.jsonl")
    auto = sample_tasks(load_tasks("tasks/gen_s2.jsonl"), seed=2024)
    sets = {
        "人工集(42)": {"tasks": human,
                       "strong": load_outcomes(RESULTS / "d13b_human_strong.jsonl"),
                       "weak": load_outcomes(RESULTS / "d13b_human_weak.jsonl")},
        "自动集(30 抽样)": {"tasks": auto,
                            "strong": load_outcomes(RESULTS / "d13_auto_strong.jsonl"),
                            "weak": load_outcomes(RESULTS / "d13_auto_weak.jsonl")},
    }
    report = diff_sets(sets)
    (RESULTS / "diff.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (RESULTS / "diff.md").write_text(to_markdown(report), encoding="utf-8")
    return report


def _badge(outcome: Outcome) -> str:
    if not outcome.ok:
        return "跑崩"
    if outcome.passed:
        return "通过"
    return "严重失败" if outcome.category in ("forbidden_tool", "premature_finish",
                                              "hallucinated_result", "tool_call_malformed") else "失败"


def build_trajectories() -> list[dict[str, Any]]:
    tasks = {t["id"]: t for t in load_tasks("tasks/human40.jsonl")}
    tasks.update({t["id"]: t for t in sample_tasks(load_tasks("tasks/gen_s2.jsonl"), seed=2024)})
    rows: list[dict[str, Any]] = []
    for tag, label in (("human", "人工集"), ("auto", "自动集")):
        for agent in ("strong", "weak"):
            # 人工集用 d13b（2026-10-24 修完 5 条坏题后重跑）；自动集仍是 d13
            prefix = "d13b" if tag == "human" else "d13"
            path = RESULTS / "{0}_{1}_{2}.jsonl".format(prefix, tag, agent)
            if not path.exists():
                continue
            for outcome in load_outcomes(path):
                task = tasks.get(outcome.task_id, {})
                forbid = set((task.get("constraints") or {}).get("forbid_tools") or [])
                steps = [
                    {
                        "step": s["step"], "tool": s["tool"],
                        "args": json.dumps(s["args"], ensure_ascii=False),
                        "result": json.dumps(s["result"], ensure_ascii=False),
                        "elapsed_ms": s["elapsed_ms"], "error": s["error"],
                        "forbidden": s["tool"] in forbid,
                    }
                    for s in outcome.trajectory
                ]
                rows.append({
                    "task_id": outcome.task_id,
                    "agent": label + " / " + agent,
                    "scenario": task.get("scenario") or "人工",
                    "instruction": task.get("instruction", ""),
                    "badge": _badge(outcome),
                    "passed": outcome.passed,
                    "steps": steps,
                    "final_state": json.dumps(outcome.final_state or {}, ensure_ascii=False),
                })
    return rows


def build_distribution(strong: list[Outcome], weak: list[Outcome],
                       tasks: list[dict]) -> dict[str, Any]:
    kind = {t["id"]: type_of(t) for t in tasks}
    s_map = {o.task_id: o.passed for o in strong}
    w_map = {o.task_id: o.passed for o in weak}
    buckets = [("0.0", 0.0), ("0.25", 0.25), ("0.5", 0.5), ("0.75", 0.75), ("1.0", 1.0)]
    counts = {b: {"要做事": 0, "不该动": 0} for b, _ in buckets}
    effective = 0
    for tid, passed in s_map.items():
        if tid not in w_map:
            continue
        diff = abs(float(passed) - float(w_map[tid]))
        label = min(buckets, key=lambda b: abs(b[1] - diff))[0]
        counts[label][kind.get(tid, "要做事")] += 1
        if diff > 0:
            effective += 1
    rows = []
    for label, _ in buckets:
        for group, color in (("要做事", COLOR_ACTION), ("不该动", COLOR_UNCHANGED)):
            rows.append({"label": "{0} {1}".format(label, group), "count": counts[label][group],
                         "color": color})
    return {
        "buckets": rows,
        "note": "{0}/{1} 条任务有效（区分度 > 0）。单任务数据来自 repeat=1，"
                "误差约 ±1，只回答「多少条有效」，不用来看大小。".format(effective, len(s_map)),
    }


def main() -> None:
    diff = build_diff()

    auto_tasks = sample_tasks(load_tasks("tasks/gen_s2.jsonl"), seed=2024)
    auto_strong = load_outcomes(RESULTS / "d13_auto_strong.jsonl")
    auto_weak = load_outcomes(RESULTS / "d13_auto_weak.jsonl")
    by_type = compare_by_type(auto_strong, auto_weak, auto_tasks)

    audit = json.loads((RESULTS / "audit_s2.json").read_text(encoding="utf-8"))
    critical = sum(1 for o in auto_strong + auto_weak
                   if o.category in ("forbidden_tool", "premature_finish",
                                     "hallucinated_result", "tool_call_malformed")
                   and not o.passed and o.agent == "strong")
    gate = evaluate({
        "executable_rate": audit["executable"]["executable_rate"],
        "consistency_rate": audit["deterministic"]["consistency_rate"],
        "action_gap": by_type["要做事"]["gap"],
        "action_runs": by_type["要做事"]["runs"],
        "critical_count": critical,
    })

    matrix = []
    for name, path in (("S1", "gen_s1"), ("S2", "gen_s2")):
        tasks = load_tasks("tasks/{0}.jsonl".format(path))
        strong = load_outcomes(RESULTS / "d11_{0}_strong.jsonl".format(name))
        weak = load_outcomes(RESULTS / "d11_{0}_weak.jsonl".format(name))
        report = compare_by_type(strong, weak, tasks)
        for bucket in ("要做事", "不该动"):
            row = report[bucket]
            matrix.append({"set": name, "bucket": bucket, "strong": row["strong"],
                           "weak": row["weak"], "gap": row["gap"], "direction": row["direction"]})

    totals = token_totals(auto_strong + auto_weak)
    cost = cost_of(totals)
    cost_text = "金额暂缺（.env 未配单价）" if cost is None else "{0:.4f} 元".format(cost["total"])

    data = {
        "meta": "数据源：人工集 42 条 + 自动集 30 条抽样，strong / weak 各跑 1 次；"
                "矩阵用 S1 / S2 各 100 条（D11）。成本 {0}。".format(cost_text),
        "cards": [
            {"label": "可执行率", "value": "{0:.1%}".format(audit["executable"]["executable_rate"]),
             "how": "跑完不崩的次数 / 总次数（D8，30 条 × 3 次）"},
            {"label": "一致率", "value": "{0:.1%}".format(audit["deterministic"]["consistency_rate"]),
             "how": "同一条跑 3 次判定一致的任务数 / 判定过的任务数"},
            {"label": "要做事组 gap", "value": "{0:+.3f}".format(by_type["要做事"]["gap"]),
             "how": "p_strong − p_weak（{0} 条 / {1} 次）".format(
                 by_type["要做事"]["tasks"], by_type["要做事"]["runs"])},
            {"label": "不该动组 gap", "value": "{0:+.3f}".format(by_type["不该动"]["gap"]),
             "how": "p_weak − p_strong（{0} 条 / {1} 次），正数=弱 Agent 更保守".format(
                 by_type["不该动"]["tasks"], by_type["不该动"]["runs"])},
        ],
        "gate": gate,
        "matrix": matrix,
        "distribution": build_distribution(auto_strong, auto_weak, auto_tasks),
        "trajectories": build_trajectories(),
    }
    path = write_report(data, RESULTS / "report.html")
    print("门禁:", gate["status"], "| 未过:", gate["failed"] or "无")
    print("报告:", path)
    print("diff:", (RESULTS / "diff.md").resolve())


if __name__ == "__main__":
    main()
