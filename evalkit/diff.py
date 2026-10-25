"""D13 diff：人工集 vs 自动集的判定差异。

【先说清一件事：两集的任务不重叠，所以做不了"同一条任务"的对比】

人工集 42 条是手写的，自动集 100 条是模型生成的，id 和 instruction 都不一样。
所以这里做的是**类别级对照**：同一类别下，人工集和自动集的通过率差多少。
任务级明细两集各自给一份，不做跨集配对。

**能回答的问题**：
- 自动集整体是不是比人工集难（通过率低）？
- 哪个类别上自动集明显更难/更容易？
- 两个集的失败原因分布差在哪？

**不能回答的问题**：
- "某一条任务"在人工集和自动集上表现不同（不存在同一条）
"""
from __future__ import annotations

from typing import Any

from evalkit.discriminate import Outcome, compare_by_type, group_of, summarize_agent

UNCHANGED_GROUPS = ("诱导越权", "信息不足", "状态陷阱")
_group_of = group_of      # 同一套映射，别再实现第二份


def by_group(outcomes: list[Outcome], tasks: list[dict]) -> dict[str, dict[str, Any]]:
    kind = {t["id"]: _group_of(t) for t in tasks}
    buckets: dict[str, list[Outcome]] = {}
    for row in outcomes:
        buckets.setdefault(kind.get(row.task_id, "?"), []).append(row)
    return {
        group: {
            "runs": len(rows),
            "pass_rate": round(sum(1 for r in rows if r.passed) / len(rows), 4),
            "avg_steps": round(sum(r.steps for r in rows) / len(rows), 2),
        }
        for group, rows in sorted(buckets.items())
    }


def diff_sets(sets: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """sets: {集名: {"tasks": [...], "full": [...], "no_order_info": [...]}}"""
    report: dict[str, Any] = {"sets": {}, "by_group": {}, "group_delta": [], "note": ""}
    for name, payload in sets.items():
        tasks, strong, weak = payload["tasks"], payload["full"], payload["no_order_info"]
        report["sets"][name] = {
            "tasks": len(tasks),
            "full": summarize_agent(strong),
            "no_order_info": summarize_agent(weak),
            "by_type": compare_by_type(strong, weak, tasks),
        }
        report["by_group"][name] = by_group(strong, tasks)

    names = list(sets)
    if len(names) == 2:
        left, right = names
        groups = sorted(set(report["by_group"][left]) | set(report["by_group"][right]))
        for group in groups:
            a = report["by_group"][left].get(group, {})
            b = report["by_group"][right].get(group, {})
            if not a or not b:
                continue
            report["group_delta"].append({
                "group": group,
                "left": a["pass_rate"],
                "right": b["pass_rate"],
                "delta": round(a["pass_rate"] - b["pass_rate"], 4),
                "left_name": left,
                "right_name": right,
            })
        report["group_delta"].sort(key=lambda row: -abs(row["delta"]))
        report["note"] = (
            "两集任务不重叠（人工手写 vs 模型生成，id 和 instruction 都不同），"
            "所以做不了任务级配对对比。这里给的是类别级对照：同一类别下两集的通过率差。"
        )
    return report


def to_markdown(report: dict[str, Any]) -> str:
    lines = ["# 人工集 vs 自动集 判定差异", ""]
    if report.get("note"):
        lines += ["> " + report["note"], ""]

    lines += ["## 整集", "", "| 集 | 任务数 | strong 通过率 | weak 通过率 | 平均步数(strong) |", "|---|---|---|---|---|"]
    for name, info in report["sets"].items():
        lines.append("| {0} | {1} | {2:.3f} | {3:.3f} | {4} |".format(
            name, info["tasks"], info["full"]["pass_rate"],
            info["no_order_info"]["pass_rate"], info["full"]["avg_steps"]))
    lines.append("")

    lines += ["## 按任务类型（契约第七节口径）", "", "| 集 | 组 | strong | weak | gap | 方向 |", "|---|---|---|---|---|---|"]
    for name, info in report["sets"].items():
        for bucket in ("要做事", "不该动"):
            row = info["by_type"][bucket]
            lines.append("| {0} | {1} | {2:.3f} | {3:.3f} | {4:+.3f} | {5} |".format(
                name, bucket, row["full"], row["no_order_info"], row["gap"], row["direction"]))
    lines.append("")

    if report["group_delta"]:
        left = report["group_delta"][0]["left_name"]
        right = report["group_delta"][0]["right_name"]
        lines += ["## 类别级对照（{0} − {1}）".format(left, right), "",
                  "| 类别 | {0} | {1} | 差 |".format(left, right), "|---|---|---|---|"]
        for row in report["group_delta"]:
            lines.append("| {0} | {1:.3f} | {2:.3f} | {3:+.3f} |".format(
                row["group"], row["left"], row["right"], row["delta"]))
        lines.append("")

    lines += ["## 失败原因分布", "", "| 集 | Agent | 分布 |", "|---|---|---|"]
    for name, info in report["sets"].items():
        for agent in ("full", "no_order_info"):
            lines.append("| {0} | {1} | {2} |".format(
                name, agent, info[agent]["failure_distribution"]))
    lines.append("")
    return "\n".join(lines)
