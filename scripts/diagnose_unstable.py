"""诊断"判定不稳"的任务：把同一条任务多次执行的轨迹并排看，判断是哪一类问题。

三类（互斥，按顺序判）：

| 类型 | 判据 | 谁的问题 |
|---|---|---|
| **A 行为不稳** | 工具序列不同（有时调了写操作，有时没调） | 模型随机性 |
| **B 参数不稳** | 工具序列相同，但最终状态不同（同一类动作，参数不同） | 模型随机性（细粒度） |
| **C 判定歧义** | 工具序列相同、最终状态相同，判定却不同 | **判定层缺陷** |

A 和 B 都是"模型没个准主意"，任务本身没问题（只是单次结果不能代表能力）。
C 才是判定层的锅——同样的输入给了不同结论，说明 `judge_task` 不纯。

**只用修复之后的日志**：任务定义改过之后，旧轨迹和新轨迹不可比。
默认只扫 `results/d27_*.jsonl`（10-26 那批）。

用法：
    python scripts/diagnose_unstable.py
    python scripts/diagnose_unstable.py task-001 task-016
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

RESULTS = Path("results")
TARGETS = ("task-001", "task-016", "task-027", "task-041")
WRITE_TOOLS = ("cancel_shipment", "submit_refund", "delete_order")


def load_rows(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            row["_src"] = path.name
            rows.append(row)
    return rows


def state_key(state: dict | None) -> str:
    """把最终状态压成一行可比对的摘要：订单状态 + 退款金额。

    **不能直接哈希整个 state**：`clock` 每次一样，`policy` 也一样，
    全量哈希虽然对，但人看不出差异在哪。摘要能一眼看出"哪个订单变了"。
    """
    if not state:
        return "-"
    orders = state.get("orders") or {}
    parts = ["{0}={1}".format(oid, (orders[oid] or {}).get("status")) for oid in sorted(orders)]
    refunds = state.get("refunds") or []
    if refunds:
        parts.append("退款 " + "/".join(
            "{0}:{1:.2f}".format(r.get("order_id"), float(r.get("amount") or 0)) for r in refunds))
    return " ".join(parts)


def tool_seq(trajectory: list[dict]) -> tuple:
    return tuple(s["tool"] for s in trajectory)


def call_detail(trajectory: list[dict]) -> str:
    """一步一行：工具(关键参数)。让人一眼看出"这次退了多少"。"""
    out = []
    for step in trajectory:
        args = step.get("args") or {}
        key = args.get("order_id") or args.get("user_id") or ""
        if "amount" in args:
            key = "{0} {1}".format(key, args["amount"]).strip()
        out.append("{0}({1})".format(step["tool"], key) if key else step["tool"])
    return " → ".join(out) if out else "（一个工具都没调）"


def classify(executions: list[dict]) -> tuple[str, str]:
    """返回 (类型, 依据)。

    **先看判定是否一致，再看路径**。反过来的话，一条 19 次全失败但中间
    调换过两个工具顺序的任务会被误判成"不稳"——它其实稳得很，只是走法不同。
    """
    verdicts = {(e["passed"], e["category"]) for e in executions}
    if len(verdicts) == 1:
        passed, category = next(iter(verdicts))
        seqs = {tool_seq(e["trajectory"]) for e in executions}
        extra = ("，但工具序列有 {0} 种（结果一样，只是走法不同）".format(len(seqs))
                 if len(seqs) > 1 else "")
        return "-", "判定 {0} 次全一致（{1}）{2}".format(
            len(executions), "通过" if passed else "失败({0})".format(category), extra)

    seqs = {tool_seq(e["trajectory"]) for e in executions}
    if len(seqs) > 1:
        counts = Counter(tool_seq(e["trajectory"]) for e in executions)
        detail = "；".join("{0} 次：{1}".format(n, "→".join(s) or "无工具") for s, n in counts.most_common())
        return "A", "判定有 {0} 种、工具序列有 {1} 种（{2}）".format(len(verdicts), len(seqs), detail)

    states = {state_key(e.get("final_state")) for e in executions}
    if len(states) > 1:
        return "B", "判定有 {0} 种；工具序列相同（{1}），但最终状态有 {2} 种".format(
            len(verdicts), "→".join(next(iter(seqs))), len(states))

    return "C", "工具序列和最终状态都相同，判定却有 {0} 种".format(len(verdicts))


def main() -> None:
    targets = tuple(sys.argv[1:]) or TARGETS
    paths = sorted(RESULTS.glob("d27_*.jsonl"))
    if not paths:
        raise SystemExit("找不到 d27_*.jsonl")
    rows = load_rows(paths)
    print("扫了 {0} 个日志，{1} 条执行".format(len(paths), len(rows)))
    print("日志:", ", ".join(p.name for p in paths), "\n")

    report = {}
    for task_id in targets:
        mine = [r for r in rows if r["task_id"] == task_id]
        if not mine:
            print("=== {0} === 没有执行记录\n".format(task_id))
            continue
        print("=== {0} === {1} 次执行".format(task_id, len(mine)))
        for row in mine:
            verdict = "通过" if row["passed"] else "失败({0})".format(row["category"])
            writes = [s["tool"] for s in row["trajectory"] if s["tool"] in WRITE_TOOLS]
            print("  {0:<22} run{1}  {2:<20} {3}".format(
                row["_src"], row["run"], verdict,
                call_detail(row["trajectory"])))
            print("  {0:<22}      写操作 {1}｜状态 {2}".format(
                "", "/".join(writes) or "无", state_key(row.get("final_state"))))
        kind, why = classify(mine)
        passed_n = sum(1 for e in mine if e["passed"])
        cats = Counter(e["category"] for e in mine if not e["passed"])
        print("  → 类型 {0}：{1}".format(kind, why))
        print("     判定分布：通过 {0} / {1}｜失败原因 {2}\n".format(
            passed_n, len(mine),
            "、".join("{0}×{1}".format(k, v) for k, v in cats.most_common()) or "无"))
        report[task_id] = {
            "runs": len(mine), "type": kind, "why": why,
            "passed": passed_n,
            "failure": dict(cats),
            "sequences": len({tool_seq(e["trajectory"]) for e in mine}),
            "states": len({state_key(e.get("final_state")) for e in mine}),
            "verdicts": len({(e["passed"], e["category"]) for e in mine}),
        }

    counts = Counter(v["type"] for v in report.values())
    print("汇总：", "  ".join("{0} {1} 条".format(k, n) for k, n in sorted(counts.items())))
    (RESULTS / "unstable_diagnosis.json").write_text(
        json.dumps({"tasks": report, "summary": dict(counts),
                    "sources": [p.name for p in paths]}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print("写入 results/unstable_diagnosis.json")


if __name__ == "__main__":
    main()
