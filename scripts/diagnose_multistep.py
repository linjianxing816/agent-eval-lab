"""查 A 在多步依赖上 0/12 全灭的原因：是能力问题还是判定/任务问题。

**不改判定、不改任务、不跑新数据。** 只做三件事：
  1. 把每次执行按"停在哪"分类（A 只查没改 / B 做一半 / C 全做但不对 / D 报错放弃）
  2. 对 A 类看它查了什么工具（判断是"发现冲突"还是"不知道下一步"）
  3. 算"如果 K 次算通过，A 能不能超过 B"

用法：python scripts/diagnose_multistep.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RESULTS = Path("results")
TASKS = Path("tasks/gen_s2.jsonl")
LOG_A = RESULTS / "modela_auto_full_90.jsonl"
LOG_B = RESULTS / "modelb_auto_full.jsonl"

WRITE_TOOLS = {"cancel_shipment", "submit_refund", "delete_order"}
QUERY_TOOLS = {"get_order", "get_user", "calc_refund"}
SCENARIO = "多步依赖"


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def classify(row: dict) -> tuple[str, str]:
    """返回 (类别, 说明)。

    A = 一次写操作都没做  B = 做了一半  C = 全做了但结果不对  D = 工具报错后放弃
    """
    tools = [s["tool"] for s in (row.get("trajectory") or [])]
    errors = [s for s in (row.get("trajectory") or []) if s.get("error")]
    writes = [t for t in tools if t in WRITE_TOOLS]

    if errors and not writes:
        return "D", "工具报错后放弃：{0}".format(errors[0].get("error"))
    if not writes:
        return "A", "只查了 {0} 个工具，没做任何写操作".format(len(tools))
    # 期望要改几个订单状态
    state = row.get("final_state") or {}
    orders = state.get("orders") or {}
    return "C", "做了 {0} 次写操作".format(len(writes))


def main() -> int:
    tasks = {t["id"]: t for t in load(TASKS)}
    a_rows, b_rows = load(LOG_A), load(LOG_B)
    ids = sorted({r["task_id"] for r in a_rows if tasks[r["task_id"]].get("scenario") == SCENARIO})

    print("多步依赖任务 {0} 条：{1}".format(len(ids), "、".join(ids)))
    print("=" * 78)

    tally = {"A": 0, "B": 0, "C": 0, "D": 0}
    a_query_profile: list[str] = []

    for tid in ids:
        task = tasks[tid]
        print("\n{0} | {1}".format(tid, task["instruction"]))
        print("  期望: {0}".format(json.dumps(task["success_criteria"], ensure_ascii=False)))
        for label, rows in (("A", a_rows), ("B", b_rows)):
            for r in [x for x in rows if x["task_id"] == tid]:
                kind, why = classify(r)
                if label == "A":
                    tally[kind] += 1
                    if kind == "A":
                        tools = [s["tool"] for s in r["trajectory"]]
                        profile = ("get_order only" if set(tools) == {"get_order"}
                                   else ("+".join(sorted(set(tools)))))
                        a_query_profile.append(profile)
                print("   {0} run{1} {2} {3:<6} {4:<20} {5}".format(
                    label, r["run"], "过" if r["passed"] else "败", kind, r["category"] or "-", why))
                if kind == "A":
                    print("      查了: {0}".format([s["tool"] for s in r["trajectory"]]))
                    print("      最后一句: {0}".format((r.get("final_content") or "")[:70].replace("\n", " ")))

    print("\n" + "=" * 78)
    print("A 的 12 次分类：A(只查没改) {0} / B(做一半) {1} / C(全做但不对) {2} / D(报错放弃) {3}".format(
        tally["A"], tally["B"], tally["C"], tally["D"]))
    print("A 类里查了什么:", a_query_profile)

    # K 要多大 A 才能超过 B
    n_a, n_b = len(a_rows), len([r for r in b_rows if r["category"] != "request_error"])
    pass_a = sum(1 for r in a_rows if r["passed"])
    pass_b = sum(1 for r in b_rows if r["passed"] and r["category"] != "request_error")
    print("\n现状：A {0}/{1} = {2:.3f} | B {3}/{4} = {5:.3f}".format(
        pass_a, n_a, pass_a / n_a, pass_b, n_b, pass_b / n_b))
    need = 0
    while (pass_a + need) / n_a <= pass_b / n_b:
        need += 1
    print("K 要多大 A 才超过 B：K ≥ {0}（(61+K)/90 > 66/89 → K > {1:.2f}）".format(
        need, n_b * pass_b / n_b * 90 / 89 - pass_a if False else 90 * pass_b / n_b - pass_a))
    for k in (tally["A"], tally["A"] + tally["C"]):
        print("  若 K={0}：A = {1}/90 = {2:.3f} {3}".format(
            k, pass_a + k, (pass_a + k) / n_a,
            "→ 超过 B" if (pass_a + k) / n_a > pass_b / n_b else "→ 仍低于 B"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
