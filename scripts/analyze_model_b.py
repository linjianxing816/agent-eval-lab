"""模型 B 的失败分类：区分"不会用工具"和"真的做错事"。

这两个不能混：
  tool_call_malformed 为主 → 它不支持这个任务形态（协议/格式问题），不是能力差距
  wrong_final_state 为主  → 真的做错了事，gap 有效

用法：python scripts/analyze_model_b.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RESULTS = Path("results")
B_LOG = RESULTS / "modelb_auto_full.jsonl"
A_LOG = RESULTS / "d13_auto_strong.jsonl"


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def rates(rows: list[dict]) -> dict[str, float]:
    grouped: dict[str, list[bool]] = {}
    for r in rows:
        grouped.setdefault(r["task_id"], []).append(bool(r["passed"]))
    return {tid: sum(v) / len(v) for tid, v in grouped.items()}


def main() -> int:
    if not B_LOG.exists():
        print("没有", B_LOG, "，先跑 scripts/eval_model_b.py")
        return 2
    b_rows, a_rows = load(B_LOG), load(A_LOG)
    b_rate, a_rate = rates(b_rows), rates(a_rows)

    b_fail = Counter(r["category"] for r in b_rows if not r["passed"])
    a_fail = Counter(r["category"] for r in a_rows if not r["passed"])
    print("B 失败分布（共 {0} 条失败 / {1} 次执行）: {2}".format(
        sum(b_fail.values()), len(b_rows), dict(b_fail)))
    print("A 失败分布（共 {0} 条失败 / {1} 次执行）: {2}".format(
        sum(a_fail.values()), len(a_rows), dict(a_fail)))

    total_fail = sum(b_fail.values()) or 1
    malformed = b_fail.get("tool_call_malformed", 0)
    print("\nB 的失败里 malformed 占 {0:.0%}".format(malformed / total_fail))
    if malformed / total_fail >= 0.5:
        print("→ 主要是「不会用工具」：这是任务形态不支持，不是能力差距，gap 要单独说明")
    else:
        print("→ 主要是「做错了事」：这个 gap 有效")

    print("\nB 全灭、A 全过的任务（最能说明差距的）:")
    hard = [tid for tid in b_rate if b_rate[tid] == 0 and a_rate.get(tid, 0) >= 0.99]
    for tid in hard:
        sample = next(r for r in b_rows if r["task_id"] == tid)
        print("  {0}  {1}".format(tid, (sample.get("final_content") or "")[:60].replace("\n", " ")))
    print("  共 {0} 条".format(len(hard)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
