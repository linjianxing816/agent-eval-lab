"""把 A 也跑到 90 个观测（30 条 × 3 次），和 B 对齐。

为什么要补这一步：D13 的 A 基线是"30 条 × 1 次"= 30 个观测，B 是 90 个。
样本量不等时算门槛要用不等公式，而且 A 的 30 个观测本身噪声更大——
直接拿 30 观测的 A 和 90 观测的 B 比，比出来的差不干净。
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evalkit.audit import load_tasks, sample_tasks
from evalkit.client import DeepSeekClient, load_settings
from evalkit.discriminate import AGENTS, run_batch, summarize_agent
from evalkit.gate import gap_threshold_two

REPEATS = 3
LOG = Path("results/modela_auto_full_90.jsonl")


def main() -> None:
    settings = dataclasses.replace(load_settings(), model="deepseek-flash")
    tasks = sample_tasks(load_tasks("tasks/gen_s2.jsonl"), seed=2024)
    runs = len(tasks) * REPEATS
    print("A (deepseek-flash):", len(tasks), "条 ×", REPEATS, "次 =", runs, "次执行", flush=True)
    client = DeepSeekClient(settings, thinking=False, max_tokens=1024, timeout=90.0)
    outcomes = run_batch(tasks, AGENTS["full"], client, repeats=REPEATS, log=LOG)
    client.close()

    mine = summarize_agent(outcomes)
    print("\nA: 通过率 {0:.3f} | 平均步数 {1:.1f} | 失败 {2}".format(
        mine["pass_rate"], mine["avg_steps"], json.dumps(mine["failure_distribution"], ensure_ascii=False)))

    b_path = Path("results/modelb_auto_full.jsonl")
    if b_path.exists():
        rows = [json.loads(x) for x in b_path.read_text(encoding="utf-8").splitlines() if x.strip()]
        b_rate = sum(1 for r in rows if r["passed"]) / len(rows)
        gap = mine["pass_rate"] - b_rate
        threshold = gap_threshold_two(mine["pass_rate"], len(outcomes), b_rate, len(rows))
        print("B: 通过率 {0:.3f}（N={1}）".format(b_rate, len(rows)))
        print("\n差: {0:+.3f}（A − B）| 门槛 {1:.3f}（N={2} vs {3}）".format(
            gap, threshold, len(outcomes), len(rows)))
        print("结论:", "能分出" if abs(gap) >= threshold else "分不出")


if __name__ == "__main__":
    main()
