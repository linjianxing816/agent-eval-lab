"""第二个模型的评测：同一批任务、同一套配置，只换模型。

**任务集必须是 A 跑过的那 30 条**（`sample_tasks(gen_s2, seed=2024)`），
否则"B 比 A 差"可能只是"这 30 条比那 57 条难"。所以这里直接复用 D13 的抽样函数。

用法：
    python scripts/eval_model_b.py          # 30 条 × 3 次 = 90 次执行 ≈ 15 分钟

**为什么是 3 次不是 1 次**：30 条 × 1 次只有 30 个观测，门槛 2×sqrt(0.5/30)=0.26——
B 比 A 差 0.24 都测不出来。3 次 = 90 个观测，门槛降到约 0.15。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evalkit.audit import load_tasks, sample_tasks
from evalkit.client import DeepSeekClient, MissingConfigError, load_model_b_settings
from evalkit.discriminate import AGENTS, run_batch, summarize_agent
from evalkit.gate import gap_threshold, gap_threshold_two, required_n

REPEATS = 3

RESULTS = Path("results")
LOG = RESULTS / "modelb_auto_full.jsonl"
BASELINE = RESULTS / "d13_auto_strong.jsonl"      # A 在同一批 30 条上的结果


def main() -> int:
    try:
        settings = load_model_b_settings()
    except MissingConfigError as exc:
        print("配置缺失:", exc)
        return 2

    tasks = sample_tasks(load_tasks("tasks/gen_s2.jsonl"), seed=2024)
    runs = len(tasks) * REPEATS
    print("模型:", settings.model, "| 任务:", len(tasks), "条 ×", REPEATS, "次 =", runs, "次执行", flush=True)
    print("门槛（p=0.5 最保守）:", round(gap_threshold(0.5, 0.5, runs), 3), flush=True)
    client = DeepSeekClient(settings, thinking=False, max_tokens=1024, timeout=90.0)
    outcomes = run_batch(tasks, AGENTS["full"], client, repeats=REPEATS, log=LOG)
    client.close()

    mine = summarize_agent(outcomes)
    print("\nB ({0}): 通过率 {1:.3f} | 平均步数 {2:.1f} | 失败 {3}".format(
        settings.model, mine["pass_rate"], mine["avg_steps"],
        json.dumps(mine["failure_distribution"], ensure_ascii=False)))

    if BASELINE.exists():
        rows = [json.loads(line) for line in BASELINE.read_text(encoding="utf-8").splitlines() if line.strip()]
        base_pass = sum(1 for r in rows if r["passed"]) / len(rows)
        base_steps = sum(r["steps"] for r in rows) / len(rows)
        print("A (deepseek-flash): 通过率 {0:.3f} | 平均步数 {1:.1f}".format(base_pass, base_steps))
        diff = mine["pass_rate"] - base_pass
        # A 只有 30 个观测（d13 每条跑 1 次），B 有 90 个——两组样本量不同，
        # 不能套等样本公式，否则门槛会算错。
        n_a, n_b = len(rows), runs
        threshold = gap_threshold_two(base_pass, n_a, mine["pass_rate"], n_b)
        print("\nA 通过率 {0:.3f}（N={1}） vs B 通过率 {2:.3f}（N={3}）".format(
            base_pass, n_a, mine["pass_rate"], n_b))
        print("差: {0:+.3f}（B − A）| 门槛 {1:.3f}（不等样本公式）".format(diff, threshold))
        print("（等样本近似门槛 {0:.3f}，仅供参考）".format(
            gap_threshold(base_pass, mine["pass_rate"], min(n_a, n_b))))
        if abs(diff) >= threshold:
            print("结论：差 {0:.3f} 超过门槛 {1:.3f} → 这套题**能**分出这两个模型".format(abs(diff), threshold))
        else:
            need = required_n(abs(diff) or 0.01, base_pass, mine["pass_rate"])
            print("结论：差 {0:.3f} 未过门槛 {1:.3f} → 分不出（要判定需要 N≈{2}）".format(
                abs(diff), threshold, need))
    else:
        print("\n没有找到 A 的基线", BASELINE, "，只能报 B 的绝对值")
    return 0


if __name__ == "__main__":
    sys.exit(main())
