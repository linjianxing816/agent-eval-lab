"""第二个模型的评测：同一批任务、同一套配置，只换模型。

**任务集必须是 A 跑过的那 30 条**（`sample_tasks(gen_s2, seed=2024)`），
否则"B 比 A 差"可能只是"这 30 条比那 57 条难"。所以这里直接复用 D13 的抽样函数。

用法：
    python scripts/eval_model_b.py          # 30 条 × 1 次 ≈ 5 分钟
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evalkit.audit import load_tasks, sample_tasks
from evalkit.client import DeepSeekClient, MissingConfigError, load_model_b_settings
from evalkit.discriminate import AGENTS, run_batch, summarize_agent

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
    print("模型:", settings.model, "| 任务:", len(tasks), "条 × 1 次", flush=True)
    client = DeepSeekClient(settings, thinking=False, max_tokens=1024, timeout=90.0)
    outcomes = run_batch(tasks, AGENTS["full"], client, repeats=1, log=LOG)
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
        print("\n差: 通过率 {0:+.3f}（B − A）".format(mine["pass_rate"] - base_pass))
        print("结论口径：差 > 0.10 说明这套题能分出这两个模型；差 < 0.05 说明分不出（或两个模型确实接近）。")
    else:
        print("\n没有找到 A 的基线", BASELINE, "，只能报 B 的绝对值")
    return 0


if __name__ == "__main__":
    sys.exit(main())
