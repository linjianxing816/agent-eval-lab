"""验证指数退避重试到底有没有用：重跑 B 的 4 条 `request_error`。

**这 4 条是"原批次里就失败"的样本**（d26_modelb.jsonl 里 category=request_error），
各重跑 3 次，看还报不报错：

- 不报错了 → 原来的错是**瞬时**的（网络抖动 / 上游瞬断），重试有效
- 还报错 → **持续性**问题（任务太长超时 / 模型不支持这个输入），重试没用

**要区分"重试救回的"和"本来就不报错的"**：
重跑时 `retry_events` 为空且成功，说明这次压根没出错（运气好），
不是重试的功劳。只看"没报错"会把两者混成一个数。

用法：python scripts/retry_probe.py
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evalkit.audit import load_tasks
from evalkit.client import DeepSeekClient, load_model_b_settings
from evalkit.discriminate import AGENTS, run_batch_async

RESULTS = Path("results")
OUT = RESULTS / "retry_probe.json"

# 原批次里 category=request_error 的 4 条（run 号写在下面，用来核对）
TARGETS = ("gen-s1-009", "gen-s1-038", "gen-s1-051", "gen-s1-080")
ORIGINAL_RUNS = {"gen-s1-009": 0, "gen-s1-038": 1, "gen-s1-051": 1, "gen-s1-080": 0}
REPEATS = 3
CONCURRENCY = 4


def main() -> None:
    tasks = [t for t in load_tasks("tasks/gen_s2.jsonl") if t["id"] in TARGETS]
    missing = [tid for tid in TARGETS if tid not in {t["id"] for t in tasks}]
    if missing:
        raise SystemExit("任务不存在: " + ", ".join(missing))
    tasks.sort(key=lambda t: TARGETS.index(t["id"]))

    settings = load_model_b_settings()
    print("模型 B:", settings.model, "| 任务", len(tasks), "×", REPEATS, "次 =",
          len(tasks) * REPEATS, "次执行", flush=True)

    client = DeepSeekClient(settings, thinking=False, max_tokens=1024, timeout=90.0)
    rows = []
    try:
        for task in tasks:
            before = len(client.retry_events)
            outcomes = asyncio.run(run_batch_async(
                [task], AGENTS["full"], client, repeats=REPEATS, concurrency=CONCURRENCY))
            events = client.retry_events[before:]
            errs = [o for o in outcomes if o.category == "request_error"]
            rows.append({
                "task_id": task["id"],
                "original_run": ORIGINAL_RUNS[task["id"]],
                "original_status": "request_error",
                "runs": len(outcomes),
                "request_errors": len(errs),
                "results": [("request_error" if o.category == "request_error"
                             else ("通过" if o.passed else o.category)) for o in outcomes],
                "retries": len(events),
                "retry_reasons": [e.reason for e in events],
                "statuses": [o.error_status for o in outcomes if o.error_status],
            })
            print("  {0:<14} {1}  重试 {2} 次".format(
                task["id"], " / ".join(rows[-1]["results"]), len(events)), flush=True)
    finally:
        client.close()

    rescued = [r for r in rows if r["request_errors"] == 0]
    still_broken = [r for r in rows if r["request_errors"] > 0]
    rescued_by_retry = [r for r in rescued if r["retries"] > 0]

    payload = {
        "model": settings.model,
        "repeats": REPEATS,
        "rows": rows,
        "summary": {
            "tasks": len(rows),
            "rescued": len(rescued),
            "rescued_by_retry": len(rescued_by_retry),
            "still_broken": len(still_broken),
            "total_retries": sum(r["retries"] for r in rows),
        },
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n{0:<14}{1:>10}{2:>22}{3:>8}".format("任务", "原状态", "重跑 3 次", "重试次数"))
    for r in rows:
        print("{0:<14}{1:>10}{2:>22}{3:>8}".format(
            r["task_id"], "request_error", " / ".join(r["results"]), r["retries"]))
    print("\n救回 {0}/{1}（其中靠重试救回 {2} 条），仍报错 {3} 条".format(
        len(rescued), len(rows), len(rescued_by_retry), len(still_broken)))
    print("写入", OUT)


if __name__ == "__main__":
    main()
