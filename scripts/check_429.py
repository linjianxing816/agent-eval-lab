"""426 次压测：只跑 32 并发，回答两个问题——13 波下加速比多少、429 阈值在哪。

**为什么不跑串行 420 次**：串行 420 次要 24 分钟（420 × 3.4s），只为拿一个分母，太贵。
加速比的定义是「并发吞吐 ÷ 串行吞吐」，所以分母只需要**吞吐**，不需要跑满同样的次数。
串行跑 30 次（约 100 秒）就够把吞吐测准（±5%），而且必须是**当下**测的——
两小时前的串行吞吐可能已经因为服务端负载变了。

**为什么用吞吐比而不是墙钟比**：墙钟比会被波数影响（见踩坑 #25），
吞吐比不会——吞吐是稳态指标，只要波数够（≥3），两种算法结果一样。

用法：
    python scripts/check_429.py            # 420 次 @32 + 30 次 @1
    python scripts/check_429.py --runs 600 --level 32
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evalkit.audit import load_tasks, sample_tasks
from evalkit.client import DeepSeekClient, load_settings
from evalkit.discriminate import AGENTS, pass_rate, run_batch_async, within_task_consistency

RESULTS = Path("results")
OUT = RESULTS / "concurrency_426.json"
TASKS = 30           # 和 90 次那档同一个抽样集，保证可比
REPEATS = 14         # 30 × 14 = 420 次，波数 420/32 = 13.1
LEVEL = 32
SERIAL_RUNS = 30     # 串行基线只测吞吐，不用跑满


def run_once(level: int, repeats: int, tasks: list[dict], settings, tag: str) -> dict:
    timings: dict = {}
    client = DeepSeekClient(settings, thinking=False, max_tokens=1024, timeout=120.0)
    started = time.perf_counter()
    try:
        outcomes = asyncio.run(run_batch_async(
            tasks, AGENTS["full"], client, repeats=repeats,
            concurrency=level, timings=timings, threads=max(level, 8)))
    finally:
        client.close()
    wall = time.perf_counter() - started
    errors = [o for o in outcomes if not o.ok]
    log = RESULTS / "d27_{0}.jsonl".format(tag)
    log.write_text("\n".join(json.dumps(o.__dict__, ensure_ascii=False) for o in outcomes) + "\n",
                   encoding="utf-8")
    return {
        "level": level, "runs": len(outcomes), "wall_s": round(wall, 2),
        "throughput": round(len(outcomes) / wall, 4) if wall else 0.0,
        "status_429": sum(1 for o in outcomes if o.error_status == 429),
        "request_error": len(errors),
        "other_status": sorted({o.error_status for o in outcomes
                                if o.error_status and o.error_status != 429}),
        "latency_mean_ms": round(sum(timings.values()) / len(timings), 1) if timings else 0.0,
        "pass_rate": pass_rate(outcomes),
        "within": within_task_consistency(outcomes),
        "waves": round(len(outcomes) / level, 2),
        # 步数 = LLM 往返次数。429 是按"请求"限流的，不是按"执行"，
        # 所以报告里要说清这一档到底发了多少个请求。
        "steps_total": sum(o.steps for o in outcomes),
        "steps_mean": round(sum(o.steps for o in outcomes) / len(outcomes), 2) if outcomes else 0.0,
    }


def main() -> None:
    runs, level = REPEATS, LEVEL
    for arg in sys.argv[1:]:
        if arg.startswith("--runs="):
            runs = int(arg.split("=", 1)[1])
        elif arg.startswith("--level="):
            level = int(arg.split("=", 1)[1])

    tasks = sample_tasks(load_tasks("tasks/gen_s2.jsonl"), seed=2024)[:TASKS]
    settings = dataclasses.replace(load_settings(), model="deepseek-flash")
    total = len(tasks) * runs
    print("压测：{0} 条 × {1} 次 = {2} 次执行 @ 并发 {3}（{4:.1f} 波）".format(
        len(tasks), runs, total, level, total / level), flush=True)

    serial = run_once(1, 1, tasks, settings, "serial30")
    print("  串行基线 30 次: {0:.1f}s  吞吐 {1:.3f} 次/s  单次 {2:.0f}ms".format(
        serial["wall_s"], serial["throughput"], serial["latency_mean_ms"]), flush=True)

    burst = run_once(level, runs, tasks, settings, "burst{0}".format(level))
    print("  并发 {0} × {1} 次: {2:.1f}s  吞吐 {3:.3f} 次/s  429×{4}  req_err×{5}  单次 {6:.0f}ms".format(
        level, total, burst["wall_s"], burst["throughput"], burst["status_429"],
        burst["request_error"], burst["latency_mean_ms"]), flush=True)

    # 吞吐比：不受波数影响，和墙钟比在波数够时等价
    speedup = round(burst["throughput"] / serial["throughput"], 2) if serial["throughput"] else 0.0
    # 墙钟比：假设串行跑满 420 次要多久（用当下吞吐推算）
    serial_projected = round(total / serial["throughput"], 1) if serial["throughput"] else 0.0
    wall_speedup = round(serial_projected / burst["wall_s"], 2) if burst["wall_s"] else 0.0

    payload = {
        "tasks": [t["id"] for t in tasks],
        "repeats": runs,
        "executions": total,
        "level": level,
        "serial": serial,
        "burst": burst,
        "speedup_by_throughput": speedup,
        "serial_projected_s": serial_projected,
        "speedup_by_wall": wall_speedup,
        "measured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": ("加速比用吞吐比（并发吞吐 ÷ 串行吞吐），不受波数影响；"
                 "墙钟比那一列是按串行吞吐推算 420 次的耗时，两者在波数够时应该一致。"),
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n加速比（吞吐比）= {0:.2f}×".format(speedup))
    print("加速比（墙钟比）= {0:.2f}×（串行推算 {1:.0f}s ÷ 实测 {2:.1f}s）".format(
        wall_speedup, serial_projected, burst["wall_s"]))
    print("429 = {0} 次；request_error = {1} 次".format(burst["status_429"], burst["request_error"]))
    print("写入", OUT)


if __name__ == "__main__":
    main()
