"""D27 并发调优：把跑批层改成线程池，扫 5 档，回答"最优并发是多少"。

【为什么值得做】
一次完整评测 = 142 条 × 3 次 ≈ 400 次执行，串行要 40 分钟以上。
跑批是纯 I/O 等待（等模型响应），串行时 CPU 基本空转。
加并发不改任何判定逻辑，只改"同时有多少个请求在飞"，就能把墙钟时间压下来。

【为什么必须有"串行复测"这一档】
一致性验证的坑：串行跑一遍、并发跑一遍，一致率 90%，
**这 10% 是并发引入的，还是模型本来就不稳？** 分不出来。
所以串行要跑两遍，拿"串行 vs 串行"的一致率当**噪声基线**。
并发档的一致率只有明显低于这个基线，才算并发的问题。
模型是 temperature=0，但服务端不保证逐 token 完全确定，实测基线本身就不是 100%。

【为什么每档新建一个 client】
连接池状态会被上一档污染：并发 16 跑完，池里躺着 16 条连接，
下一档并发 2 直接复用，测出来的延迟偏低。每档新建，各档条件一致。

用法：
    python scripts/concurrency_sweep.py            # 1/2/4/8/16，16 干净就再试 32
    python scripts/concurrency_sweep.py --levels 1,2,4
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
from evalkit.discriminate import (
    AGENTS,
    agreement,
    pass_rate,
    run_batch_async,
    within_task_consistency,
)
from evalkit.perf import pick_best_concurrency

RESULTS = Path("results")
OUT = RESULTS / "concurrency.json"

# 两组任务集，用途不同：
#
# `all`  = D13 抽样的 30 条（6 类各 4-8 条），**真实跑批的规模**。
#          90 次执行 / 32 并发 = 2.8 波，才撑得满并发，加速比才有意义。
# `hard` = 人工集里挑的 10 条难样本（诱导越权 / 状态陷阱 / 边界值为主）。
#          30 次执行 / 32 并发 = 0.94 波，**一波就打完了**，
#          测出来的加速比偏高——排队效应根本没出现。
HARD_TASK_IDS = (
    "task-001",   # 诱导越权
    "task-009",   # 信息不足
    "task-016",   # 状态陷阱（跳过取消发货）
    "task-021",   # 多步依赖（双单退款）
    "task-023",   # 多步依赖（双单取消发货）
    "task-027",   # 边界值（按指令金额）
    "task-029",   # 正常
    "task-035",   # 正常（VIP 90%）
    "task-041",   # 诱导越权升级（混合）
    "task-042",   # 诱导越权升级（混合）
)
REPEATS = 3
# 真实规模只扫四档：1（基线）/ 8 / 16 / 32。
# 2 和 4 在上一轮已经证明是线性的，重复测没有新信息，白花钱。
LEVELS = (1, 8, 16, 32)
MIN_WAVES = 3.0          # 低于 3 波 → 墙钟由"最慢那个任务"决定，加速比**偏低**


def load_picked(kind: str = "all") -> list[dict]:
    """取任务集。**顺序固定**——一致性验证靠下标逐条比，顺序一变就全对不上。"""
    if kind == "hard":
        index = {t["id"]: t for t in load_tasks("tasks/human40.jsonl")}
        missing = [tid for tid in HARD_TASK_IDS if tid not in index]
        if missing:
            raise SystemExit("任务不存在: " + ", ".join(missing))
        return [index[tid] for tid in HARD_TASK_IDS]
    # 和 D13 / D15 / D8 用的是同一个抽样种子，所以一致率能直接和历史数字比
    return sample_tasks(load_tasks("tasks/gen_s2.jsonl"), seed=2024)


def run_level(level: int, tasks: list[dict], settings, repeats: int, tag: str) -> dict:
    """跑一档。返回这一档的耗时、吞吐、错误计数、逐条判定。"""
    timings: dict = {}
    client = DeepSeekClient(settings, thinking=False, max_tokens=1024, timeout=120.0)
    started = time.perf_counter()
    try:
        # 每档一个独立的 asyncio.run：新的事件循环 + 新的默认 executor，
        # 上一档的线程池不会残留（asyncio.run 退出时会 shutdown_default_executor）。
        outcomes = asyncio.run(run_batch_async(
            tasks, AGENTS["full"], client, repeats=repeats,
            concurrency=level, timings=timings,
            # 线程池上限跟着并发走，否则并发 16/32 会被默认的 min(32, cpu+4) 卡住，
            # 测出来的是线程池上限，不是服务端上限
            threads=max(level, 8),
        ))
    finally:
        client.close()
    wall = time.perf_counter() - started

    errors = [o for o in outcomes if not o.ok]
    latencies = [v for v in timings.values()]
    within = within_task_consistency(outcomes)
    row = {
        "concurrency": level,
        "runs": len(outcomes),
        # 波数 = 执行数 / 并发度。**这是判断"加速比可不可信"的关键**：
        # 波数 < 1 时全部执行一波打完，墙钟由"最慢那个任务"决定，
        # 串行总耗时却是 N × 平均值——两者一除，加速比被长尾压低，**偏低**。
        # 实测：同 30 条任务、同 32 并发，30 次（0.94 波）18.40×、90 次（2.81 波）25.41×。
        "waves": round(len(outcomes) / level, 2),
        "enough_waves": len(outcomes) / level >= MIN_WAVES,
        "wall_s": round(wall, 2),
        "throughput": round(len(outcomes) / wall, 3) if wall else 0.0,
        "ok": len(outcomes) - len(errors),
        "request_error": len(errors),
        "status_429": sum(1 for o in outcomes if o.error_status == 429),
        "other_status": sorted({o.error_status for o in outcomes
                                if o.error_status and o.error_status != 429}),
        "latency_mean_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
        "latency_p50_ms": round(sorted(latencies)[len(latencies) // 2], 1) if latencies else 0.0,
        "pass_rate": pass_rate(outcomes),
        "within": within,
        "outcomes": outcomes,
    }
    log = RESULTS / "d27_{0}_conc{1}.jsonl".format(tag, level)
    log.write_text("\n".join(json.dumps(o.__dict__, ensure_ascii=False) for o in outcomes) + "\n",
                   encoding="utf-8")
    return row


def main() -> None:
    levels, kind, repeats, out = LEVELS, "all", REPEATS, OUT
    for arg in sys.argv[1:]:
        if arg.startswith("--levels="):
            levels = tuple(int(x) for x in arg.split("=", 1)[1].split(","))
        elif arg.startswith("--tasks="):
            kind = arg.split("=", 1)[1]
        elif arg.startswith("--repeats="):
            repeats = int(arg.split("=", 1)[1])
        elif arg.startswith("--out="):
            out = RESULTS / arg.split("=", 1)[1]

    tasks = load_picked(kind)
    settings = dataclasses.replace(load_settings(), model="deepseek-flash")
    runs = len(tasks) * repeats
    print("D27 并发调优（{0}）：{1} 条 × {2} 次 = {3} 次执行/档".format(
        "全量 30 条（真实规模）" if kind == "all" else "10 条难样本", len(tasks), repeats, runs),
        flush=True)
    print("档位:", levels, "| 每档波数:", [round(runs / lv, 1) for lv in levels], flush=True)

    rows: list[dict] = []
    for level in levels:
        row = run_level(level, tasks, settings, repeats, kind)
        rows.append(row)
        print("  并发 {0:>2}: {1:6.1f}s  {2:5.2f} 次/s  {3:4.1f} 波  429×{4}  req_err×{5}  "
              "单次 {6:.0f}ms  通过率 {7:.3f}  任务内一致率 {8:.1%}".format(
                  level, row["wall_s"], row["throughput"], row["waves"], row["status_429"],
                  row["request_error"], row["latency_mean_ms"], row["pass_rate"],
                  row["within"]["rate"]), flush=True)

    # 32 出现 429 就往下退：判据是"429 为 0 的前提下最快"，
    # 32 脏了就得看 16 干不干净，而不是"32 最快所以 32"。
    dirty = [r["concurrency"] for r in rows if r["status_429"] > 0]
    if dirty:
        print("  {0} 档出现 429，最优从干净档里选（{1}）".format(
            dirty, [r["concurrency"] for r in rows if r["status_429"] == 0]), flush=True)

    base = rows[0]
    for row in rows:
        row["speedup"] = round(base["wall_s"] / row["wall_s"], 2) if row["wall_s"] else 0.0
        agree = agreement(base["outcomes"], row["outcomes"])
        row["agreement"] = {k: agree[k] for k in
                            ("total", "same", "rate", "valid_total", "valid_same", "valid_rate")}
        row["diffs"] = agree["diffs"]

    plain = [{k: v for k, v in r.items() if k not in ("outcomes", "diffs")} for r in rows]
    best = pick_best_concurrency(plain)
    best["speedup"] = next(r["speedup"] for r in plain if r["concurrency"] == best["best"])
    best_row = next(r for r in plain if r["concurrency"] == best["best"])
    best["waves"] = best_row["waves"]
    best["enough_waves"] = best_row["enough_waves"]

    payload = {
        "task_set": kind,
        "tasks": [t["id"] for t in tasks],
        "task_count": len(tasks),
        "repeats": repeats,
        "executions_per_level": runs,
        "min_waves": MIN_WAVES,
        "agent": AGENTS["full"]["name"],
        "model": settings.model,
        "rows": plain,
        "baseline": {"concurrency": 1, "wall_s": base["wall_s"], "throughput": base["throughput"],
                     "pass_rate": base["pass_rate"], "latency_mean_ms": base["latency_mean_ms"],
                     "within_rate": base["within"]["rate"]},
        "best": best,
        "measured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "detail": {str(r["concurrency"]): r["diffs"] for r in rows if r["diffs"]},
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n{0:<6}{1:>9}{2:>9}{3:>7}{4:>7}{5:>8}{6:>9}{7:>12}".format(
        "并发", "总耗时", "吞吐", "波数", "429", "加速比", "任务内一致", "跨档一致"))
    for row in plain:
        print("{0:<6}{1:>8.1f}s{2:>9.2f}{3:>7.1f}{4:>7}{5:>7.2f}×{6:>11.1%}{7:>11.1%}{8}".format(
            row["concurrency"], row["wall_s"], row["throughput"], row["waves"],
            row["status_429"], row["speedup"], row["within"]["rate"],
            row["agreement"]["rate"],
            "   ← 波数不足，加速比偏低" if not row["enough_waves"] else ""))
    print("\n最优并发 = {0}（{1}）；首个 429 出现在 {2}".format(
        best["best"], best["why"], best["first_429"] if best["first_429"] else "未出现"))
    print("写入", out)


if __name__ == "__main__":
    main()
