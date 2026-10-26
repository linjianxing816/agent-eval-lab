"""D15 / D16 推理性能：单请求延迟（TTFT / TPOT）与并发扫描。

**为什么必须流式测**：`chat()` 的耗时是"整个响应收完"，
里面混着输出 token 的生成时间。用它当 TTFT 会把"首 token 慢"和"每 token 慢"
混成一个数，而这两件事的优化手段完全不同（前者是排队/网络，后者是解码速度）。

**为什么报分位数不报均值**：延迟是长尾分布，均值会被少数慢请求拉高，
P50 告诉你"典型体验"，P95 告诉你"多久难受一次"。两个都要。

**429 不重试**：429 本身就是结论——这个并发度超了服务端配额。
重试会把"超了"这件事抹掉。
"""
from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable

from evalkit.client import ChatError


def percentile(values: list[float], q: float) -> float:
    """线性插值分位数。q 取 0~1。

    样本少的时候不要用"最近秩"——8 个样本取 P95 会直接等于最大值，
    看起来像"最慢那次就是典型情况"，误导。
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 3)
    pos = (len(ordered) - 1) * q
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return round(ordered[low], 3)
    weight = pos - low
    return round(ordered[low] * (1 - weight) + ordered[high] * weight, 3)


def summarize(values: list[float]) -> dict[str, float]:
    """P50 / P95 / min / max。空列表返回全 0，不抛。"""
    if not values:
        return {"n": 0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "n": len(values),
        "p50": percentile(values, 0.5),
        "p95": percentile(values, 0.95),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
        "mean": round(sum(values) / len(values), 3),
    }


def measure_latency(stream_call: Callable[[], Any], samples: int,
                    gap_s: float = 0.5) -> dict[str, Any]:
    """串行跑 N 次流式请求，返回 TTFT / TPOT / 端到端的分布。

    `stream_call` 是一个无参函数，返回带 ttft_ms / tpot_ms / e2e_ms 的对象。
    串行是为了让每个样本都在"没有别的请求排队"的条件下测——
    并发下的延迟是另一个问题（D16 管）。
    """
    ttft: list[float] = []
    tpot: list[float] = []
    e2e: list[float] = []
    tokens: list[int] = []
    # 每次请求里"首 token 占总耗时"的比例。用来回答"瓶颈是首 token 还是每 token"——
    # 只有成对的数据才算得出，所以单独存。
    shares: list[float] = []
    errors: list[str] = []
    for index in range(samples):
        try:
            result = stream_call()
        except ChatError as exc:
            errors.append(str(exc))
            continue
        if result.ttft_ms is not None:
            ttft.append(result.ttft_ms)
            if result.e2e_ms:
                shares.append(result.ttft_ms / result.e2e_ms * 100)
        if result.tpot_ms is not None:
            tpot.append(result.tpot_ms)
        e2e.append(result.e2e_ms)
        tokens.append(result.output_tokens)
        if index < samples - 1 and gap_s:
            time.sleep(gap_s)
    return {
        "samples": samples,
        "ok": len(e2e),
        "errors": errors,
        "ttft": summarize(ttft),
        "tpot": summarize(tpot),
        "e2e": summarize(e2e),
        "output_tokens": summarize([float(t) for t in tokens]),
        "ttft_share_pct": summarize(shares),
    }


@dataclass
class ConcurrencyRow:
    concurrency: int
    requests: int
    ok: int
    throughput: float        # req/s
    p50_ms: float
    p95_ms: float
    status_429: int
    other_errors: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "concurrency": self.concurrency, "requests": self.requests, "ok": self.ok,
            "throughput": round(self.throughput, 3), "p50": self.p50_ms, "p95": self.p95_ms,
            "status_429": self.status_429, "other_errors": self.other_errors,
        }


def _one_call(call: Callable[[], float]) -> tuple[float | None, int | None]:
    """返回 (耗时 ms, HTTP 状态码)。失败时耗时也算——失败请求一样占连接。"""
    started = time.perf_counter()
    try:
        call()
        return (time.perf_counter() - started) * 1000, None
    except ChatError as exc:
        return (time.perf_counter() - started) * 1000, getattr(exc, "status", None)


def sweep_concurrency(call_factory: Callable[[], Callable[[], float]],
                      levels: tuple[int, ...] = (1, 2, 4),
                      per_level: int = 12) -> list[dict[str, Any]]:
    """并发扫描。

    `call_factory()` 每次返回一个"发一次请求并返回耗时"的函数。
    每档跑 `per_level` 个请求，**并发数就是同时在飞的请求数**。
    429 单独计数，不重试。
    """
    rows: list[ConcurrencyRow] = []
    for level in levels:
        started = time.perf_counter()
        latencies: list[float] = []
        status_429 = 0
        other = 0
        with ThreadPoolExecutor(max_workers=level) as pool:
            futures = [pool.submit(_one_call, call_factory()) for _ in range(per_level)]
            for future in futures:
                elapsed, status = future.result()
                if elapsed is not None:
                    latencies.append(elapsed)
                if status == 429:
                    status_429 += 1
                elif status is not None:
                    other += 1
        wall = time.perf_counter() - started
        rows.append(ConcurrencyRow(
            concurrency=level, requests=per_level,
            ok=len(latencies) - status_429 - other,
            throughput=per_level / wall if wall else 0.0,
            p50_ms=percentile(latencies, 0.5),
            p95_ms=percentile(latencies, 0.95),
            status_429=status_429, other_errors=other,
        ))
    return [row.as_dict() for row in rows]


def pick_best_concurrency(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """最优并发 = **在 429 为 0 的前提下最快**，不是最快。

    为什么不选最快：429 意味着这一档的吞吐是拿"失败请求"换的。
    失败重试之后真实耗时只会更长，而且失败请求里那些任务的判定结果全废了——
    加速比好看，但那一批数据不能用。

    优先级（逐级放宽，每级都记下放宽原因）：
    1. 429 = 0 且 request_error = 0 → 首选。超时也算不稳定，理由同上。
    2. 429 = 0 但有 request_error → 退而求其次，取 request_error 最少的。
    3. 全部有 429 → 取 429 最少的，并在结论里明说"没找到干净的一档"。

    `first_429` 是**第一个出现 429 的档位**，用来回答"为什么不是更高"：
    最优必须严格小于它。
    """
    if not rows:
        return {"best": None, "reason": "没有数据", "first_429": None, "clean": []}

    ordered = sorted(rows, key=lambda r: r["concurrency"])
    first_429 = next((r["concurrency"] for r in ordered if r.get("status_429", 0) > 0), None)

    def pick(candidates: list[dict], why: str) -> dict[str, Any]:
        # 吞吐相同时取并发更小的：同样快就用更少的连接，对上游更友好
        best = max(candidates, key=lambda r: (r["throughput"], -r["concurrency"]))
        return {"best": best["concurrency"], "why": why, "row": best}

    clean = [r for r in ordered if r.get("status_429", 0) == 0 and r.get("request_error", 0) == 0]
    if clean:
        picked = pick(clean, "429 和 request_error 都是 0 的档里吞吐最高")
    else:
        no_429 = [r for r in ordered if r.get("status_429", 0) == 0]
        if no_429:
            fewest = min(r.get("request_error", 0) for r in no_429)
            picked = pick([r for r in no_429 if r.get("request_error", 0) == fewest],
                          "没有一档做到 request_error=0；在 429=0 的档里取失败最少的（{0} 条）".format(fewest))
        else:
            fewest = min(r.get("status_429", 0) for r in ordered)
            picked = pick([r for r in ordered if r.get("status_429", 0) == fewest],
                          "每一档都出现 429；取 429 最少的（{0} 次）——这一档的数据只能当参考".format(fewest))

    return {
        "best": picked["best"],
        "why": picked["why"],
        "row": picked["row"],
        "first_429": first_429,
        "clean": [r["concurrency"] for r in clean],
        "strictly_below_429": (first_429 is None) or (picked["best"] < first_429),
    }

