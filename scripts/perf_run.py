"""D15 单请求延迟 + D16 并发扫描。跑一次把两份数据都写进 results/perf.json。

用法：python scripts/perf_run.py

**同一个 prompt、同一套工具 schema**，和评测时完全一致——换了 prompt，
TTFT 会变（输入越长首 token 越慢），测出来的数就不能用来解释评测里的 llm_time。
"""
from __future__ import annotations

import dataclasses
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evalkit.agent.strong import STRONG_AGENT
from evalkit.client import DeepSeekClient, load_settings
from evalkit.perf import measure_latency, sweep_concurrency

RESULTS = Path("results")
INSTRUCTION = "用户 U1001 要退订单 A123，按规则处理"
MESSAGES = [
    {"role": "system", "content": STRONG_AGENT["system_prompt"]},
    {"role": "user", "content": INSTRUCTION},
]
TOOLS = STRONG_AGENT["tools"]


def main() -> None:
    settings = dataclasses.replace(load_settings(), model="deepseek-flash")
    client = DeepSeekClient(settings, thinking=False, max_tokens=1024, timeout=90.0)

    print("D15：单请求延迟 20 次（串行，间隔 0.5s）", flush=True)
    latency = measure_latency(
        lambda: client.chat_stream(MESSAGES, TOOLS), samples=20, gap_s=0.5)
    print("   TTFT  P50 {0:.0f}ms  P95 {1:.0f}ms".format(
        latency["ttft"]["p50"], latency["ttft"]["p95"]), flush=True)
    print("   TPOT  P50 {0:.1f}ms  P95 {1:.1f}ms".format(
        latency["tpot"]["p50"], latency["tpot"]["p95"]), flush=True)

    print("D16：并发扫描 1/2/4，每档 12 个请求", flush=True)

    def factory():
        def call() -> float:
            return client.chat_stream(MESSAGES, TOOLS).e2e_ms
        return call

    rows = sweep_concurrency(factory, levels=(1, 2, 4), per_level=12)
    for row in rows:
        print("   并发 {0}: 吞吐 {1:.2f} req/s  P50 {2:.0f}ms  P95 {3:.0f}ms  429×{4}".format(
            row["concurrency"], row["throughput"], row["p50"], row["p95"], row["status_429"]),
            flush=True)

    payload = {
        "instruction": INSTRUCTION,
        "tools": [t["function"]["name"] for t in TOOLS],
        "latency": latency,
        "concurrency": rows,
        "measured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (RESULTS / "perf.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    print("写入 results/perf.json", flush=True)


if __name__ == "__main__":
    main()
