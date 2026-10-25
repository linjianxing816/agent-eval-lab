"""连接复用前后对比：同一个 client 连测 20 次（带 schema 10 + 不带 10）。

改之前每次 `chat_stream` 都 `with httpx.Client(...)`，等于每次重新做 TCP + TLS 握手。
现在 client 常驻，第二次开始复用连接。

用法：python scripts/ttft_reuse.py
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evalkit.agent.full import FULL_AGENT
from evalkit.client import DeepSeekClient, load_settings
from evalkit.perf import measure_latency

MESSAGES = [
    {"role": "system", "content": FULL_AGENT["system_prompt"]},
    {"role": "user", "content": "用户 U1001 要退订单 A123，按规则处理"},
]


def main() -> None:
    settings = dataclasses.replace(load_settings(), model="deepseek-flash")
    with DeepSeekClient(settings, thinking=False, max_tokens=1024, timeout=90.0) as client:
        print("复用连接 · 带 6 个工具 schema：10 次", flush=True)
        with_tools = measure_latency(lambda: client.chat_stream(MESSAGES, FULL_AGENT["tools"]),
                                     samples=10, gap_s=0.5)
        print("   TTFT P50 {0:.0f}ms  P95 {1:.0f}ms".format(
            with_tools["ttft"]["p50"], with_tools["ttft"]["p95"]), flush=True)

        print("复用连接 · 不带 schema：10 次", flush=True)
        without = measure_latency(lambda: client.chat_stream(MESSAGES, []), samples=10, gap_s=0.5)
        print("   TTFT P50 {0:.0f}ms  P95 {1:.0f}ms".format(
            without["ttft"]["p50"], without["ttft"]["p95"]), flush=True)

    Path("results/ttft_reuse.json").write_text(json.dumps(
        {"with_tools": with_tools, "without_tools": without}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print("写入 results/ttft_reuse.json", flush=True)


if __name__ == "__main__":
    main()
