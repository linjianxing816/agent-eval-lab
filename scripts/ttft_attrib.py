"""TTFT 归因：同一个 prompt，带工具 schema vs 不带，看首 token 里 prefill 占多少。

带 schema ≈ 4,130 输入 token，不带 ≈ 几百 token。如果 TTFT 掉到三分之一以下，
说明 prefill（处理输入）占大头；如果几乎不变，说明排队/网络占大头。

用法：python scripts/ttft_attrib.py
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

RESULTS = Path("results")
MESSAGES = [
    {"role": "system", "content": FULL_AGENT["system_prompt"]},
    {"role": "user", "content": "用户 U1001 要退订单 A123，按规则处理"},
]


def main() -> None:
    settings = dataclasses.replace(load_settings(), model="deepseek-flash")
    client = DeepSeekClient(settings, thinking=False, max_tokens=1024, timeout=90.0)

    print("不带 schema：10 次", flush=True)
    without = measure_latency(lambda: client.chat_stream(MESSAGES, []), samples=10, gap_s=0.5)
    print("   TTFT P50 {0:.0f}ms  P95 {1:.0f}ms".format(
        without["ttft"]["p50"], without["ttft"]["p95"]), flush=True)

    path = RESULTS / "ttft_attrib.json"
    payload = {
        "without_tools": without,
        "with_tools_ttft_p50": None,     # 由 build_report 从 perf.json 填，避免抄错
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("写入", path, flush=True)


if __name__ == "__main__":
    main()
