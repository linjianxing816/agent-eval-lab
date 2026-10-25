"""TTFT 归因 · 配对测量。

上一版是「先测 10 次带 schema，再测 10 次不带」——两次测量隔了几分钟，
网络/排队状况变了，基线从 503ms 漂到 750ms，**漂移量比要测的效应还大**。
拿这种数据算"prefill 占 37%"是错的。

正确做法：**交替测**，每轮紧接着测一次带 schema、一次不带，算同一轮内的差。
这样两边的环境条件几乎一样，差值才归因得到"输入长度"。

用法：python scripts/ttft_pair.py
"""
from __future__ import annotations

import dataclasses
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evalkit.agent.full import FULL_AGENT
from evalkit.client import DeepSeekClient, load_settings

MESSAGES = [
    {"role": "system", "content": FULL_AGENT["system_prompt"]},
    {"role": "user", "content": "用户 U1001 要退订单 A123，按规则处理"},
]
PAIRS = 10


def main() -> None:
    settings = dataclasses.replace(load_settings(), model="deepseek-flash")
    rows = []
    with DeepSeekClient(settings, thinking=False, max_tokens=1024, timeout=90.0) as client:
        for i in range(PAIRS):
            # 交替顺序：偶数轮先带后不带，奇数轮反过来，抵消"越跑越快/越慢"的漂移
            order = [("with", FULL_AGENT["tools"]), ("without", [])]
            if i % 2 == 1:
                order.reverse()
            measured = {}
            for label, tools in order:
                result = client.chat_stream(MESSAGES, tools)
                measured[label] = result.ttft_ms
            rows.append({"pair": i, "with_tools": measured["with"],
                         "without_tools": measured["without"],
                         "delta": measured["with"] - measured["without"]})
            print("pair {0}: 带 {1:.0f}ms  不带 {2:.0f}ms  差 {3:+.0f}ms".format(
                i, measured["with"], measured["without"],
                measured["with"] - measured["without"]), flush=True)

    deltas = [r["delta"] for r in rows]
    withs = [r["with_tools"] for r in rows]
    withouts = [r["without_tools"] for r in rows]
    summary = {
        "pairs": PAIRS,
        "with_tools_p50": round(statistics.median(withs), 1),
        "without_tools_p50": round(statistics.median(withouts), 1),
        "delta_p50": round(statistics.median(deltas), 1),
        "delta_min": round(min(deltas), 1),
        "delta_max": round(max(deltas), 1),
        "delta_positive_pairs": sum(1 for d in deltas if d > 0),
        "rows": rows,
    }
    Path("results/ttft_paired.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n配对结果：带 {0}ms / 不带 {1}ms / 中位差 {2:+.1f}ms（{3}/{4} 轮为正）".format(
        summary["with_tools_p50"], summary["without_tools_p50"], summary["delta_p50"],
        summary["delta_positive_pairs"], PAIRS), flush=True)


if __name__ == "__main__":
    main()
