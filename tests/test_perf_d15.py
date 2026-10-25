"""D15 / D16 推理性能：分位数、流式解析、并发扫描。全部离线。"""
from __future__ import annotations

import httpx
import pytest

from evalkit.client import ChatError, DeepSeekClient, Settings
from evalkit.perf import measure_latency, percentile, summarize, sweep_concurrency


class _Metrics:
    def __init__(self, ttft, tpot, e2e, tokens=10):
        self.ttft_ms = ttft
        self.tpot_ms = tpot
        self.e2e_ms = e2e
        self.output_tokens = tokens


# ---------- 分位数 ----------

def test_percentile_uses_linear_interpolation():
    values = [10.0, 20.0, 30.0, 40.0]
    assert percentile(values, 0.5) == 25.0
    assert percentile(values, 0.0) == 10.0
    assert percentile(values, 1.0) == 40.0
    # 4 个样本的 P95 不该等于最大值（最近秩算法会，线性插值不会）
    assert percentile(values, 0.95) < 40.0


def test_summarize_reports_all_four_numbers():
    s = summarize([100.0, 200.0, 300.0])
    assert s["n"] == 3
    assert s["p50"] == 200.0
    assert s["min"] == 100.0 and s["max"] == 300.0
    assert summarize([])["n"] == 0


# ---------- 流式解析 ----------

def _sse(*chunks: str) -> bytes:
    lines = []
    for chunk in chunks:
        lines.append("data: " + chunk)
        lines.append("")
    lines.append("data: [DONE]")
    lines.append("")
    return "\n".join(lines).encode("utf-8")


def test_chat_stream_measures_ttft_and_tpot():
    body = _sse(
        '{"choices":[{"delta":{"role":"assistant"}}]}',          # 元数据 delta，不算首 token
        '{"choices":[{"delta":{"content":"你"}}]}',
        '{"choices":[{"delta":{"content":"好"}}]}',
        '{"choices":[{"delta":{"content":"呀"}}]}',
        '{"choices":[],"usage":{"prompt_tokens":100,"completion_tokens":3}}',
    )
    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=body))
    client = DeepSeekClient(Settings("https://x", "k", "m"), transport=transport)
    result = client.chat_stream([{"role": "user", "content": "hi"}], [])
    assert result.text == "你好呀"
    assert result.output_tokens == 3
    assert result.ttft_ms is not None and result.ttft_ms >= 0
    assert result.tpot_ms is not None
    assert result.e2e_ms >= result.ttft_ms


def test_chat_stream_assembles_tool_call_deltas():
    body = _sse(
        '{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"get_order","arguments":"{\\"order"}}]}}]}',
        '{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"_id\\":\\"A123\\"}"}}]}}]}',
    )
    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=body))
    client = DeepSeekClient(Settings("https://x", "k", "m"), transport=transport)
    result = client.chat_stream([{"role": "user", "content": "hi"}], [])
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["function"]["name"] == "get_order"
    assert result.tool_calls[0]["function"]["arguments"] == '{"order_id":"A123"}'


def test_chat_stream_raises_with_status_on_429():
    transport = httpx.MockTransport(lambda request: httpx.Response(429, content=b"{}"))
    client = DeepSeekClient(Settings("https://x", "k", "m"), transport=transport)
    with pytest.raises(ChatError) as exc:
        client.chat_stream([{"role": "user", "content": "hi"}], [])
    assert exc.value.status == 429


# ---------- 延迟采样 ----------

def test_measure_latency_collects_distribution_without_sleep():
    values = iter([100.0, 200.0, 300.0])
    result = measure_latency(
        lambda: _Metrics(next(values), 10.0, next(iter([150.0]))), samples=3, gap_s=0)
    assert result["ok"] == 3
    assert result["ttft"]["p50"] == 200.0
    assert result["ttft"]["max"] == 300.0


def test_measure_latency_counts_errors_but_keeps_going():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 2:
            raise ChatError("HTTP 429: 限流", status=429)
        return _Metrics(100.0, 10.0, 150.0)

    result = measure_latency(flaky, samples=3, gap_s=0)
    assert result["ok"] == 2
    assert len(result["errors"]) == 1


# ---------- 并发扫描 ----------

def test_sweep_counts_429_and_does_not_retry():
    state = {"n": 0}

    def factory():
        def call():
            state["n"] += 1
            if state["n"] % 2 == 0:
                raise ChatError("HTTP 429: 限流", status=429)
            return 1.0
        return call

    rows = sweep_concurrency(factory, levels=(2,), per_level=4)
    assert len(rows) == 1
    assert rows[0]["concurrency"] == 2
    assert rows[0]["status_429"] == 2
    assert rows[0]["requests"] == 4
    assert state["n"] == 4                      # 每个请求只发一次，没有重试


def test_sweep_reports_throughput_and_percentiles():
    rows = sweep_concurrency(lambda: (lambda: 0.01), levels=(1, 2), per_level=4)
    assert [r["concurrency"] for r in rows] == [1, 2]
    assert all(r["throughput"] > 0 for r in rows)
    assert all(r["p50"] >= 0 for r in rows)


def test_measure_latency_reports_ttft_share():
    # 首 token 占端到端的比例 —— 用来判断瓶颈是首 token 还是每 token
    result = measure_latency(lambda: _Metrics(300.0, 10.0, 1200.0), samples=3, gap_s=0)
    assert result["ttft_share_pct"]["p50"] == 25.0      # 300/1200
