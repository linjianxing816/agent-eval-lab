"""指数退避重试：可重试判定 / 次数 / 退避曲线 / 抖动。全部离线。

**为什么重试值得单独测**：
- 重试次数写错（比如 `>=` 写成 `>`）会多重试一次，成本翻倍还看不出来
- 抖动范围写错会让并发请求同一毫秒一起重试，二次打爆上游
- 不可重试的错误被重试，会把 400 拖成 4 次请求

`MockTransport` 按顺序吐响应，`sleep_fn` 注入空函数跳过真实等待——
否则一条测试要跑 7 秒（1+2+4）。
"""
from __future__ import annotations

import random

import httpx
import pytest

from evalkit.client import (
    BASE_DELAY_S,
    JITTER_RATIO,
    MAX_RETRIES,
    ChatError,
    DeepSeekClient,
    Settings,
    backoff_delay,
)

SETTINGS = Settings(base_url="https://example.com", api_key="sk-test", model="m")

OK_BODY = {
    "choices": [{"message": {"role": "assistant", "content": "好的"}}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 2},
}


def _client(statuses: list[int], **kwargs) -> tuple[DeepSeekClient, list]:
    """按顺序返回 `statuses` 里的状态码，最后一个会一直重复。"""
    calls: list = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        status = statuses[min(len(calls) - 1, len(statuses) - 1)]
        if status == 200:
            return httpx.Response(200, json=OK_BODY)
        return httpx.Response(status, json={"error": "x"})

    client = DeepSeekClient(SETTINGS, transport=httpx.MockTransport(handler),
                            sleep_fn=lambda _: None, **kwargs)
    return client, calls


def _ask(client: DeepSeekClient):
    return client.chat([{"role": "user", "content": "hi"}], [])


# ---------- 可重试 / 不可重试 ----------

def test_429_retries_then_gives_up():
    """429 是限流，过一会儿就好——重试满 3 次才放弃。"""
    client, calls = _client([429])
    with pytest.raises(ChatError) as exc:
        _ask(client)
    assert exc.value.status == 429                 # **原状态码必须保住**，上层要统计
    assert len(calls) == 1 + MAX_RETRIES
    assert [e.reason for e in client.retry_events] == ["HTTP 429"] * MAX_RETRIES
    assert [e.attempt for e in client.retry_events] == [1, 2, 3]


@pytest.mark.parametrize("status", [500, 502, 503, 504])
def test_server_errors_are_retryable(status):
    client, calls = _client([status, 200])
    _ask(client)
    assert len(calls) == 2
    assert len(client.retry_events) == 1


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_client_errors_are_not_retryable(status):
    """参数错 / 鉴权 / 路径错，重试一百次也是同样结果，只会浪费时间。"""
    client, calls = _client([status])
    with pytest.raises(ChatError) as exc:
        _ask(client)
    assert exc.value.status == status
    assert len(calls) == 1
    assert client.retry_events == []


def test_timeout_is_retryable():
    """超时和网络错误在抛出处标了 `retryable=True`——它们没有状态码可判。"""
    calls: list = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        raise httpx.ReadTimeout("boom", request=request)

    client = DeepSeekClient(SETTINGS, transport=httpx.MockTransport(handler),
                            sleep_fn=lambda _: None)
    with pytest.raises(ChatError):
        _ask(client)
    assert len(calls) == 1 + MAX_RETRIES
    assert len(client.retry_events) == MAX_RETRIES


# ---------- 次数与成功 ----------

def test_second_attempt_succeeds_stops_retrying():
    """第 2 次就成功了，不能继续重试——多打一次就多花一次钱。"""
    client, calls = _client([429, 200])
    reply = _ask(client)
    assert reply.message["content"] == "好的"
    assert len(calls) == 2
    assert len(client.retry_events) == 1
    assert client.retry_events[0].attempt == 1


def test_gives_up_after_max_retries():
    """一共 1 次原始 + 3 次重试 = 4 次请求，不多不少。"""
    client, calls = _client([503])
    with pytest.raises(ChatError):
        _ask(client)
    assert len(calls) == MAX_RETRIES + 1


def test_max_retries_configurable():
    client, calls = _client([429], max_retries=1)
    with pytest.raises(ChatError):
        _ask(client)
    assert len(calls) == 2


# ---------- 退避曲线 ----------

def test_backoff_doubles_each_time():
    """1s → 2s → 4s。用固定 seed 消掉抖动再比。"""
    rng = random.Random(0)
    delays = [backoff_delay(i, rng) for i in (1, 2, 3)]
    for i, delay in enumerate(delays, 1):
        base = BASE_DELAY_S * (2 ** (i - 1))
        assert base * (1 - JITTER_RATIO) <= delay <= base * (1 + JITTER_RATIO), (i, delay)
    # 单调递增：2 倍基数 + 20% 抖动，最坏情况也不会塌到前一次以下
    assert delays[0] < delays[1] < delays[2]


def test_jitter_within_20_percent():
    """抖动必须在 ±20% 内。范围写大了并发时撞车，写小了退化成固定间隔。"""
    rng = random.Random(7)
    for attempt in (1, 2, 3):
        base = BASE_DELAY_S * (2 ** (attempt - 1))
        samples = [backoff_delay(attempt, rng) for _ in range(500)]
        assert min(samples) >= base * (1 - JITTER_RATIO) - 0.001
        assert max(samples) <= base * (1 + JITTER_RATIO) + 0.001
        # 真的在抖，不是一个常数
        assert max(samples) - min(samples) > base * JITTER_RATIO


def test_retry_event_records_delay():
    client, _ = _client([429, 200])
    _ask(client)
    event = client.retry_events[0]
    assert event.status == 429
    assert 0.8 <= event.delay_s <= 1.2          # 第一次退避 ≈ 1s ±20%
