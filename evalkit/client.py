"""模型客户端：把「发一次 chat 请求」这件事收成一个接口。

三个实现：
- `ModelClient`：Protocol，循环只依赖它，方便离线测
- `DeepSeekClient`：真调用（OpenAI 兼容格式）
- `FakeClient`：预置返回，离线跑循环，不花钱不联网

**thinking 默认关闭**。DeepSeek 的 thinking 默认是开的，开了会：CoT 计入输出 token、
`temperature` 失效、每步延迟里混着思考时间（D15 测 TTFT 就没意义了）。

**key 只从 .env 读**，报错里只打状态码不打 body（401 的 body 可能回显 key）。
"""
from __future__ import annotations

import json
import logging
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import httpx
from dotenv import load_dotenv

DEFAULT_TIMEOUT = 60.0
CONNECT_TIMEOUT = 10.0
PLACEHOLDER_KEY = "sk-在这里填你的key"
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"

logger = logging.getLogger("evalkit.client")

# ---------- 重试策略 ----------
# **哪些能重试**：这类错误过一会儿大概率就好了。
#   429 限流、5xx 服务端抖动、超时、连接错误
# **哪些不能**：重试一百次也是同样的结果，只会把时间浪费掉。
#   400 参数错、401/403 鉴权、404 路径错
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
MAX_RETRIES = 3
BASE_DELAY_S = 1.0
# **抖动必须有**：并发跑批时多个请求会同时撞上 429，
# 固定间隔会让它们在同一毫秒一起重试，二次打爆上游。
JITTER_RATIO = 0.2


class ChatError(RuntimeError):
    """请求失败：超时 / 网络 / HTTP 非 200 / 响应结构异常。

    `status` 只在拿到 HTTP 状态码时非空（429 要靠它统计，不能靠解析文案）。
    `retryable` 由抛出点显式给：`None` 表示"没判过"，重试层按 status 兜底。
    """

    def __init__(self, message: str, status: int | None = None,
                 retryable: bool | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.retryable = retryable


@dataclass(frozen=True)
class RetryEvent:
    """一次重试的记录。测试和日志都用它，**不靠解析日志文本**。"""

    attempt: int          # 第几次重试，1-based
    reason: str           # "HTTP 429" / "请求超时" / "网络错误"
    delay_s: float        # 这次退避了多久
    status: int | None


def backoff_delay(attempt: int, rng: random.Random | None = None) -> float:
    """第 `attempt` 次重试前等多久：1s → 2s → 4s，每次 ±20% 抖动。

    抖动是**乘法**（`base × (1 ± 0.2)`）不是加法：间隔越长抖得越多，
    这样 4 秒那次能把同时重试的请求摊开 1.6 秒，1 秒那次摊开 0.4 秒。
    """
    base = BASE_DELAY_S * (2 ** (attempt - 1))
    span = base * JITTER_RATIO
    generator = rng or random
    return round(base + generator.uniform(-span, span), 3)


def _is_retryable(exc: ChatError) -> bool:
    """能不能重试。抛出处已经判过就用它的结论，没判过按状态码兜底。

    **兜底只看状态码，不看 message**：靠中文字符串匹配"超时"太脆，
    换个措辞就失效。真正需要重试的超时/网络错误在抛出处已经标了 `retryable=True`。
    """
    if exc.retryable is not None:
        return exc.retryable
    return exc.status is not None and exc.status in RETRYABLE_STATUS


def _reason_of(exc: ChatError) -> str:
    """日志和 `RetryEvent` 里用的短原因。有状态码就用状态码，没有就截消息。"""
    if exc.status is not None:
        return "HTTP {0}".format(exc.status)
    return str(exc)[:40]


class MissingConfigError(RuntimeError):
    """配置缺失或仍是占位符。消息里只有变量名，绝不含值。"""


@dataclass(frozen=True)
class Settings:
    base_url: str
    api_key: str
    model: str


@dataclass(frozen=True)
class ChatReply:
    message: dict        # 原样保留 content / tool_calls / reasoning_content
    elapsed_ms: float
    usage: dict


@dataclass(frozen=True)
class StreamMetrics:
    """一次流式请求的三个时间点和派生指标。

    `ttft_ms` 是"用户感觉卡不卡"，`tpot_ms` 是"长回答等多久"——
    两者是不同的问题，不能合成一个"平均延迟"。
    """

    ttft_ms: float | None      # t1 - t0：首 token 延迟
    tpot_ms: float | None      # (t2 - t1) / (输出 token 数 - 1)：每 token 时间
    e2e_ms: float              # t2 - t0：端到端
    output_tokens: int
    text: str
    tool_calls: list[dict]
    usage: dict


def load_env(env_path: Path | None = None) -> None:
    """把 .env 灌进 os.environ。

    **任何读 os.environ 的地方都要先调它**——只 import dotenv 不会自动加载。
    踩过：cost_of() 直接读 os.environ，用户把单价填进 .env 后仍然是 None。
    """
    path = ENV_PATH if env_path is None else env_path
    if path.exists():
        load_dotenv(path, override=True)


def load_settings(env_path: Path | None = None) -> Settings:
    """从 .env 读配置。与 scripts/smoke_llm.py 的同名函数行为一致——
    那边是自检脚本，不反向依赖 evalkit，所以两份实现各自独立。"""
    path = ENV_PATH if env_path is None else env_path
    if path.exists():
        load_dotenv(path, override=True)
    missing = [k for k in ("AI_BASE_URL", "AI_API_KEY", "AI_MODEL") if not (os.environ.get(k) or "").strip()]
    if missing:
        raise MissingConfigError("缺少配置项: " + ", ".join(missing) + " （检查 .env）")
    api_key = os.environ["AI_API_KEY"].strip()
    if api_key == PLACEHOLDER_KEY:
        raise MissingConfigError("AI_API_KEY 还是占位符，请在 .env 里填真实 key")
    return Settings(
        base_url=os.environ["AI_BASE_URL"].strip(),
        api_key=api_key,
        model=os.environ["AI_MODEL"].strip(),
    )


def load_model_b_settings(env_path: Path | None = None) -> Settings:
    """第二个被测模型（D22）。没配就抛 MissingConfigError，消息里只说变量名。

    **和 A 用同一套 loop / system prompt / 工具 schema**，只换 model + key + base_url——
    否则"哪个模型更好"和"哪套配置更好"就混在一起了。
    """
    path = ENV_PATH if env_path is None else env_path
    if path.exists():
        load_dotenv(path, override=True)
    missing = [k for k in ("MODEL_B_BASE_URL", "MODEL_B_API_KEY", "MODEL_B_MODEL")
               if not (os.environ.get(k) or "").strip()]
    if missing:
        raise MissingConfigError("缺少配置项: " + ", ".join(missing) + " （检查 .env）")
    return Settings(
        base_url=os.environ["MODEL_B_BASE_URL"].strip(),
        api_key=os.environ["MODEL_B_API_KEY"].strip(),
        model=os.environ["MODEL_B_MODEL"].strip(),
    )


def chat_url(base_url: str) -> str:
    """拼出 chat/completions 地址。层级由 .env 负责，这里不补 /v1。"""
    stripped = base_url.strip().rstrip("/")
    if not stripped.startswith(("http://", "https://")):
        raise MissingConfigError("AI_BASE_URL 必须以 http:// 或 https:// 开头")
    return stripped + "/chat/completions"


class ModelClient(Protocol):
    def chat(self, messages: list[dict], tools: list[dict]) -> ChatReply:
        ...


class DeepSeekClient:
    def __init__(self, settings: Settings, thinking: bool = False,
                 timeout: float = DEFAULT_TIMEOUT, transport: httpx.BaseTransport | None = None,
                 max_tokens: int = 1024, max_retries: int = MAX_RETRIES,
                 sleep_fn=None) -> None:
        self._settings = settings
        self._thinking = thinking
        self._timeout = timeout
        self._transport = transport
        self._max_tokens = max_tokens
        self.last_elapsed_ms: float = 0.0
        # 复用一个 httpx.Client：每次新建的话每次都要重新做 TCP + TLS 握手，
        # 那部分时间会全部算进 TTFT。实测见 results/perf.json 的 before/after。
        self._http: httpx.Client | None = None
        self.max_retries = max_retries
        # **每次重试都记一条**，测试和"重试到底救回了多少"的分析都读它
        self.retry_events: list[RetryEvent] = []
        # 测试注入 `lambda _: None` 跳过真实等待；生产用 time.sleep
        self._sleep = sleep_fn or time.sleep

    def _client(self) -> httpx.Client:
        """懒创建并复用。第一次调用才建，避免只构造不请求的场景白占连接。"""
        if self._http is None:
            kwargs: dict = {"timeout": httpx.Timeout(self._timeout, connect=CONNECT_TIMEOUT)}
            if self._transport is not None:
                kwargs["transport"] = self._transport
            self._http = httpx.Client(**kwargs)
        return self._http

    def close(self) -> None:
        """显式关掉连接池。不关会有 ResourceWarning，长跑脚本里也会攒 fd。"""
        if self._http is not None:
            self._http.close()
            self._http = None

    def __enter__(self) -> "DeepSeekClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def chat(self, messages: list[dict], tools: list[dict]) -> ChatReply:
        """带指数退避重试的 chat。**上层调用方一行都不用改。**

        重试三层判断：
        1. 这次错误可不可重试（`ChatError.retryable`，抛出处已经判好）
        2. 还有没有重试次数（最多 `max_retries` 次）
        3. 退避多久（1s → 2s → 4s，±20% 抖动）

        最后一次失败时抛原异常——**不包一层"重试失败"**，
        否则上层拿到的 `status` 就丢了，429 统计会归零。
        """
        attempt = 0
        while True:
            try:
                reply = self._chat_once(messages, tools)
            except ChatError as exc:
                if not _is_retryable(exc) or attempt >= self.max_retries:
                    if attempt:
                        logger.error("[retry] 重试 %d 次仍失败，放弃：%s", attempt, exc)
                    raise
                attempt += 1
                delay = backoff_delay(attempt)
                self.retry_events.append(RetryEvent(
                    attempt=attempt, reason=_reason_of(exc), delay_s=delay,
                    status=getattr(exc, "status", None)))
                logger.warning("[retry %d/%d] %s | 退避 %.2fs",
                               attempt, self.max_retries, _reason_of(exc), delay)
                self._sleep(delay)
                continue
            if attempt:
                logger.info("[retry] 第 %d 次重试成功", attempt)
            return reply

    def _chat_once(self, messages: list[dict], tools: list[dict]) -> ChatReply:
        payload = {
            "model": self._settings.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": 0,
            "max_tokens": self._max_tokens,
            # 原始 HTTP 请求里这个参数直接放 body 顶层（SDK 里才叫 extra_body）
            "thinking": {"type": "enabled" if self._thinking else "disabled"},
        }
        headers = {"Authorization": "Bearer " + self._settings.api_key, "Content-Type": "application/json"}
        started = time.perf_counter()
        try:
            resp = self._client().post(chat_url(self._settings.base_url), json=payload, headers=headers)
        except httpx.TimeoutException as exc:
            raise ChatError("请求超时（>{0:.0f}s），检查网络或代理".format(self._timeout),
                            retryable=True) from exc
        except httpx.HTTPError as exc:
            raise ChatError("网络错误: {0}: {1}".format(type(exc).__name__, exc),
                            retryable=True) from exc
        elapsed_ms = (time.perf_counter() - started) * 1000
        self.last_elapsed_ms = elapsed_ms
        if resp.status_code != 200:
            # 不带 body：401/403 的 body 可能回显 key
            hint = "多半是 base_url 层级填错" if resp.status_code == 404 else "常见原因：key 无效 / 额度不足 / 限流"
            raise ChatError("HTTP {0}: {1}".format(resp.status_code, hint),
                            status=resp.status_code,
                            retryable=resp.status_code in RETRYABLE_STATUS)
        try:
            data = resp.json()
            message = data["choices"][0]["message"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            # 结构异常不重试：要么是厂商改了协议，要么是网关塞了 HTML，重试还是同一个结果
            raise ChatError("响应结构异常，取不到 choices[0].message", retryable=False) from exc
        if not isinstance(message, dict):
            raise ChatError("choices[0].message 不是对象", retryable=False)
        return ChatReply(message=message, elapsed_ms=elapsed_ms, usage=data.get("usage") or {})


    def chat_stream(self, messages: list[dict], tools: list[dict]) -> StreamMetrics:
        """流式请求，测 TTFT / TPOT。

        **不能用 chat() 的耗时当 TTFT**：那是"整个响应收完"的时间，
        里面混着输出 token 的生成时间。要分开就必须真的流式收。

        只统计 `content` 或 `tool_calls` 的 delta——`role` 那种元数据 delta 不算首 token，
        否则 TTFT 会变成"连接建立时间"，看起来好得不真实。
        """
        payload = {
            "model": self._settings.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": 0,
            "max_tokens": self._max_tokens,
            "thinking": {"type": "enabled" if self._thinking else "disabled"},
            "stream": True,
        }
        headers = {"Authorization": "Bearer " + self._settings.api_key, "Content-Type": "application/json"}

        started = time.perf_counter()
        first_token_at: float | None = None
        tokens = 0
        texts: list[str] = []
        calls: dict[int, dict] = {}
        usage: dict = {}
        try:
            with self._client().stream("POST", chat_url(self._settings.base_url),
                                       json=payload, headers=headers) as resp:
                    if resp.status_code != 200:
                        resp.read()
                        hint = ("多半是 base_url 层级填错" if resp.status_code == 404
                                else "常见原因：key 无效 / 额度不足 / 限流")
                        raise ChatError("HTTP {0}: {1}".format(resp.status_code, hint),
                                        status=resp.status_code)
                    for line in resp.iter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        body = line[5:].strip()
                        if not body or body == "[DONE]":
                            continue
                        chunk = json.loads(body)
                        if chunk.get("usage"):
                            usage = chunk["usage"]
                        choices = chunk.get("choices") or []
                        if not choices:
                            continue
                        delta = choices[0].get("delta") or {}
                        piece = delta.get("content") or ""
                        tool_deltas = delta.get("tool_calls") or []
                        if not piece and not tool_deltas:
                            continue
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                        if piece:
                            tokens += 1
                            texts.append(piece)
                        for item in tool_deltas:
                            index = item.get("index", 0)
                            slot = calls.setdefault(index, {"id": "", "type": "function",
                                                            "function": {"name": "", "arguments": ""}})
                            if item.get("id"):
                                slot["id"] = item["id"]
                            fn = item.get("function") or {}
                            if fn.get("name"):
                                slot["function"]["name"] += fn["name"]
                            if fn.get("arguments"):
                                slot["function"]["arguments"] += fn["arguments"]
        except httpx.TimeoutException as exc:
            raise ChatError("流式请求超时（>{0:.0f}s）".format(self._timeout)) from exc
        except httpx.HTTPError as exc:
            raise ChatError("网络错误: {0}: {1}".format(type(exc).__name__, exc)) from exc
        except ValueError as exc:
            raise ChatError("SSE 数据块不是合法 JSON") from exc

        ended = time.perf_counter()
        e2e_ms = (ended - started) * 1000
        ttft_ms = (first_token_at - started) * 1000 if first_token_at else None
        # 用 usage 里的输出 token 数更准；没有就退回"内容 delta 的个数"
        out_tokens = int(usage.get("completion_tokens") or tokens)
        # TPOT 的分母是"首 token 之后还要等几个 token"，所以减 1
        tpot_ms = ((ended - first_token_at) * 1000 / (out_tokens - 1)
                   if first_token_at and out_tokens > 1 else None)
        self.last_elapsed_ms = e2e_ms
        return StreamMetrics(
            ttft_ms=ttft_ms, tpot_ms=tpot_ms, e2e_ms=e2e_ms, output_tokens=out_tokens,
            text="".join(texts), tool_calls=[calls[k] for k in sorted(calls)], usage=usage,
        )


class FakeClient:
    """按顺序吐出预置的 message，用于离线测循环。吐完还调就是测试写错了。"""

    def __init__(self, replies: list[dict], usage: dict | None = None) -> None:
        self._replies = list(replies)
        self._usage = usage or {"prompt_tokens": 100, "completion_tokens": 20}
        self.calls: list[list[dict]] = []      # 每次请求的 messages 快照，供测试断言

    def chat(self, messages: list[dict], tools: list[dict]) -> ChatReply:
        self.calls.append([dict(m) for m in messages])
        if not self._replies:
            raise AssertionError("FakeClient 的预置回复用完了：循环比预期多跑了一轮")
        return ChatReply(message=self._replies.pop(0), elapsed_ms=0.0, usage=dict(self._usage))
