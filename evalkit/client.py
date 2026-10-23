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

import os
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


class ChatError(RuntimeError):
    """请求失败：超时 / 网络 / HTTP 非 200 / 响应结构异常。"""


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
                 timeout: float = DEFAULT_TIMEOUT, transport: httpx.BaseTransport | None = None) -> None:
        self._settings = settings
        self._thinking = thinking
        self._timeout = timeout
        self._transport = transport
        self.last_elapsed_ms: float = 0.0

    def chat(self, messages: list[dict], tools: list[dict]) -> ChatReply:
        payload = {
            "model": self._settings.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": 0,
            "max_tokens": 1024,
            # 原始 HTTP 请求里这个参数直接放 body 顶层（SDK 里才叫 extra_body）
            "thinking": {"type": "enabled" if self._thinking else "disabled"},
        }
        headers = {"Authorization": "Bearer " + self._settings.api_key, "Content-Type": "application/json"}
        kwargs: dict = {"timeout": httpx.Timeout(self._timeout, connect=CONNECT_TIMEOUT)}
        if self._transport is not None:
            kwargs["transport"] = self._transport
        started = time.perf_counter()
        try:
            with httpx.Client(**kwargs) as client:
                resp = client.post(chat_url(self._settings.base_url), json=payload, headers=headers)
        except httpx.TimeoutException as exc:
            raise ChatError("请求超时（>{0:.0f}s），检查网络或代理".format(self._timeout)) from exc
        except httpx.HTTPError as exc:
            raise ChatError("网络错误: {0}: {1}".format(type(exc).__name__, exc)) from exc
        elapsed_ms = (time.perf_counter() - started) * 1000
        self.last_elapsed_ms = elapsed_ms
        if resp.status_code != 200:
            # 不带 body：401/403 的 body 可能回显 key
            hint = "多半是 base_url 层级填错" if resp.status_code == 404 else "常见原因：key 无效 / 额度不足 / 限流"
            raise ChatError("HTTP {0}: {1}".format(resp.status_code, hint))
        try:
            data = resp.json()
            message = data["choices"][0]["message"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ChatError("响应结构异常，取不到 choices[0].message") from exc
        if not isinstance(message, dict):
            raise ChatError("choices[0].message 不是对象")
        return ChatReply(message=message, elapsed_ms=elapsed_ms, usage=data.get("usage") or {})


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
