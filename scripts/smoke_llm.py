"""D0 最小验证脚本：验证 DeepSeek（或任意 OpenAI 兼容）接口连通。

只做三件事：读 .env 配置 -> 发一条 chat 请求 -> 打印回答 / 耗时 / token。
这不是 evalkit 的业务代码，D1 之后保留作为连通性自检工具。

【base_url 层级规则】（与 .env.example 顶部注释一致）
AI_BASE_URL 必须填到「拼上 /chat/completions 就能用」的那一层。
本脚本只做 base_url.rstrip('/') + '/chat/completions'，不会替你补 /v1。
不同厂商层级不同，由 .env 自己负责：
  DeepSeek   https://api.deepseek.com
  通义       https://dashscope.aliyuncs.com/compatible-mode/v1
填错的表现：404 或 invalid url。

【密钥安全】
key 只从 .env 读；任何报错信息、日志、打印里都不出现 key 本身
（包括 HTTP 错误响应的 body —— body 可能回显 key，所以只报状态码）。

Python 3.11 兼容：本文件不用 3.12+ 语法（尤其 PEP 701 的 f-string
同种引号嵌套），格式化统一走 str.format()。
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
from dotenv import load_dotenv

DEFAULT_TIMEOUT = 30.0
CONNECT_TIMEOUT = 10.0
REQUIRED_KEYS = ("AI_BASE_URL", "AI_API_KEY", "AI_MODEL")
PLACEHOLDER_KEY = "sk-在这里填你的key"

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent


class MissingConfigError(RuntimeError):
    """配置缺失 / 占位。消息里只有变量名，绝不含值。"""


class ChatError(RuntimeError):
    """调用失败：超时 / 网络错 / HTTP 非 200 / 响应结构异常 / 空输出。"""


@dataclass(frozen=True)
class Settings:
    base_url: str
    api_key: str
    model: str


@dataclass(frozen=True)
class ChatResult:
    text: str
    elapsed_ms: float
    prompt_tokens: int
    completion_tokens: int


def build_url(base_url: str) -> str:
    """拼出 chat/completions 完整地址。

    规则见模块 docstring：不补 /v1，层级由 .env 负责。
    两个边界：末尾带 / 和不带 /，都要拼出同一个地址（有单测覆盖）。
    """
    stripped = base_url.strip()
    if not stripped:
        raise MissingConfigError("AI_BASE_URL 为空，请检查 .env")
    if not stripped.startswith(("http://", "https://")):
        raise MissingConfigError("AI_BASE_URL 必须以 http:// 或 https:// 开头")
    return stripped.rstrip("/") + "/chat/completions"


def load_settings(env_path: Path | None = None) -> Settings:
    """从 .env 读配置。缺项或仍是占位符时抛 MissingConfigError。

    优先级：已存在的进程环境变量 > .env 文件（override=True 时 .env 覆盖进程值，
    这里选 override=True，保证「改了 .env 就生效」，避免被 shell 残留变量坑）。
    """
    if env_path is None:
        env_path = PROJECT_DIR / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=True)
    missing = [name for name in REQUIRED_KEYS if not (os.environ.get(name) or "").strip()]
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


def chat(
    settings: Settings,
    prompt: str,
    timeout: float = DEFAULT_TIMEOUT,
    transport: httpx.BaseTransport | None = None,
) -> ChatResult:
    """发一次 chat 请求。所有失败路径都显式抛 ChatError，不静默。"""
    url = build_url(settings.base_url)
    payload = {
        "model": settings.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
        "temperature": 0,
    }
    headers = {
        "Authorization": "Bearer " + settings.api_key,
        "Content-Type": "application/json",
    }
    client_kwargs = {"timeout": httpx.Timeout(timeout, connect=CONNECT_TIMEOUT)}
    if transport is not None:
        client_kwargs["transport"] = transport
    started = time.perf_counter()
    try:
        with httpx.Client(**client_kwargs) as client:
            resp = client.post(url, json=payload, headers=headers)
    except httpx.TimeoutException as exc:
        raise ChatError("请求超时（>{:.0f}s），检查网络或代理".format(timeout)) from exc
    except httpx.HTTPError as exc:
        # exc 的内容不含 key，但为统一口径，只保留异常类型名
        raise ChatError("网络错误: {}: {}".format(type(exc).__name__, exc)) from exc
    elapsed_ms = (time.perf_counter() - started) * 1000

    if resp.status_code != 200:
        # 故意不带 body：401/403 的 body 可能回显 key（哪怕是打码形式），
        # 验收标准是「报错里不包含 key 本身」，所以只报状态码 + 排查提示。
        hint = "常见原因：key 无效 / 额度不足 / base_url 层级填错"
        if resp.status_code == 404:
            hint = "多半是 base_url 层级填错（看 .env.example 顶部的对照表）"
        raise ChatError("HTTP {}: {}".format(resp.status_code, hint))

    try:
        data = resp.json()
    except ValueError as exc:
        raise ChatError("响应不是合法 JSON（HTTP 200）") from exc
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ChatError("响应结构异常，取不到 choices[0].message.content") from exc
    if not isinstance(content, str) or not content.strip():
        raise ChatError("HTTP 200 但 content 为空")
    usage = data.get("usage") or {}
    return ChatResult(
        text=content,
        elapsed_ms=elapsed_ms,
        prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
        completion_tokens=int(usage.get("completion_tokens", 0) or 0),
    )


def main() -> int:
    try:
        settings = load_settings()
    except MissingConfigError as exc:
        print("[配置错误] {}".format(exc), file=sys.stderr)
        return 2
    # 打印口径：model 和 url 可以打，key 任何形式都不打
    print("接口: {}".format(build_url(settings.base_url)))
    print("模型: {}".format(settings.model))
    print("----")
    try:
        result = chat(settings, "用一句话说明什么是回归测试")
    except ChatError as exc:
        print("[调用失败] {}".format(exc), file=sys.stderr)
        return 1
    print("回答: {}".format(result.text.strip()))
    print("耗时: {:.0f} ms".format(result.elapsed_ms))
    print("token: 输入 {} / 输出 {}".format(result.prompt_tokens, result.completion_tokens))
    return 0


if __name__ == "__main__":
    sys.exit(main())
