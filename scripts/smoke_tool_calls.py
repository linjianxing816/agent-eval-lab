"""D0 最后一步：验证模型返回的是「结构化 tool_calls」，而不是在正文里写一段 JSON。

为什么不复用 smoke_llm.chat()：模型走 tool_calls 时 message.content 通常是 None，
而 smoke_llm.chat() 把空 content 当 ChatError 抛。那边已过 11 个测试，不动它，
这里只复用 load_settings / build_url / Settings / ChatError。

判定口径（三条硬判据，全过才算 PASS）：
  1. message.tool_calls 是非空 list
  2. 每个 call 有 id、有 function.name 且在允许列表内；type 若存在必须是 "function"
     （type 缺失不致命——D2 回填只用 id/name/arguments；存在但不是 function 说明协议不对）
  3. function.arguments 是合法 JSON 对象，且含必填参数 order_id

另有一条专门识别「假装调用」：tool_calls 为空、但 content 里出现 {"name": "xxx"} → FAKE_CALL_IN_TEXT。

退出码：0 PASS / 1 调用失败 / 2 配置错误 / 3 模型不支持或返回格式不合格

thinking 模式：DeepSeek 默认开启（effort=high），本脚本显式关闭，理由见 THINKING_DISABLED 上方注释。

密钥安全：key 只从 .env 读；打印和报错里都不含 key，HTTP 非 200 只报状态码（body 可能回显 key）。

Python 3.11 兼容：不用 3.12+ 语法（PEP 701 f-string 同种引号嵌套），格式化统一 str.format()。
"""
from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from smoke_llm import (  # noqa: E402
    ChatError,
    MissingConfigError,
    Settings,
    build_url,
    load_settings,
)

DEFAULT_TIMEOUT = 30.0
CONNECT_TIMEOUT = 10.0
# thinking 默认是开的（effort=high），这里显式关掉。理由三条：
#   1. 开着 thinking，每步延迟里混着 CoT 生成时间，D15 的 TTFT / 每步延迟口径被污染
#   2. CoT 算输出 token，按输出价计费，成本直接翻倍
#   3. thinking 模式下 temperature 不生效（不报错但无效），D4 的稳定性实验失去前提
# 关掉之后：temperature=0 生效、reasoning_content 不再返回、回填时无需带它。
THINKING_DISABLED = {"type": "disabled"}
PROMPT = "帮我查一下订单 A123 现在是什么状态"
TOOL_RESULT = '{"order_id": "A123", "user": "U1001", "amount": 299.0, "status": "shipped"}'
ALLOWED_TOOLS = ("get_order",)
REQUIRED_ARGS = ("order_id",)

GET_ORDER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_order",
        "description": "按订单号查询订单详情（下单用户、金额、状态）",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "订单号，如 A123"},
            },
            "required": ["order_id"],
        },
    },
}

# 正文里的 {"name": "get_order", ...} 是最典型的「假装调用」形态
FAKE_CALL_PATTERN = re.compile(r'"name"\s*:\s*"([A-Za-z_][A-Za-z0-9_]*)"')
PROBLEM_PRIORITY = (
    "MISSING_FIELD",
    "UNKNOWN_TOOL",
    "ARGUMENTS_NOT_JSON",
    "ARGUMENTS_NOT_OBJECT",
    "MISSING_ARGUMENT",
)
FAIL_HINT = {
    "FAKE_CALL_IN_TEXT": "换模型：通义 qwen-plus 或 Moonshot moonshot-v1-8k（都支持 function calling）",
    "NO_TOOL_CALLS": "模型没意识到该调工具，可先换更强模型再试；仍不行就换厂商",
    "ARGUMENTS_NOT_JSON": "会调工具但参数不是合法 JSON，D2 解析必崩；先换模型，不要加修复层掩盖",
}


@dataclass(frozen=True)
class ToolCall:
    index: int
    id: str
    name: str
    arguments_raw: str
    arguments: dict | None
    problem: str  # "" 表示这条没问题
    detail: str


@dataclass(frozen=True)
class ToolCallReport:
    verdict: str
    reason: str
    calls: list[ToolCall]
    content: str | None
    reasoning: str | None = None

    @property
    def passed(self) -> bool:
        return self.verdict == "PASS"


def _bad(index, call_id, name, raw, problem, detail):
    return ToolCall(index, call_id, name, raw, None, problem, detail)


def _inspect_call(index: int, call: dict, allowed_tools) -> ToolCall:
    """单条 tool_call 的三条硬判据。返回 problem 为空表示这条合格。"""
    if not isinstance(call, dict):
        return _bad(index, "", "", "", "MISSING_FIELD", "tool_calls[{0}] 不是对象".format(index))
    call_id = call.get("id")
    if not isinstance(call_id, str) or not call_id.strip():
        return _bad(index, "", "", "", "MISSING_FIELD", "缺 id，回填工具结果时没法对应")
    call_type = call.get("type")
    if call_type is not None and call_type != "function":
        return _bad(index, call_id, "", "", "MISSING_FIELD", "type={0}，期望 function".format(call_type))
    fn = call.get("function")
    if not isinstance(fn, dict):
        return _bad(index, call_id, "", "", "MISSING_FIELD", "缺 function 对象")
    name = fn.get("name")
    if not isinstance(name, str) or not name.strip():
        return _bad(index, call_id, "", "", "MISSING_FIELD", "缺 function.name")
    if name not in allowed_tools:
        return _bad(index, call_id, name, "", "UNKNOWN_TOOL", "name={0} 不在允许列表".format(name))
    raw = fn.get("arguments")
    if isinstance(raw, dict):
        # 少数厂商直接回 dict（非标准），也认
        arguments, raw_text = raw, json.dumps(raw, ensure_ascii=False)
    elif isinstance(raw, str) and raw.strip():
        raw_text = raw
        try:
            parsed = json.loads(raw)
        except ValueError as exc:
            return _bad(index, call_id, name, raw_text, "ARGUMENTS_NOT_JSON", "arguments 不是合法 JSON: {0}".format(exc))
        if not isinstance(parsed, dict):
            return _bad(index, call_id, name, raw_text, "ARGUMENTS_NOT_OBJECT", "arguments 解析出 {0}，不是对象".format(type(parsed).__name__))
        arguments = parsed
    else:
        return _bad(index, call_id, name, "", "ARGUMENTS_NOT_JSON", "arguments 为空或不是字符串")
    missing = [arg for arg in REQUIRED_ARGS if arg not in arguments]
    if missing:
        return ToolCall(index, call_id, name, raw_text, arguments, "MISSING_ARGUMENT", "缺参数: " + ", ".join(missing))
    return ToolCall(index, call_id, name, raw_text, arguments, "", "OK")


def inspect_message(message: dict, allowed_tools=ALLOWED_TOOLS) -> ToolCallReport:
    """纯函数：所有判定都在这，单测只测它，不联网。"""
    content = message.get("content")
    reasoning = message.get("reasoning_content")
    if not isinstance(reasoning, str):
        reasoning = None
    raw_calls = message.get("tool_calls") or []
    if not isinstance(raw_calls, list) or not raw_calls:
        text = content if isinstance(content, str) else ""
        hit = FAKE_CALL_PATTERN.search(text)
        if hit:
            return ToolCallReport(
                "FAKE_CALL_IN_TEXT",
                'tool_calls 为空，但正文里出现 {{"name": "{0}"}} —— 模型在假装调用'.format(hit.group(1)),
                [],
                content,
                reasoning,
            )
        return ToolCallReport("NO_TOOL_CALLS", "tool_calls 为空，模型直接给了文字回答", [], content, reasoning)
    calls = [_inspect_call(i, call, allowed_tools) for i, call in enumerate(raw_calls)]
    problems = [c.problem for c in calls if c.problem]
    if problems:
        verdict = sorted(problems, key=PROBLEM_PRIORITY.index)[0]
        first = next(c for c in calls if c.problem == verdict)
        return ToolCallReport(verdict, "第 {0} 个 call: {1}".format(first.index, first.detail), calls, content, reasoning)
    names = ", ".join(c.name for c in calls)
    return ToolCallReport(
        "PASS",
        "结构化 tool_calls {0} 个（{1}），name / arguments / 必填参数齐全".format(len(calls), names),
        calls,
        content,
        reasoning,
    )


def _post_chat(settings: Settings, messages: list, tools: list, timeout: float, transport):
    """发一次带 tools 的 chat 请求，返回 (message dict, elapsed_ms, usage)。失败显式抛 ChatError。"""
    payload = {
        "model": settings.model,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "temperature": 0,
        "max_tokens": 512,
        # 原始 HTTP 请求里这个参数直接放 body 顶层（SDK 里才叫 extra_body）
        "thinking": THINKING_DISABLED,
    }
    headers = {"Authorization": "Bearer " + settings.api_key, "Content-Type": "application/json"}
    kwargs = {"timeout": httpx.Timeout(timeout, connect=CONNECT_TIMEOUT)}
    if transport is not None:
        kwargs["transport"] = transport
    started = time.perf_counter()
    try:
        with httpx.Client(**kwargs) as client:
            resp = client.post(build_url(settings.base_url), json=payload, headers=headers)
    except httpx.TimeoutException as exc:
        raise ChatError("请求超时（>{0:.0f}s），检查网络或代理".format(timeout)) from exc
    except httpx.HTTPError as exc:
        raise ChatError("网络错误: {0}: {1}".format(type(exc).__name__, exc)) from exc
    elapsed_ms = (time.perf_counter() - started) * 1000
    if resp.status_code != 200:
        # 故意不带 body：401/403 的 body 可能回显 key
        hint = "常见原因：key 无效 / 额度不足 / base_url 层级填错"
        if resp.status_code == 404:
            hint = "多半是 base_url 层级填错（看 .env.example 顶部的对照表）"
        raise ChatError("HTTP {0}: {1}".format(resp.status_code, hint))
    try:
        data = resp.json()
    except ValueError as exc:
        raise ChatError("响应不是合法 JSON（HTTP 200）") from exc
    try:
        message = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ChatError("响应结构异常，取不到 choices[0].message") from exc
    if not isinstance(message, dict):
        raise ChatError("choices[0].message 不是对象")
    return message, elapsed_ms, data.get("usage") or {}


def request_with_tools(settings: Settings, prompt: str, tools=None, timeout: float = DEFAULT_TIMEOUT, transport=None):
    tools = [GET_ORDER_TOOL] if tools is None else tools
    return _post_chat(settings, [{"role": "user", "content": prompt}], tools, timeout, transport)


def follow_up(settings: Settings, prompt: str, report: ToolCallReport, tool_result: str = TOOL_RESULT,
              timeout: float = DEFAULT_TIMEOUT, transport=None):
    """第二轮：把 tool 结果回填再问一次，返回 (最终回答文本, elapsed_ms, usage)。"""
    if not report.passed:
        raise ValueError("只有 PASS 的报告才能回填")
    calls = [
        {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments_raw}}
        for c in report.calls
    ]
    assistant = {"role": "assistant", "content": report.content or "", "tool_calls": calls}
    # 带 tools 的请求在 thinking 模式下必须回填 reasoning_content，否则 API 返 400。
    # 现在 thinking 已关、这个字段不会返回，所以是「有就带上」而不是「必须有」——
    # 以后想开 thinking 做对照实验，这里不用再改。
    if report.reasoning:
        assistant["reasoning_content"] = report.reasoning
    messages = [{"role": "user", "content": prompt}, assistant]
    for call in report.calls:
        messages.append({"role": "tool", "tool_call_id": call.id, "content": tool_result})
    message, elapsed_ms, usage = _post_chat(settings, messages, [GET_ORDER_TOOL], timeout, transport)
    text = message.get("content")
    return (text if isinstance(text, str) else ""), elapsed_ms, usage


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    second_turn = "--second-turn" in argv
    try:
        settings = load_settings()
    except MissingConfigError as exc:
        print("[配置错误] {0}".format(exc), file=sys.stderr)
        return 2
    print("接口: {0}".format(build_url(settings.base_url)))
    print("模型: {0}".format(settings.model))
    print("thinking: {0}".format(THINKING_DISABLED["type"]))
    print("----")
    print("[请求] {0}   (tools: {1})".format(PROMPT, ", ".join(ALLOWED_TOOLS)))
    try:
        message, elapsed_ms, usage = request_with_tools(settings, PROMPT)
    except ChatError as exc:
        print("[调用失败] {0}".format(exc), file=sys.stderr)
        return 1
    print("[耗时] {0:.0f} ms   token: 输入 {1} / 输出 {2}".format(
        elapsed_ms, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)))
    report = inspect_message(message)
    print("[原始] message.content   = {0}".format("None" if report.content is None else repr(report.content)))
    print("[原始] reasoning_content = {0}".format("None（thinking 已关，符合预期）" if report.reasoning is None else "{} 字（thinking 没关掉！）".format(len(report.reasoning))))
    print("[原始] message.tool_calls = {0} 个".format(len(report.calls)))
    for call in report.calls:
        print("  #{0}  id={1}  name={2}".format(call.index, call.id, call.name or "(空)"))
        print("      arguments={0}   {1}".format(call.arguments_raw or "(空)", "OK" if not call.problem else call.problem))
    print("[判定] {0}".format("PASS" if report.passed else "FAIL / " + report.verdict))
    print("[理由] {0}".format(report.reason))
    if not report.passed:
        print("[下一步] {0}".format(FAIL_HINT.get(report.verdict, "换模型：通义 qwen-plus 或 Moonshot moonshot-v1-8k")))
        return 3
    if not second_turn:
        return 0
    try:
        text, ms2, _usage2 = follow_up(settings, PROMPT, report)
    except ChatError as exc:
        print("[回填失败] {0}".format(exc), file=sys.stderr)
        return 1
    print("[回填] role=tool 已发送: {0}".format(TOOL_RESULT))
    print("[收尾] {0:.0f} ms  最终回答: {1}".format(ms2, text.strip() or "(空)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
