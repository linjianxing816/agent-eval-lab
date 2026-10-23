"""smoke_tool_calls 的离线单测：判定逻辑全走纯函数，最后一次 MockTransport 验证 content=None 能解析。

不联网、不花钱。真实模型是否支持 tool_calls 只能靠真调用验证，单测只管「判定对不对」。
"""
from __future__ import annotations

import sys
from pathlib import Path

import httpx

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import smoke_tool_calls as stc  # noqa: E402


def _msg(content=None, calls=None):
    message = {"role": "assistant", "content": content}
    if calls is not None:
        message["tool_calls"] = calls
    return message


def _call(name="get_order", arguments='{"order_id": "A123"}', call_id="call_1"):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def test_pass_structured_tool_calls():
    report = stc.inspect_message(_msg(None, [_call()]))
    assert report.passed
    assert report.calls[0].name == "get_order"
    assert report.calls[0].arguments == {"order_id": "A123"}


def test_fake_call_in_text():
    # 最阴的失败：看着像成功，实际 D2 循环拿不到 tool_calls
    text = '好的，我调用 get_order：{"name": "get_order", "arguments": {"order_id": "A123"}}'
    report = stc.inspect_message(_msg(text, []))
    assert report.verdict == "FAKE_CALL_IN_TEXT"
    assert not report.passed


def test_no_tool_calls_plain_text():
    report = stc.inspect_message(_msg("订单 A123 当前已发货，金额 299 元。", []))
    assert report.verdict == "NO_TOOL_CALLS"


def test_arguments_not_json():
    report = stc.inspect_message(_msg(None, [_call(arguments="{order_id: A123}")]))
    assert report.verdict == "ARGUMENTS_NOT_JSON"


def test_missing_required_argument():
    report = stc.inspect_message(_msg(None, [_call(arguments='{"id": "A123"}')]))
    assert report.verdict == "MISSING_ARGUMENT"


def test_unknown_tool_rejected():
    report = stc.inspect_message(_msg(None, [_call(name="delete_order")]))
    assert report.verdict == "UNKNOWN_TOOL"


def test_request_with_tools_parses_content_none():
    # tool_calls 场景下 content 常为 None：smoke_llm.chat() 会在这抛错，本脚本不能
    payload = {
        "choices": [{"message": _msg(None, [_call()])}],
        "usage": {"prompt_tokens": 168, "completion_tokens": 52},
    }
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
    settings = stc.Settings(base_url="https://api.deepseek.com", api_key="sk-test-not-real", model="deepseek-chat")
    message, elapsed_ms, usage = stc.request_with_tools(settings, "查订单 A123", transport=transport)
    assert stc.inspect_message(message).passed
    assert usage["prompt_tokens"] == 168
    assert elapsed_ms >= 0
