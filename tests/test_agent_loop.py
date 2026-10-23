"""Agent 执行循环：状态机、步数口径、messages 回填、loop 检测。

全用 FakeClient 驱动，不联网不花钱。
"""
from __future__ import annotations

import json

import pytest

from evalkit.agent.loop import call_signature, run_task
from evalkit.agent.strong import STRONG_AGENT
from evalkit.client import ChatError, FakeClient
from evalkit.env.sandbox import Sandbox
from evalkit.env.state import EnvState


def _call(name="get_order", arguments='{"order_id": "A123"}', call_id="call_1"):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def _reply(content=None, calls=None):
    message = {"role": "assistant", "content": content}
    if calls is not None:
        message["tool_calls"] = calls
    return message


def _run(client, max_steps=12):
    sandbox = Sandbox(EnvState())
    result = run_task(
        client, sandbox,
        instruction="帮我查订单 A123",
        system_prompt=STRONG_AGENT["system_prompt"],
        tools=STRONG_AGENT["tools"],
        max_steps=max_steps,
    )
    return result, sandbox


def test_no_tool_calls_exits_normally():
    result, _ = _run(FakeClient([_reply("订单 A123 已发货。")]))
    assert result.exit_reason == "no_tool_calls"
    assert result.steps == 1
    assert result.final_content == "订单 A123 已发货。"
    assert result.tool_calls_total == 0


def test_empty_content_and_no_tool_calls_is_normal_exit():
    # content 为空串也算正常完成，交给判定层判 empty_output
    result, _ = _run(FakeClient([_reply("")]))
    assert result.exit_reason == "no_tool_calls"
    assert result.final_content == ""


def test_assistant_message_is_echoed_verbatim():
    first = _reply(None, [_call()])
    result, _ = _run(FakeClient([first, _reply("查完了")]))
    # 规则 1：原样回填，tool_call_id 必须还是模型给的那个
    assert result.messages[2] is first
    assert result.messages[3]["role"] == "tool"
    assert result.messages[3]["tool_call_id"] == "call_1"
    assert isinstance(result.messages[3]["content"], str)
    assert "shipped" in result.messages[3]["content"]


def test_step_limit_when_args_always_differ():
    calls = [_call(arguments='{"order_id": "A12%d"}' % i, call_id="c%d" % i) for i in range(4)]
    client = FakeClient([_reply(None, [c]) for c in calls])
    result, sandbox = _run(client, max_steps=3)
    assert result.exit_reason == "step_limit"
    assert result.steps == 3
    assert len(result.per_step_tokens) == 3
    assert not any(step.repeated for step in sandbox.trajectory())


def test_loop_detected_on_third_identical_call():
    same = [_reply(None, [_call(call_id="c%d" % i)]) for i in range(3)]
    result, sandbox = _run(FakeClient(same), max_steps=12)
    assert result.exit_reason == "loop_detected"
    assert result.steps == 3
    # 第 3 次照常执行，但标 repeated=True
    assert [s.repeated for s in sandbox.trajectory()] == [False, False, True]


def test_alternating_calls_are_not_a_loop():
    # A→B→A→B 不是循环：用计数器比"连续"，不能比"总共出现几次"
    a = _call(arguments='{"order_id": "A123"}', call_id="a")
    b = _call(arguments='{"order_id": "B456"}', call_id="b")
    replies = [_reply(None, [a]), _reply(None, [b]), _reply(None, [a]), _reply(None, [b]), _reply("done")]
    result, sandbox = _run(FakeClient(replies), max_steps=12)
    assert result.exit_reason == "no_tool_calls"
    assert not any(step.repeated for step in sandbox.trajectory())


def test_malformed_arguments_are_refilled_and_loop_detected():
    # 一直返回同样的坏 JSON：回填 error 让模型重试，但连续 3 次就该停
    bad = [_reply(None, [_call(arguments="{order_id: A123}", call_id="c%d" % i)]) for i in range(3)]
    result, sandbox = _run(FakeClient(bad), max_steps=12)
    assert result.exit_reason == "loop_detected"
    tool_message = json.loads(result.messages[3]["content"])
    assert "参数必须是 JSON 对象" in tool_message["error"]
    assert sandbox.trajectory()[0].args == {"_raw": "{order_id: A123}"}


def test_chat_error_carries_step_number():
    client = FakeClient([_reply(None, [_call()]), _reply("done")])

    def boom(messages, tools):
        raise ChatError("HTTP 429: 限流")

    client.chat = boom
    with pytest.raises(ChatError) as exc:
        _run(client)
    assert "第 1 步请求失败" in str(exc.value)


def test_call_signature_normalizes_key_order():
    assert call_signature("get_order", {"a": 1, "b": 2}) == call_signature("get_order", {"b": 2, "a": 1})
    assert call_signature("get_order", {"a": 1}) != call_signature("get_user", {"a": 1})
