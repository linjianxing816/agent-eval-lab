"""Agent 执行循环：问模型 → 执行工具 → 回填结果 → 再问，直到模型不再调工具。

【messages 三条硬规则 —— 写错任何一条，API 会报错或行为异常】
1. **assistant 消息原样回填**。模型返回的 message（含 tool_calls 数组）整个放进 messages，
   不许重新构造。`tool_call_id` 是模型生成的，你重建就变成新的 id，下一轮 tool 消息的
   `tool_call_id` 对不上，API 直接报错。
2. **tool 消息必须带 tool_call_id，content 必须是字符串**（`json.dumps`），不能直接塞 dict。
3. **每轮记 prompt_tokens** 到 `per_step_tokens`。D15/D16 要分析上下文膨胀，
   现在不留数据到时候没有。D2 不做截断（工具返回都很小）。

【退出条件】只看 tool_calls 列表，**不看 content 是否为 None**（D0 实测 `''` 和 `None` 都出现过）。
空 tool_calls + 空 content 算正常完成，content 为空交给判定层判 `empty_output`——
循环层做业务判断，就和"工具里拦金额"犯同一个错：把要测的东西提前屏蔽了。

【loop 检测】
- 签名 = 工具名 + 规范化参数（`json.dumps(sort_keys=True)`）。不排序的话
  `{"a":1,"b":2}` 和 `{"b":2,"a":1}` 会被当成不同调用，循环永远检测不到。
- **参数解析失败时用原始字符串当签名**：模型可能一直返回坏 JSON，回填 error 让它重试是对的，
  但连续 3 次同一工具返回同样的坏字符串就是循环，要停。
- **"连续"用计数器**，不是"总共出现 3 次"，否则 A→B→A→B→A 会误判。
- 第 3 次照常执行，但轨迹标 `repeated=True`——否则 D3 看到 3 条一样的轨迹，
  不知道哪条是"被判循环"的那条。

【并行 tool_calls 串行化】
模型一次要 3 个调用，这里按返回顺序逐个执行——第二个能看到第一个的效果
（先 cancel 再 refund 会成功）。这是**简化**：真实并发系统里这两个可能同时发生、
互相竞争，结果不确定。我们的 mock 环境是单线程确定的，所以这么写。

【ChatError 不吞、也不丢轨迹】
`sandbox` 是外部传入的，抛异常时轨迹天然保留在调用方手里。**不要改成在循环内部 new Sandbox**，
否则跑到一半断了，前面几步做过什么就查不到了。
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from evalkit.client import ChatError, ModelClient
from evalkit.env.sandbox import Sandbox

DEFAULT_MAX_STEPS = 12
LOOP_THRESHOLD = 3


@dataclass(frozen=True)
class LoopResult:
    steps: int                    # 实际用了几步
    exit_reason: str              # no_tool_calls | step_limit | loop_detected
    final_content: str            # 最后一轮模型说的话（可能为空串）
    tool_calls_total: int
    messages: list[dict]          # 完整对话，D14 轨迹查看器要用
    per_step_tokens: list[int]    # 每轮请求的输入 token，D15 分析上下文膨胀
    per_step_completion_tokens: list[int]   # 每轮输出的 token
    per_step_ms: list[float]      # 每轮 LLM 往返耗时，D15 测每步延迟要用


def call_signature(name: str, args: dict) -> str:
    """工具调用的指纹。参数排序后再序列化，否则键序不同会被当成两次不同的调用。"""
    try:
        body = json.dumps(args, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        body = repr(args)
    return "{0}|{1}".format(name, body)


def _parse_call(call: dict) -> tuple[str, dict | None, str | None]:
    """拆一个 tool_call。返回 (工具名, 参数字典, 原始字符串)。

    arguments 是 JSON 字符串，要 json.loads。解析失败时 args 为 None、raw 保留原串，
    交给 sandbox 兜底成"参数必须是 JSON 对象"的错误再回填给模型重试。
    """
    if not isinstance(call, dict):
        return "", None, ""
    fn = call.get("function") or {}
    name = fn.get("name") or ""
    raw = fn.get("arguments")
    if isinstance(raw, dict):
        return name, raw, None
    if not isinstance(raw, str) or not raw.strip():
        return name, None, "" if raw is None else str(raw)
    try:
        parsed = json.loads(raw)
    except ValueError:
        return name, None, raw
    if not isinstance(parsed, dict):
        return name, None, raw
    return name, parsed, None


def _signature(name: str, args: dict | None, raw: str | None) -> str:
    if args is None:
        return "{0}|raw:{1}".format(name, raw)
    return call_signature(name, args)


def run_task(client: ModelClient, sandbox: Sandbox, *, instruction: str, system_prompt: str,
             tools: list[dict], max_steps: int = DEFAULT_MAX_STEPS) -> LoopResult:
    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": instruction},
    ]
    per_step_tokens: list[int] = []
    per_step_completion: list[int] = []
    per_step_ms: list[float] = []
    last_sig: str | None = None
    repeat_count = 0
    exit_reason = "step_limit"
    final_content = ""
    steps = 0

    for step in range(1, max_steps + 1):
        steps = step
        try:
            reply = client.chat(messages, tools)
        except ChatError as exc:
            # 带上步数：D4 记 request_error 时要区分「跑到一半断了」和「第一步就没通」——
            # 前者说明前几步的动作已经发生在环境里，后者什么都没做。
            # **status 必须原样带走**：包装时丢了它，上层就只能看到 request_error，
            # 429（限流）和超时（网络）分不开，并发扫描没法判断哪一档把上游打爆了。
            raise ChatError("第 {0} 步请求失败: {1}".format(step, exc),
                            status=getattr(exc, "status", None)) from exc
        per_step_tokens.append(int(reply.usage.get("prompt_tokens", 0) or 0))
        per_step_completion.append(int(reply.usage.get("completion_tokens", 0) or 0))
        per_step_ms.append(float(reply.elapsed_ms))
        message = reply.message
        messages.append(message)                       # 规则 1：原样回填，不重建
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            exit_reason = "no_tool_calls"
            final_content = message.get("content") or ""
            break
        hit_loop = False
        for index, call in enumerate(tool_calls):
            name, args, raw = _parse_call(call)
            sig = _signature(name, args, raw)
            if sig == last_sig:
                repeat_count += 1
            else:
                last_sig, repeat_count = sig, 1
            repeated = repeat_count >= LOOP_THRESHOLD
            result = sandbox.call(name, args if raw is None else raw, step, repeated=repeated)
            messages.append({
                "role": "tool",
                "tool_call_id": call.get("id") or "call_{0}_{1}".format(step, index),
                "content": json.dumps(result, ensure_ascii=False),   # 规则 2：必须是字符串
            })
            hit_loop = hit_loop or repeated
        if hit_loop:
            exit_reason = "loop_detected"
            break

    return LoopResult(
        steps=steps,
        exit_reason=exit_reason,
        final_content=final_content,
        tool_calls_total=sandbox.tool_calls_total,
        messages=messages,
        per_step_tokens=per_step_tokens,
        per_step_completion_tokens=per_step_completion,
        per_step_ms=per_step_ms,
    )
