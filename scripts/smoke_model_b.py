"""第二个模型的自检：只验「会不会走结构化 tool_calls」，不做评测。

为什么要单独验：不同厂商对 OpenAI 兼容层的实现有差异——有的把工具调用写进正文，
有的漏 `id`，有的把 arguments 变成对象而不是字符串。这些差异会让评测结果全部作废，
所以先用一次调用确认协议对得上。

判据（三条全过才算 PASS）：
  1. message.tool_calls 是非空 list
  2. 每个 call 有 id + function.name（在允许列表内）
  3. function.arguments 是合法 JSON 对象，且含 order_id

退出码：0 PASS / 2 配置缺失 / 3 协议不合格

用法：
    python scripts/smoke_model_b.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evalkit.agent.full import FULL_AGENT
from evalkit.client import ChatError, DeepSeekClient, MissingConfigError, load_model_b_settings

MESSAGES = [
    {"role": "system", "content": FULL_AGENT["system_prompt"]},
    {"role": "user", "content": "用户 U1001 要退订单 A123，先查一下这单现在什么状态"},
]
ALLOWED = {t["function"]["name"] for t in FULL_AGENT["tools"]}


def main() -> int:
    try:
        settings = load_model_b_settings()
    except MissingConfigError as exc:
        print("配置缺失:", exc)
        print("请在 .env 里填 MODEL_B_BASE_URL / MODEL_B_API_KEY / MODEL_B_MODEL")
        return 2

    print("model:", settings.model, "| base_url:", settings.base_url)
    client = DeepSeekClient(settings, thinking=False, max_tokens=512, timeout=60.0)
    try:
        reply = client.chat(MESSAGES, FULL_AGENT["tools"])
    except ChatError as exc:
        print("调用失败:", exc)
        return 1

    calls = reply.message.get("tool_calls") or []
    content = reply.message.get("content") or ""
    # A 那边是 thinking 关掉的。B 如果返回 reasoning_content，说明它在思考模式，
    # 两个模型的延迟/成本/输出长度都不可比，评测结论就作废。
    reasoning = reply.message.get("reasoning_content")
    print("tool_calls:", len(calls), "| content 长度:", len(content),
          "| reasoning_content:", "无" if not reasoning else "{0} 字".format(len(str(reasoning))))
    for call in calls:
        fn = call.get("function") or {}
        print("  -", call.get("id"), fn.get("name"), (fn.get("arguments") or "")[:80])

    problems = []
    if reasoning:
        problems.append("返回里带 reasoning_content（模型在思考模式，和 A 的条件不一致）")
    if not calls:
        problems.append("tool_calls 为空")
        if '{"name"' in content.replace(" ", ""):
            problems.append("正文里出现了伪 tool_calls（模型把调用写成了文本）")
    for call in calls:
        fn = call.get("function") or {}
        if not call.get("id"):
            problems.append("缺少 id")
        if fn.get("name") not in ALLOWED:
            problems.append("工具名不在允许列表: " + str(fn.get("name")))
        try:
            args = json.loads(fn.get("arguments") or "")
        except ValueError:
            problems.append("arguments 不是合法 JSON")
            continue
        if not isinstance(args, dict) or "order_id" not in args:
            problems.append("arguments 缺 order_id")

    if problems:
        print("\nFAIL:", "; ".join(sorted(set(problems))))
        return 3
    print("\nPASS：协议对得上，可以跑全量")
    return 0


if __name__ == "__main__":
    sys.exit(main())
