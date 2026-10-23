"""结果层：按 `success_criteria` 的 path 取最终状态，逐条比对。

**永远不抛异常。** 路径解析失败（`KeyError` / `IndexError` / `TypeError`）判失败，
detail 写「路径 refunds[-1].amount 无法解析，可能 Agent 没有执行对应操作」。

理由：判定层是尺子。尺子量不了东西，只能说「这里没有东西」，不能说「尺子坏了」。
崩掉的后果是分不清「模型没做事」和「我代码有 bug」。

**金额比较两边都 round(x, 2)**：`158.0 * 0.9 = 142.20000000000002 != 142.2`。
不 round 会把「判定写错」伪装成「模型算错」。
"""
from __future__ import annotations

from typing import Any, NamedTuple

MONEY_DIGITS = 2


class Verdict(NamedTuple):
    """三层的统一返回。`passed=True` + `category` 非空 = 通过但记警告。

    定义在 outcome.py 而不是 __init__.py，是因为 __init__ 要先 import 三个子模块，
    子模块再回头 import __init__ 里的 Verdict 就是循环导入。__init__ 会把它再导出一遍，
    外部仍然只写 `from evalkit.judge import Verdict`。
    """

    passed: bool
    category: str | None
    detail: str


def split_path(path: str) -> list[str | int]:
    """把 `"refunds[-1].amount"` 拆成 `["refunds", -1, "amount"]`。"""
    tokens: list[str | int] = []
    buf = ""
    i = 0
    while i < len(path):
        ch = path[i]
        if ch == ".":
            if buf:
                tokens.append(buf)
                buf = ""
        elif ch == "[":
            if buf:
                tokens.append(buf)
                buf = ""
            end = path.find("]", i)
            if end < 0:
                raise ValueError("路径 {0} 的方括号没有闭合".format(path))
            tokens.append(int(path[i + 1:end]))
            i = end
        else:
            buf += ch
        i += 1
    if buf:
        tokens.append(buf)
    return tokens


def resolve_path(data: Any, path: str) -> Any:
    """按 path 取值。取不到就让它抛 KeyError / IndexError / TypeError，由调用方兜。"""
    current = data
    for token in split_path(path):
        if isinstance(token, int):
            if not isinstance(current, (list, tuple)):
                raise TypeError("{0} 不是列表，不能用下标 [{1}]".format(type(current).__name__, token))
            current = current[token]
        else:
            if not isinstance(current, dict):
                raise TypeError("{0} 不是对象，取不到字段 {1}".format(type(current).__name__, token))
            current = current[token]
    return current


def values_match(expected: Any, actual: Any) -> bool:
    """数字一律 round 到两位再比；其他类型直接比。"""
    if isinstance(expected, bool) or isinstance(actual, bool):
        return expected == actual
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return round(float(expected), MONEY_DIGITS) == round(float(actual), MONEY_DIGITS)
    return expected == actual


# 报告里给人看的字段名。取不到就退回路径本身。
FIELD_LABELS = {
    "status": "状态",
    "amount": "金额",
    "refund_amount": "退款金额",
    "user": "用户",
    "vip": "VIP",
    "name": "姓名",
}


def field_label(path: str) -> str:
    return FIELD_LABELS.get(path.rsplit(".", 1)[-1], path)


def check_outcome(state: dict, criteria: list[dict]) -> Verdict:
    """逐条比对，**收集全部不通过项**。

    只报第一条的话，D4 统计"失败原因分布"就是错的——一条任务同时错状态和错金额时，
    你只知道第一个。所以这里全查一遍，detail 里全列出来。
    `category` 仍是单一值（取第一条失败的类别），方便按类别计数。
    """
    if not criteria:
        return Verdict(True, None, "success_criteria 为空，结果层默认通过")
    failures: list[tuple[str, str]] = []       # (单条 detail, 汇总用的短句)
    for item in criteria:
        path = item.get("path", "")
        expected = item.get("expect")
        try:
            actual = resolve_path(state, path)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            failures.append((
                "路径 {0} 无法解析（{1}: {2}），可能 Agent 没有执行对应操作".format(
                    path, type(exc).__name__, exc
                ),
                "路径 {0} 无法解析（{1}）".format(field_label(path), type(exc).__name__),
            ))
            continue
        if not values_match(expected, actual):
            failures.append((
                "路径 {0} 期望 {1!r}，实际 {2!r}".format(path, expected, actual),
                "{0} expected {1!r} got {2!r}".format(field_label(path), expected, actual),
            ))
    if not failures:
        return Verdict(True, None, "{0} 条 success_criteria 全部通过".format(len(criteria)))
    if len(failures) == 1:
        # 单条保持原来的说法，"1 条不通过: ..." 读着别扭
        return Verdict(False, "wrong_final_state", failures[0][0])
    return Verdict(
        False,
        "wrong_final_state",
        "{0} 条不通过: {1}".format(len(failures), "; ".join(short for _, short in failures)),
    )
