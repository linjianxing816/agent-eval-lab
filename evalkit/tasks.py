"""任务文件：加载 + 校验。

**只校验，不生成、不执行。** 人工 40 条和 8 个模板都是用户自己写，这里负责在他写完
之后把格式问题一次全报出来——`templates/` 和 `tasks/` 里的内容不归这个模块管。

【12 项检查】
| 编号 | 级别 | 查什么 |
|---|---|---|
| E01 | error | 缺必填字段（id / instruction / tools_allowed / success_criteria） |
| E02 | error | `id` 重复 |
| E03 | error | `success_criteria` 的路径解析不了（字段名错、引用了不存在的订单号） |
| E04 | error | `success_criteria` 是空列表（这条任务永远通过） |
| E05 | error | `tools_allowed` 含不存在的工具 |
| E06 | error | `forbid_tools` 含不存在的工具 |
| E07 | error | `max_steps` 不是正整数 |
| E08 | error | JSON 行解析失败 |
| E09 | **warning** | 期望值和实际类型不匹配但无害（期望 int，实际 float） |
| E10 | error | 同一工具既在 `tools_allowed` 又在 `forbid_tools` |
| E11 | **warning** | `tools_allowed` 是空列表（可能故意的，报出来让人确认） |
| E12 | error | `instruction` 为空或超过 500 字 |
| E13 | error | 期望值和实际类型根本不同（期望 `"299"` 字符串，实际 `299.0`）——这条任务永远失败 |

E09 和 E11 是 warning 不是 error：E09 在我们的比较器下不会导致失败（数值统一 round 后
比较，`299 == 299.0`），但写法不严谨；E11 可能是故意的（测"无工具可用"）。
**E13 是 error**：字符串和数字永远不相等，任务写下去就是死的。
**warning 不挡提交，但一定要看见。**

【路径怎么校验：靠 schema，不靠实例】
`refunds[-1].amount` 在初始状态里必然 `IndexError`（`refunds` 是空列表），但它是合法路径。
如果只按初始状态解析，所有引用 `refunds[-1]` 的任务都会被误报——一个老是误报的检查器，
人就不看了。

所以用 `STATE_SCHEMA`（定义在 `evalkit/env/state.py`，紧挨着 `DEFAULT_INITIAL_STATE`）
描述 mock 环境的**结构**：
- `orders` / `users` 是"任意 key → 固定字段"的字典，用 `"*"` 通配
- `refunds` 是列表，元素结构固定为 `{order_id, amount, ts}`

这样 `refunds[-1].amount` 合法、`refunds[-1].amout` 报错，**空列表也查得出来**。
实例解析得到的值只用于两件事：确认订单号真的存在（E03）、比对类型（E09）。
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evalkit.env.state import DEFAULT_INITIAL_STATE, STATE_SCHEMA
from evalkit.env.tools import TOOL_NAMES
from evalkit.judge.outcome import split_path

REQUIRED_FIELDS = ("id", "instruction", "tools_allowed", "success_criteria")
MAX_INSTRUCTION_CHARS = 500
DEFAULT_MAX_STEPS = 12

MISSING = object()


@dataclass(frozen=True)
class Issue:
    line: int
    task_id: str
    code: str
    level: str          # error / warning
    message: str

    def format(self) -> str:
        where = "第 {0} 条".format(self.line)
        if self.task_id:
            where += "（{0}）".format(self.task_id)
        return "{0}: [{1}][{2}] {3}".format(where, self.code, self.level, self.message)


def resolve_with_schema(path: str, state: dict) -> tuple[str | None, Any]:
    """按 schema 校验路径。返回 (错误信息, 实例上的值)。

    错误信息为 None 表示路径合法。值可能是 `MISSING`——合法但实例里还没有
    （比如 `refunds[-1].amount` 在空列表上），调用方要自己判。
    """
    try:
        tokens = split_path(path)
    except ValueError as exc:
        return "路径 {0} 语法错误：{1}".format(path, exc), MISSING
    schema: Any = STATE_SCHEMA
    data: Any = state
    for token in tokens:
        if isinstance(token, int):
            if not isinstance(schema, list):
                return "路径 {0}：{1} 不是列表，不能用下标".format(path, token), MISSING
            schema = schema[0]
            if isinstance(data, list) and -len(data) <= token < len(data):
                data = data[token]
            else:
                data = MISSING
            continue
        if not isinstance(schema, dict):
            return "路径 {0}：字段 {1} 的上层不是对象".format(path, token), MISSING
        if token in schema:
            schema = schema[token]
        elif "*" in schema:
            schema = schema["*"]
        else:
            return "路径 {0}：没有字段 {1}（可用：{2}）".format(
                path, token, ", ".join(sorted(schema))
            ), MISSING
        if data is MISSING:
            continue
        if isinstance(data, dict) and token in data:
            data = data[token]
        else:
            return "路径 {0} 无法解析：初始状态里没有 {1}".format(path, token), MISSING
    return None, data


def validate_task(task: dict, line: int, seen_ids: dict[str, int]) -> list[Issue]:
    issues: list[Issue] = []

    def add(code: str, level: str, message: str) -> None:
        issues.append(Issue(line, str(task.get("id") or ""), code, level, message))

    for field in REQUIRED_FIELDS:
        if field not in task:
            add("E01", "error", "缺少 {0}".format(field))
    if issues:
        return issues          # 必填都没齐，后面的检查没有意义

    task_id = str(task["id"])
    if task_id in seen_ids:
        add("E02", "error", "id={0} 与第 {1} 条重复".format(task_id, seen_ids[task_id]))
    else:
        seen_ids[task_id] = line

    instruction = task["instruction"]
    if not isinstance(instruction, str) or not instruction.strip():
        add("E12", "error", "instruction 为空")
    elif len(instruction) > MAX_INSTRUCTION_CHARS:
        add("E12", "error", "instruction 超过 {0} 字（当前 {1} 字）".format(MAX_INSTRUCTION_CHARS, len(instruction)))

    allowed = task["tools_allowed"]
    constraints = task.get("constraints") or {}
    forbid = constraints.get("forbid_tools") or []
    if not isinstance(allowed, list):
        add("E05", "error", "tools_allowed 必须是列表")
        allowed = []
    elif not allowed:
        add("E11", "warning", "tools_allowed 是空列表，这个任务无法执行任何操作（故意的话忽略）")
    for name in allowed:
        if name not in TOOL_NAMES:
            add("E05", "error", "tools_allowed 含未知工具 {0}（可用：{1}）".format(name, ", ".join(TOOL_NAMES)))
    for name in forbid:
        if name not in TOOL_NAMES:
            add("E06", "error", "forbid_tools 含未知工具 {0}（可用：{1}）".format(name, ", ".join(TOOL_NAMES)))
    both = sorted(set(allowed) & set(forbid))
    if both:
        add("E10", "error", "{0} 同时出现在 tools_allowed 和 forbid_tools".format(", ".join(both)))

    max_steps = constraints.get("max_steps", DEFAULT_MAX_STEPS)
    if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps < 1:
        add("E07", "error", "max_steps={0} 必须是 ≥1 的整数".format(max_steps))

    criteria = task["success_criteria"]
    if not isinstance(criteria, list):
        add("E04", "error", "success_criteria 必须是列表")
    elif not criteria:
        add("E04", "error", "success_criteria 是空列表，这条任务永远通过")
    else:
        state = _merged_state(task)
        for index, item in enumerate(criteria):
            issues.extend(_check_criterion(item, index, state, line, task_id))
    return issues


def _merged_state(task: dict) -> dict:
    """任务自己的 initial_state 覆盖到默认状态上（和运行时一致）。"""
    state = json.loads(json.dumps(DEFAULT_INITIAL_STATE, ensure_ascii=False))
    initial = task.get("initial_state") or {}
    for key, value in initial.items():
        if isinstance(value, dict) and isinstance(state.get(key), dict):
            state[key].update(value)
        else:
            state[key] = value
    return state


def _check_criterion(item: dict, index: int, state: dict, line: int, task_id: str) -> list[Issue]:
    out: list[Issue] = []
    if not isinstance(item, dict) or "path" not in item or "expect" not in item:
        out.append(Issue(line, task_id, "E03", "error", "success_criteria[{0}] 必须含 path 和 expect".format(index)))
        return out
    path = str(item["path"])
    expected = item["expect"]
    error, actual = resolve_with_schema(path, state)
    if error is not None:
        out.append(Issue(line, task_id, "E03", "error", "success_criteria[{0}] {1}".format(index, error)))
        return out
    if actual is MISSING:
        return out                      # 合法但运行时才出现，类型无从比对
    if isinstance(expected, bool) or isinstance(actual, bool):
        return out
    expected_numeric = isinstance(expected, (int, float))
    actual_numeric = isinstance(actual, (int, float))
    if expected_numeric and actual_numeric:
        if type(expected) is not type(actual):
            out.append(Issue(
                line, task_id, "E09", "warning",
                "success_criteria[{0}] 路径 {1} 期望 {2}，实际状态是 {3}（无害，建议金额统一写小数）".format(
                    index, path, type(expected).__name__, type(actual).__name__
                ),
            ))
        return out
    if type(expected) is not type(actual):
        # 字符串和数字永远不相等 → 这条任务写下去就是死的
        out.append(Issue(
            line, task_id, "E13", "error",
            "success_criteria[{0}] 路径 {1} 期望 {2}（{3!r}），实际状态是 {4}（{5!r}），这条任务永远失败".format(
                index, path, type(expected).__name__, expected, type(actual).__name__, actual
            ),
        ))
    return out


def check_file(path: str | Path) -> tuple[list[dict], list[Issue]]:
    """读 jsonl，逐行校验。一次返回全部问题，不在第一个错处停。"""
    file_path = Path(path)
    if not file_path.exists():
        return [], [Issue(0, "", "E08", "error", "文件不存在: {0}".format(file_path))]
    tasks: list[dict] = []
    issues: list[Issue] = []
    seen_ids: dict[str, int] = {}
    for line_no, raw in enumerate(file_path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            task = json.loads(raw)
        except ValueError as exc:
            issues.append(Issue(line_no, "", "E08", "error", "不是合法 JSON: {0}".format(exc)))
            continue
        if not isinstance(task, dict):
            issues.append(Issue(line_no, "", "E08", "error", "这一行不是 JSON 对象"))
            continue
        issues.extend(validate_task(task, line_no, seen_ids))
        tasks.append(task)
    return tasks, issues


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if not args:
        print("用法: python -m evalkit.tasks <tasks.jsonl> [更多文件...]", file=sys.stderr)
        return 2
    exit_code = 0
    for path in args:
        tasks, issues = check_file(path)
        errors = [i for i in issues if i.level == "error"]
        warnings = [i for i in issues if i.level == "warning"]
        print("== {0}: {1} 条任务, {2} 个错误, {3} 个警告".format(path, len(tasks), len(errors), len(warnings)))
        for issue in issues:
            print("  " + issue.format())
        if errors:
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
