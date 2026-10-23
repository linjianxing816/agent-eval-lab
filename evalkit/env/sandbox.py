"""工具执行沙箱：调工具 + 每次调用落一条轨迹。被测 Agent 改环境只能经过这里。

【两种 error，永远别混（D3 判基建失败时按这张表读）】
| 情况       | TrajectoryStep.error | result["error"] |
| 业务失败   | None                 | 有              |
| 工具抛异常 | 有                   | 有              |
| 成功       | None                 | 无              |
**D3 判「是不是基建失败」只看 `TrajectoryStep.error`，永远不许看 `result["error"]`**
——业务失败也有这个键。不写死这条，两种失败会被一起归成"环境不稳定"，
你会去查 mock 环境，而真正的问题是判定层读错了字段。
- 业务失败 = 模型操作不对 → 算模型的错，进失败率统计
- 基建失败 = mock 环境代码炸了 → 不该算模型的错，要单独剔除

【其他】args 落轨迹前 deepcopy；result 完整存不截断；未知工具 / 参数缺漏 / args 不是 dict
全部落轨迹；计时用 perf_counter 不用 time.time；这里**不做** max_steps 和循环检测（D2/D3）。
"""
from __future__ import annotations

import copy
import inspect
import time
from dataclasses import asdict, dataclass
from typing import Any

from evalkit.env.state import EnvState
from evalkit.env.tools import TOOL_REGISTRY


@dataclass(frozen=True)
class TrajectoryStep:
    step: int            # 第几次 LLM 往返；并行 tool_calls 共用同一个 step
    tool: str
    args: dict
    result: dict
    elapsed_ms: float
    error: str | None    # 只在「工具内部抛异常」时非空，业务失败留 None

    def to_dict(self) -> dict:
        return asdict(self)


class Sandbox:
    def __init__(self, state: EnvState | None = None) -> None:
        self._state = EnvState() if state is None else state
        self._steps: list[TrajectoryStep] = []

    @property
    def state(self) -> EnvState:
        return self._state

    @property
    def tool_calls_total(self) -> int:
        """工具调用总次数。和步数不是一个东西：一步可以调多个工具。"""
        return len(self._steps)

    def reset(self, initial: dict | None = None) -> None:
        """新任务 = 干净环境 + 空轨迹。"""
        self._state = EnvState(initial)
        self._steps.clear()

    def trajectory(self) -> list[TrajectoryStep]:
        """返回新 list，不把内部列表交出去。"""
        return list(self._steps)

    def clear_trajectory(self) -> None:
        self._steps.clear()

    def call(self, name: str, args: Any, step: int) -> dict:
        """执行一次工具调用。永远返回 dict，不抛异常。"""
        started = time.perf_counter()
        result, error = self._execute(name, args)
        elapsed_ms = (time.perf_counter() - started) * 1000
        self._steps.append(
            TrajectoryStep(
                step=step,
                tool=name,
                args=copy.deepcopy(args) if isinstance(args, dict) else {"_raw": args},
                result=copy.deepcopy(result),
                elapsed_ms=elapsed_ms,
                error=error,
            )
        )
        return result

    def _execute(self, name: str, args: Any) -> tuple[dict, str | None]:
        # 模型返回的 arguments 是 JSON 字符串，D2 要 json.loads 才变 dict。解析失败时
        # args 可能是 str 或 None，不兜住就会 TypeError 崩在沙箱里，然后你会以为是模型的错。
        # 这是模型传参错，不是基建错 → error 留 None。
        if not isinstance(args, dict):
            kind = type(args).__name__
            return {"error": "参数必须是 JSON 对象，收到的是 {0}，请检查 arguments 格式".format(kind)}, None
        if name not in TOOL_REGISTRY:
            available = ", ".join(sorted(TOOL_REGISTRY))
            return {"error": "工具 {0} 不存在。可用工具：{1}".format(name, available)}, None
        func = TOOL_REGISTRY[name]
        try:
            inspect.signature(func).bind(self._state, **args)
        except TypeError:
            return {"error": self._arg_message(name, func, args)}, None
        try:
            return func(self._state, **args), None
        except Exception as exc:  # noqa: BLE001 —— 兜住所有工具异常，但必须记录，绝不静默
            detail = "{0}: {1}".format(type(exc).__name__, exc)
            return {
                "error": "工具 {0} 执行失败：{1}。重试可能无效，请换一种方式或直接报告。".format(name, detail)
            }, detail

    @staticmethod
    def _arg_message(name: str, func: Any, args: dict) -> str:
        """把 bind() 的 TypeError 翻成人话——Agent 靠这句话决定下一步。"""
        params = [p for p in inspect.signature(func).parameters if p != "state"]
        missing = [p for p in params if p not in args]
        extra = [k for k in args if k not in params]
        if missing:
            return "工具 {0} 缺少必填参数 {1}，请补上后重试（可用参数：{2}）。".format(
                name, ", ".join(missing), ", ".join(params))
        if extra:
            return "工具 {0} 不接受参数 {1}，可用参数为 {2}。".format(name, ", ".join(extra), ", ".join(params))
        return "工具 {0} 参数不合法，可用参数为 {1}。".format(name, ", ".join(params))
