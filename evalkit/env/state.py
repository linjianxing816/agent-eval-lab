"""mock 环境的状态容器。

三条设计约束（改之前先看）：
1. **每个任务一个 EnvState 实例**。D16 要跑 1/2/4/8 并发扫描，全局单例会串状态。
2. **reset / snapshot / restore 全部走 deepcopy**。不深拷贝的话，任务 A 改过的状态
   会污染任务 B——这是「同一任务跑 100 次初始状态完全一致」的前提。
3. **这里只放数据，不放业务规则**。订单状态机 shipped → cancelled → refunded 的
   流转校验在 tools.py，因为「能不能改」是工具的事，不是容器的事。

订单状态只有三种，单向不可逆：
    shipped（已发货） → cancelled（已取消发货） → refunded（已退款）
"""
from __future__ import annotations

import copy
from typing import Any

ORDER_STATUSES = ("shipped", "cancelled", "refunded")

# 初始状态模板。注意 B456 故意用 158.0 —— 158.0 * 0.9 在二进制浮点里是
# 142.20000000000002，不是 142.2。用它逼出金额 round 的问题，299 测不出来。
DEFAULT_INITIAL_STATE: dict[str, Any] = {
    "users": {
        "U1001": {"name": "张三", "vip": True},
        "U1002": {"name": "李四", "vip": False},
    },
    "orders": {
        "A123": {"user": "U1001", "amount": 299.0, "status": "shipped"},
        "B456": {"user": "U1002", "amount": 158.0, "status": "shipped"},
    },
    "refunds": [],
    # 时钟也是状态的一部分，不是 datetime.now()。
    # 理由：now() 每次不同 → 同一任务跑两次 refunds 里 ts 就不同 → D13 做两次实验对比时
    # 会报"有差异"，你一看只是时间戳，然后每加一个带时间的字段就要改一次排除规则。
    # mock 环境里，除了模型输出，其他一切都要确定。
    "clock": "2026-01-01T00:00:00Z",
    "policy": {
        "shipped_requires_cancel": True,
        "vip_full_refund": True,
        "non_vip_refund_ratio": 0.9,
    },
}


class EnvState:
    """一个任务的环境状态。构造即就绪，reset() 回到构造时的样子。"""

    def __init__(self, initial: dict | None = None) -> None:
        source = DEFAULT_INITIAL_STATE if initial is None else initial
        # 模板深拷贝两份：一份留作 reset 的基准，一份给工具去改。
        # 不拷贝的话，工具改了 _data 就等于改了 DEFAULT_INITIAL_STATE，
        # 下一个任务直接继承上一个任务的脏状态。
        self._initial: dict[str, Any] = copy.deepcopy(source)
        self._data: dict[str, Any] = copy.deepcopy(self._initial)

    @property
    def data(self) -> dict:
        """当前状态。工具直接改这个 dict（不返回拷贝，拷贝会让修改失效）。"""
        return self._data

    def reset(self) -> None:
        """回到初始状态。deepcopy 保证与上一次任务的修改彻底无关。"""
        self._data = copy.deepcopy(self._initial)

    def snapshot(self) -> dict:
        """当前状态的深拷贝，可用于事后对比或 restore。"""
        return copy.deepcopy(self._data)

    def restore(self, snapshot: dict) -> None:
        """用快照覆盖当前状态。入参也深拷贝，避免外部持有的快照被工具改坏。"""
        self._data = copy.deepcopy(snapshot)

    def get_order(self, order_id: str) -> dict | None:
        """只读查询，查不到返回 None。要不要报错由 tools.py 决定。"""
        return self._data["orders"].get(order_id)

    def get_user(self, user_id: str) -> dict | None:
        """只读查询，查不到返回 None。"""
        return self._data["users"].get(user_id)
