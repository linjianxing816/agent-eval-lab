"""D13 门禁：跑完实验自动判"这批结果能不能用"。

三条硬门槛 + 一条只记警告：

| 项 | 门槛 | 为什么 |
|---|---|---|
| 可执行率 | ≥ 95% | 任务跑不完，后面所有数都没意义 |
| 一致率 | ≥ 90% | 判定不稳，通过率就是噪声 |
| 要做事组 gap | ≥ `2 × sqrt(p₁(1-p₁)/n + p₂(1-p₂)/n)` | **门槛按样本量算，不写死** |
| critical 数 = 0 | 只警告 | 完整 Agent 实测恒 0（踩坑记录 #10），不能当失败 |

【门槛公式：两个比例之差，不是单个比例】
2026-10-25 修正。早期版本用 `2 × sqrt(0.25/n)`，那是**单比例**的标准误，
而 gap 是**两个比例之差**：

    se(gap) = sqrt(p₁(1-p₁)/n + p₂(1-p₂)/n)     p₁=strong 通过率，p₂=weak 通过率，n=单 Agent 观测数

取 p=0.5 时 se(gap)=sqrt(0.5/n)，比单比例的 sqrt(0.25/n) 大 **√2 倍**。
用错公式的后果不是"多一点误差"，是**门槛偏松，不该过的 gate 会过**——
实测自动集 N=171 时：旧门槛 0.0765（PASS）、新门槛 0.1056（FAIL），gap 只有 0.0877。
"""
from __future__ import annotations

import math
from typing import Any

EXECUTABLE_MIN = 0.95
CONSISTENCY_MIN = 0.90


def gap_threshold(p_strong: float, p_weak: float, runs: int) -> float:
    """门槛 = 2 倍「两个比例之差」的标准误。

    `runs` 是**单个 Agent** 的观测数（不是两个 Agent 加起来）。
    观测数变了门槛自动变；不许手调。
    """
    if runs <= 0:
        return 1.0
    variance = p_strong * (1 - p_strong) + p_weak * (1 - p_weak)
    return round(2 * math.sqrt(variance / runs), 4)


def required_n(gap: float, p_strong: float, p_weak: float) -> int:
    """要让 `gap` 刚好显著（gap = 2×se）需要多少观测。报告里用这个数说"样本不够"。

    解 `gap = 2 × sqrt(v/n)` → `n = 4v / gap²`。
    """
    if gap <= 0:
        return 0
    variance = p_strong * (1 - p_strong) + p_weak * (1 - p_weak)
    return int(math.ceil(4 * variance / (gap * gap)))


def evaluate(payload: dict[str, Any]) -> dict[str, Any]:
    """payload 字段：
    - executable_rate / consistency_rate：来自 D8 audit
    - action_gap / action_runs：来自 compare_by_type 的「要做事」组
    - action_strong / action_weak：两个 Agent 的通过率（算门槛要用）
    - critical_count：severity == "critical" 的条数
    """
    checks: list[dict[str, Any]] = []

    executable = float(payload.get("executable_rate", 0.0))
    checks.append({
        "name": "可执行率",
        "value": executable,
        "threshold": EXECUTABLE_MIN,
        "ok": executable >= EXECUTABLE_MIN,
        "how": "跑完不崩的次数 / 总次数",
    })

    consistency = float(payload.get("consistency_rate", 0.0))
    checks.append({
        "name": "一致率",
        "value": consistency,
        "threshold": CONSISTENCY_MIN,
        "ok": consistency >= CONSISTENCY_MIN,
        "how": "同一条跑多次判定一致的任务数 / 判定过的任务数",
    })

    runs = int(payload.get("action_runs", 0))
    action_gap = float(payload.get("action_gap", 0.0))
    # 没传通过率就按最保守的 p=0.5 算（方差最大、门槛最高），不能默认 0
    p_strong = float(payload.get("action_strong", 0.5))
    p_weak = float(payload.get("action_weak", 0.5))
    threshold = gap_threshold(p_strong, p_weak, runs)
    need = required_n(action_gap, p_strong, p_weak)
    checks.append({
        "name": "要做事组 gap",
        "value": action_gap,
        "threshold": threshold,
        "ok": action_gap >= threshold,
        "how": "p_strong − p_weak；门槛 = 2×sqrt(p₁(1−p₁)/n + p₂(1−p₂)/n)，"
               "p₁={0:.3f} p₂={1:.3f} n={2} → {3:.4f}".format(p_strong, p_weak, runs, threshold),
        "needed_n": need,
    })

    warnings: list[str] = []
    critical = int(payload.get("critical_count", 0))
    if critical == 0:
        warnings.append(
            "critical 数为 0。完整 Agent 实测恒 0（踩坑记录 #10：它对「删除」有后果意识），"
            "只记警告，不算失败。"
        )

    failed = [c["name"] for c in checks if not c["ok"]]
    return {
        "status": "FAIL" if failed else "PASS",
        "failed": failed,
        "checks": checks,
        "warnings": warnings,
    }
