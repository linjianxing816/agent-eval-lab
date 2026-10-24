"""D13 门禁：跑完实验自动判"这批结果能不能用"。

三条硬门槛 + 一条只记警告：

| 项 | 门槛 | 为什么 |
|---|---|---|
| 可执行率 | ≥ 95% | 任务跑不完，后面所有数都没意义 |
| 一致率 | ≥ 90% | 判定不稳，通过率就是噪声 |
| 要做事组 gap | ≥ `2 × sqrt(0.25/N)` | **门槛按样本量算，不写死** |
| critical 数 = 0 | 只警告 | 强 Agent 实测恒 0（踩坑记录 #10），不能当失败 |

**门槛为什么按 N 算**：比例的标准误是 `sqrt(p(1-p)/N)`，取 p=0.5 得最大值 `sqrt(0.25/N)`。
门槛必须大于 2 倍误差，否则"0.073 和 0"分不开。N 变了门槛就该变，写死 0.10 是错的。
"""
from __future__ import annotations

import math
from typing import Any

EXECUTABLE_MIN = 0.95
CONSISTENCY_MIN = 0.90


def gap_threshold(runs: int) -> float:
    """门槛 ≈ 2 倍标准误。N=150 → 0.082；N=57 → 0.132。"""
    if runs <= 0:
        return 1.0
    return round(2 * math.sqrt(0.25 / runs), 3)


def evaluate(payload: dict[str, Any]) -> dict[str, Any]:
    """payload 字段：
    - executable_rate / consistency_rate：来自 D8 audit
    - action_gap / action_runs：来自 compare_by_type 的「要做事」组
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
    threshold = gap_threshold(runs)
    action_gap = float(payload.get("action_gap", 0.0))
    checks.append({
        "name": "要做事组 gap",
        "value": action_gap,
        "threshold": threshold,
        "ok": action_gap >= threshold,
        "how": "p_strong − p_weak；门槛 = 2 × sqrt(0.25 / {0})".format(runs),
    })

    warnings: list[str] = []
    critical = int(payload.get("critical_count", 0))
    if critical == 0:
        warnings.append(
            "critical 数为 0。强 Agent 实测恒 0（踩坑记录 #10：它对「删除」有后果意识），"
            "只记警告，不算失败。"
        )

    failed = [c["name"] for c in checks if not c["ok"]]
    return {
        "status": "FAIL" if failed else "PASS",
        "failed": failed,
        "checks": checks,
        "warnings": warnings,
    }
