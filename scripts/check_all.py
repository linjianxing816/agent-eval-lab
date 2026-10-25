"""把人工查出来的坏题变成自动检查：跑三条静态检查，不调模型。

三条检查对应 `evalkit/taskgen/variant.py`：

| # | 函数 | 级别 | 抓什么 |
|---|---|---|---|
| 1 | `check_multi_refund_amounts` | 单任务 | 多单退款时，断言金额和各自 VIP 身份对不上 |
| 2 | `check_batch_consistency` | 批次 | 同一金额在不同任务里期望不一致 |
| 3 | `check_verb_match` | 单任务 | 指令只有"核实"类动词，断言却要求"变成 refunded" |

**只跑这三条**，不跑 `check_structure` 里那些类别级规则——
人工集的类别结构和自动集不同，套上去会一片误报。

用法：
    python scripts/check_all.py                      # 检查两个任务文件
    python scripts/check_all.py tasks/xxx.jsonl      # 检查指定文件（回归验证用）
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evalkit.taskgen.variant import (  # noqa: E402
    check_batch_consistency,
    check_multi_refund_amounts,
    check_verb_match,
)

DEFAULT_FILES = ("tasks/gen_s2.jsonl", "tasks/human40.jsonl")
CHECK_NAMES = {
    "amount": "检查 1 多单金额一致性",
    "batch": "检查 2 同金额跨任务一致",
    "verb": "检查 3 指令动词匹配",
}


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run(path: Path, quiet: bool = False) -> dict[str, list]:
    """对一个任务文件跑三条检查，返回 {检查名: [命中详情]}。"""
    tasks = load(path)
    hits: dict[str, list] = {"amount": [], "batch": [], "verb": []}

    for task in tasks:
        tid = task.get("id", "?")
        for key, fn in (("amount", check_multi_refund_amounts), ("verb", check_verb_match)):
            reason = fn(task)
            if reason:
                hits[key].append({"task_id": tid, "reason": reason,
                                  "instruction": str(task.get("instruction") or "")[:60]})

    for conflict in check_batch_consistency(tasks):
        hits["batch"].append({"task_id": "（批次级）", "reason": conflict["message"],
                              "instruction": ""})

    if not quiet:
        print("\n" + "=" * 74)
        print("{0}（{1} 条）".format(path, len(tasks)))
        print("=" * 74)
        for key, name in CHECK_NAMES.items():
            items = hits[key]
            mark = "✓ 0 命中" if not items else "✗ {0} 命中".format(len(items))
            print("  {0:<24} {1}".format(name, mark))
            for item in items:
                print("      [{0}] {1}".format(item["task_id"], item["reason"]))
                if item["instruction"]:
                    print("              {0}".format(item["instruction"]))
    return hits


def main(argv: list[str]) -> int:
    files = [Path(a) for a in argv[1:]] or [Path(f) for f in DEFAULT_FILES]
    total = 0
    for path in files:
        if not path.exists():
            print("找不到", path)
            return 2
        hits = run(path)
        total += sum(len(v) for v in hits.values())

    print("\n" + "-" * 74)
    print("合计命中：{0}".format(total))
    return 0 if total == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
