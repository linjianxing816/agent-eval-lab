"""修坏题前后的对照：同一套指标、同一批任务，只有任务定义变了。

修前 = `modela_auto_full_90.jsonl` / `modelb_auto_full.jsonl`
修后 = `d24_modela.jsonl` / `d24_modelb.jsonl`

口径和 `analyze_layered.py` 一致（剔除 request_error、红线从轨迹数），
所以两边的数可以直接比。**不改门槛公式、不调样本量。**

用法：python scripts/compare_fix.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_layered import metrics, separability  # noqa: E402
from evalkit.gate import gap_threshold_two  # noqa: E402

RESULTS = Path("results")
TASKS = Path("tasks/gen_s2.jsonl")
SCENARIO = "多步依赖"

BEFORE = {"A": RESULTS / "modela_auto_full_90.jsonl", "B": RESULTS / "modelb_auto_full.jsonl"}
AFTER = {"A": RESULTS / "d24_modela.jsonl", "B": RESULTS / "d24_modelb.jsonl"}


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def scenario_rate(rows: list[dict], tasks: dict[str, dict], scenario: str) -> tuple[float | None, int]:
    ids = {tid for tid, t in tasks.items() if t.get("scenario") == scenario}
    mine = [r for r in rows if r["task_id"] in ids and r["category"] not in ("request_error", "rate_limited")]
    if not mine:
        return None, 0
    return round(sum(1 for r in mine if r["passed"]) / len(mine), 4), len(mine)


def main() -> int:
    tasks = {t["id"]: t for t in load(TASKS)}
    if not AFTER["A"].exists() or not AFTER["B"].exists():
        print("修后数据还没齐：", AFTER["A"].exists(), AFTER["B"].exists())
        return 2

    stats = {}
    for tag, paths in (("before", BEFORE), ("after", AFTER)):
        stats[tag] = {}
        for label in ("A", "B"):
            rows = load(paths[label])
            m = metrics(rows, tasks)
            m["multi"], m["multi_n"] = scenario_rate(rows, tasks, SCENARIO)
            m["rows"] = rows
            stats[tag][label] = m

    print("{0:<26} {1:>10} {2:>10} {3:>10} {4:>10}".format(
        "指标", "修前 A", "修前 B", "修后 A", "修后 B"))
    print("-" * 70)
    for key, name, fmt in (
        ("pass_rate", "通过率", "{0:.3f}"),
        ("red_rate", "红线违规率", "{0:.1%}"),
        ("multi", "多步依赖通过率", "{0:.0%}"),
        ("halluc_rate", "幻觉率", "{0:.1%}"),
        ("malformed_rate", "格式错误率", "{0:.1%}"),
    ):
        cells = []
        for tag in ("before", "after"):
            for label in ("A", "B"):
                v = stats[tag][label][key]
                cells.append("—" if v is None else fmt.format(v))
        print("{0:<26} {1:>10} {2:>10} {3:>10} {4:>10}".format(name, *cells))

    print("\n执行数（已剔 request_error）：")
    for tag in ("before", "after"):
        print("  {0}: A {1}（剔 {2}）| B {3}（剔 {4}）".format(
            "修前" if tag == "before" else "修后",
            stats[tag]["A"]["runs"], stats[tag]["A"]["dropped"],
            stats[tag]["B"]["runs"], stats[tag]["B"]["dropped"]))

    for tag in ("before", "after"):
        a, b = stats[tag]["A"], stats[tag]["B"]
        gap = round(a["pass_rate"] - b["pass_rate"], 4)
        threshold = gap_threshold_two(a["pass_rate"], a["runs"], b["pass_rate"], b["runs"])
        sep = separability(a["rows"], b["rows"])
        print("\n{0}：gap {1:+.4f} | 门槛 {2:.4f} → {3}".format(
            "修前" if tag == "before" else "修后", gap, threshold,
            "能区分" if abs(gap) >= threshold else "分不出"))
        print("   逐任务：严格完全分离 {0}/{1}、宽松 {2}/{1}、完全相同 {3}".format(
            sep["separable_strict"], sep["total"], sep["separable"], sep["same"]))

    # 6 类交叉表（只打修后）
    print("\n修后 6 类 × 2 模型：")
    for scenario in ("诱导越权", "信息不足", "状态陷阱", "多步依赖", "边界值", "正常"):
        cells = []
        for label in ("A", "B"):
            rate, n = scenario_rate(stats["after"][label]["rows"], tasks, scenario)
            cells.append("—" if rate is None else "{0:.1%}({1})".format(rate, n))
        print("  {0:<8} A {1:>10}   B {2:>10}".format(scenario, *cells))

    (RESULTS / "compare_fix.json").write_text(json.dumps({
        k: {label: {kk: vv for kk, vv in m.items() if kk != "rows"} for label, m in v.items()}
        for k, v in stats.items()
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n写入 results/compare_fix.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
