"""幻觉专项分析：把 d26 里 category == hallucinated_result 的执行全部拉出来，逐条还原。

**只读**。不改判定逻辑、不改任务集、不调模型。判定 detail 字段 jsonl 里没存，
所以这里用 `judge_task()` 拿**已存下来的 final_state + trajectory + final_content** 重判一遍
（纯函数，同样的输入必给同样的输出，0 成本）。脚本开头会先验一遍
「180 条重判结果与 jsonl 里存的 category 逐条一致」，验不过就直接退出——
重判不可信的话，后面所有结论都不能用。

用法：
    python -m scripts.analyze_hallucination          # 打全部分析
    python -m scripts.analyze_hallucination --json   # 顺手落 results/hallucination_analysis.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evalkit.agent.loop import LoopResult          # noqa: E402
from evalkit.audit import load_tasks              # noqa: E402
from evalkit.env.sandbox import TrajectoryStep    # noqa: E402
from evalkit.judge import judge_task              # noqa: E402
from evalkit.judge.trajectory import CLAIM_WORDS  # noqa: E402

# 定稿批次用的是 gen_s2.jsonl（100 条生成任务），不是 gen_s1.jsonl。
# 两个文件的 id 相同但内容不同（gen_s1 是另一轮生成），拿错文件会让「期望值」全错。
TASKS = ROOT / "tasks" / "gen_s2.jsonl"
BATCHES = (
    ("A", ROOT / "results" / "d26_modela.jsonl"),
    ("B", ROOT / "results" / "d26_modelb.jsonl"),
)
TARGET = "hallucinated_result"
INFRA = ("request_error", "rate_limited")   # 基建失败不算模型的错


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def rebuild(row: dict) -> tuple[list[TrajectoryStep], LoopResult]:
    """把 jsonl 里的轨迹还原成判定层要的对象。字段名与 TrajectoryStep 一一对应。"""
    steps = [
        TrajectoryStep(
            step=s["step"], tool=s["tool"], args=s["args"], result=s["result"],
            elapsed_ms=s["elapsed_ms"], error=s["error"],
            repeated=s["repeated"], malformed=s["malformed"],
        )
        for s in row["trajectory"]
    ]
    result = LoopResult(
        steps=row["steps"], exit_reason="no_tool_calls", final_content=row["final_content"],
        tool_calls_total=len(steps), messages=[], per_step_tokens=[],
        per_step_completion_tokens=[], per_step_ms=[],
    )
    return steps, result


def rejudge(row: dict, task: dict) -> dict:
    steps, result = rebuild(row)
    return judge_task(task, row["final_state"], result, steps)


def refund_amounts(row: dict) -> list:
    return [s["args"].get("amount") for s in row["trajectory"] if s["tool"] == "submit_refund"]


def tools_of(row: dict) -> list[str]:
    return [s["tool"] for s in row["trajectory"]]


def has_error_result(row: dict) -> list[str]:
    """返回所有「工具返回里有 error 键」的步骤描述。用来证明有没有编造实体。"""
    out = []
    for s in row["trajectory"]:
        if "error" in (s["result"] or {}):
            out.append("第 {0} 步 {1}".format(s["step"], s["tool"]))
    return out


def classify(row: dict, task: dict) -> tuple[str, str]:
    """按「幻觉发生在哪一层」分类。返回 (层, 子形态)。

    判定顺序：先看实体（有没有引用不存在的东西）→ 再看数值（金额有没有工具依据）
    → 再看工具返回（说的和返回的对不对得上）→ 都不成立就是「状态层」。
    """
    # L1 实体层：引用了环境里不存在的 order_id
    if has_error_result(row):
        return "L1", "编造订单号/实体（工具返回过 error 仍继续）"
    # L2 数值层：提交了金额但没调 calc_refund——退款金额的唯一权威来源被跳过
    if "submit_refund" in tools_of(row) and "calc_refund" not in tools_of(row):
        n = len(refund_amounts(row))
        return "L2", "单笔跳过 calc_refund" if n <= 1 else "多笔全部跳过 calc_refund"
    # L3 工具层：content 说的事和工具返回对不上。逐条比对太脆，这里用「声称退款成功
    # 但 refunds 为空」这一个硬判据——真正的编造工具返回一定会在这里露出来。
    claims_refund = "退款" in (row["final_content"] or "") and (
        "成功" in (row["final_content"] or "") or "已退" in (row["final_content"] or "")
    )
    if claims_refund and not row["final_state"].get("refunds"):
        return "L3", "声称已退款但环境里没有退款记录"
    # L4 状态层：订单初始已是 cancelled，仍执行 submit_refund（重复退款）。
    # 注意：这不是「题错」。工具只执行不裁决，状态机允许 cancelled→refunded，
    # 但业务上「已取消的订单再退一次」是重复退款，判定层判失败是对的。
    return "L4", "重复退款：订单已取消仍执行退款（模型没把状态当决策依据）"


def claim_hit(content: str) -> str:
    text = content or ""
    return next((w for w in CLAIM_WORDS if w in text), "")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="落一份 results/hallucination_analysis.json")
    args = ap.parse_args(argv)

    tasks = {t["id"]: t for t in load_tasks(TASKS)}
    data = {name: load(path) for name, path in BATCHES}

    # 拿错任务文件最快的信号：轨迹里的 task_id 在任务集里查不到。
    # gen_s1 / gen_s2 / gen_s1_old_23bad 的 id 完全相同、内容不同，查不到才说明拿错了。
    missing = sorted({r["task_id"] for rows in data.values() for r in rows} - set(tasks))
    if missing:
        print("任务集 {0} 里找不到：{1}".format(TASKS.name, missing))
        print("八成是拿错文件了（gen_s1 / gen_s2 的 id 相同、内容不同），期望值会全错。")
        return 1

    # ---- 0. 先验重判可信：180 条逐条对齐 ----
    # 基建失败（request_error / rate_limited）不归判定层管——它是跑批层在 LLM 调用
    # 失败时写进去的，judge_task 永远产不出这个值。这 4 条跳过比对，不是放水。
    total = mismatched = infra = 0
    for name, rows in data.items():
        for row in rows:
            total += 1
            if row["category"] in INFRA:
                infra += 1
                continue
            got = rejudge(row, tasks[row["task_id"]])
            if got["category"] != row["category"]:
                mismatched += 1
                print("  ✗ {0} {1} run{2}: 存 {3} / 重判 {4}".format(
                    name, row["task_id"], row["run"], row["category"], got["category"]))
    print("【0】重判校验：{0} 条，跳过基建失败 {1} 条，比对 {2} 条，不一致 {3} 条".format(
        total, infra, total - infra, mismatched))
    if mismatched:
        print("重判不可信，后面的结论不能用。先查 TrajectoryStep 字段是否与 jsonl 对齐。")
        return 1

    # ---- 1. 九条逐条 ----
    hits = []
    for name, rows in data.items():
        for row in rows:
            if row.get("category") == TARGET:
                hits.append((name, row, tasks[row["task_id"]], rejudge(row, tasks[row["task_id"]])))

    print("\n【1】{0} 共 {1} 条".format(TARGET, len(hits)))
    for name, row, task, verdict in hits:
        print("\n" + "-" * 96)
        print("{0} | {1} run{2} | {3} 步 | 场景 {4}".format(
            name, row["task_id"], row["run"], row["steps"], task["scenario"]))
        print("指令：{0}".format(task["instruction"]))
        print("期望：{0}".format([(c["path"], c["expect"]) for c in task["success_criteria"]]))
        print("轨迹：")
        for s in row["trajectory"]:
            print("  [{0}] {1}({2}) -> {3}".format(
                s["step"], s["tool"], json.dumps(s["args"], ensure_ascii=False),
                json.dumps(s["result"], ensure_ascii=False)))
        print("最终：{0}".format(json.dumps(row["final_state"].get("orders"), ensure_ascii=False)))
        print("退款：{0}".format(json.dumps(row["final_state"].get("refunds"), ensure_ascii=False)))
        print("原文：{0}".format(row["final_content"]))
        print("判定：{0}（命中词「{1}」）| {2}".format(
            verdict["category"], claim_hit(row["final_content"]), verdict["detail"]))

    # ---- 2. 分层统计 ----
    layers: dict[str, list] = defaultdict(list)
    for name, row, task, _ in hits:
        layer, shape = classify(row, task)
        layers[layer].append((name, row["task_id"], row["run"], shape))
    print("\n【2】按层分类")
    for layer in sorted(layers):
        print("  {0}：{1} 次".format(layer, len(layers[layer])))
        for item in layers[layer]:
            print("     {0} {1} run{2} — {3}".format(*item))
    for layer in ("L1", "L3"):
        if layer not in layers:
            print("  {0}：0 次（空类，证据见文档）".format(layer))

    # ---- 3. 系统性 vs 偶发：每个涉及任务的全部 run 并排 ----
    print("\n【3】系统性 vs 偶发：同任务其余 run 对照")
    for tid in sorted({row["task_id"] for _, row, _, _ in hits}):
        print("\n  {0} | {1} | 期望 {2}".format(
            tid, tasks[tid]["instruction"],
            [c["expect"] for c in tasks[tid]["success_criteria"]]))
        for name, rows in data.items():
            for row in [r for r in rows if r["task_id"] == tid]:
                print("    {0} run{1}: {2:<22} 步{3} 算={4} 金额={5} 命中词=「{6}」".format(
                    name, row["run"], row["category"] or "PASS", row["steps"],
                    "Y" if "calc_refund" in tools_of(row) else "N",
                    refund_amounts(row), claim_hit(row["final_content"])))

    # ---- 4. calc_refund 调用率：底层缺陷有多大 ----
    print("\n【4】退款有没有先算（全量）")
    for name, rows in data.items():
        valid = [r for r in rows if r["category"] not in INFRA]
        refunded = [r for r in valid if "submit_refund" in tools_of(r)]
        skipped = [r for r in refunded if "calc_refund" not in tools_of(r)]
        print("  {0}: 有效 {1} 次 | 退款 {2} 次 | 其中没算 {3} 次（{4:.1%}）".format(
            name, len(valid), len(refunded), len(skipped),
            len(skipped) / len(refunded) if refunded else 0))
        print("     没算的那些落到：{0}".format(
            dict(Counter(r["category"] or "PASS" for r in skipped))))

    # ---- 5. 状态陷阱：034/041 把结论拉偏多少 ----
    print("\n【5】状态陷阱 类别：含 / 不含 gen-s1-034、gen-s1-041")
    tainted = {"gen-s1-034", "gen-s1-041"}
    for name, rows in data.items():
        st = [r for r in rows if tasks[r["task_id"]]["scenario"] == "状态陷阱"
              and r["category"] not in INFRA]
        clean = [r for r in st if r["task_id"] not in tainted]
        fmt = lambda rs: "{0}/{1} = {2:.4f}".format(  # noqa: E731
            sum(1 for r in rs if r["passed"]), len(rs),
            sum(1 for r in rs if r["passed"]) / len(rs) if rs else 0)
        print("  {0}: 含 {1} | 不含 {2}".format(name, fmt(st), fmt(clean)))

    if args.json:
        out = {
            "rejudge": {"total": total, "mismatched": mismatched},
            "count": len(hits),
            "items": [
                {"model": name, "task_id": row["task_id"], "run": row["run"],
                 "steps": row["steps"], "tools": tools_of(row),
                 "amounts": refund_amounts(row), "content": row["final_content"],
                 "layer": classify(row, task)[0], "shape": classify(row, task)[1],
                 "judge_detail": verdict["detail"], "claim_word": claim_hit(row["final_content"])}
                for name, row, task, verdict in hits
            ],
            "layers": {k: len(v) for k, v in layers.items()},
        }
        path = ROOT / "results" / "hallucination_analysis.json"
        path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print("\n已写出 {0}".format(path.relative_to(ROOT)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
