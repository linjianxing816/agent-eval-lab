"""把 results/ 里的原始日志装配成 D13 的 diff 和 D14 的报告页。

一条命令：
    python -m evalkit.build_report
产出：
    results/diff.json   人工集 vs 自动集
    results/diff.md     同上，人读版
    results/report.html 双击可打开（数据全内联，无 fetch）
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from evalkit.audit import load_tasks, sample_tasks
from evalkit.diff import diff_sets, to_markdown
from evalkit.discriminate import (
    ACTION_GROUPS,
    UNCHANGED_GROUPS,
    Outcome,
    compare_by_type,
    group_of,
    normalize_agent,
    summarize_agent,
    token_mean,
    type_of,
)
from evalkit.gate import evaluate, gap_threshold_two, required_n
from evalkit.report import write_report

RESULTS = Path("results")
COLOR_ACTION = "#1f5fbf"
COLOR_UNCHANGED = "#c77700"


def load_if_exists(path: Path) -> list[Outcome]:
    return load_outcomes(path) if path.exists() else []


def merge_override(base: list[Outcome], override: list[Outcome]) -> list[Outcome]:
    """用新数据**替换**同 task_id 的旧记录。

    改了 instruction 就必须重跑那条（judge 不读 instruction，光重判没用），
    重跑后旧记录作废——两份都留着会把同一任务算两遍。
    """
    touched = {o.task_id for o in override}
    return [o for o in base if o.task_id not in touched] + override


def load_outcomes(path: Path) -> list[Outcome]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        rows.append(Outcome(
            # 旧日志里写的是 strong / weak，统一映射成 full / no_order_info
            task_id=raw["task_id"], agent=normalize_agent(raw["agent"]), run=raw["run"], ok=raw["ok"],
            passed=raw["passed"], category=raw["category"], steps=raw["steps"],
            prompt_tokens=raw["prompt_tokens"], completion_tokens=raw["completion_tokens"],
            trajectory=tuple(raw.get("trajectory") or ()),
            final_state=raw.get("final_state"),
            # 旧日志没有这个字段 → 空字符串，不报错
            final_content=raw.get("final_content") or "",
        ))
    return rows


def build_diff() -> dict[str, Any]:
    human = load_tasks("tasks/human40.jsonl")
    auto = sample_tasks(load_tasks("tasks/gen_s2.jsonl"), seed=2024)
    sets = {
        "人工集(42)": {"tasks": human,
                       "full": load_outcomes(RESULTS / "d13b_human_strong.jsonl"),
                       "no_order_info": load_outcomes(RESULTS / "d13b_human_weak.jsonl")},
        "自动集(30 抽样)": {"tasks": auto,
                            "full": load_outcomes(RESULTS / "d13_auto_strong.jsonl"),
                            "no_order_info": load_outcomes(RESULTS / "d13_auto_weak.jsonl")},
    }
    report = diff_sets(sets)
    (RESULTS / "diff.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (RESULTS / "diff.md").write_text(to_markdown(report), encoding="utf-8")
    return report


def _badge(outcome: Outcome) -> str:
    if not outcome.ok:
        return "跑崩"
    if outcome.passed:
        return "通过"
    return "严重失败" if outcome.category in ("forbidden_tool", "premature_finish",
                                              "hallucinated_result", "tool_call_malformed") else "失败"


def build_trajectories() -> list[dict[str, Any]]:
    """轨迹查看器的数据。

    要做事组用 D15+D16（task-035 用 D16 覆盖旧的），不该动组用 D13。
    **不能按 (task_id, agent, run) 去重**：不同批次 run 号都从 0 开始，
    去重会把新记录当成旧的跳过——实测 task-035 就显示成了改动前的失败轨迹。
    """
    tasks = {t["id"]: t for t in load_tasks("tasks/human40.jsonl")}
    tasks.update({t["id"]: t for t in load_tasks("tasks/gen_s2.jsonl")})

    def pick_from(path: Path, bucket: str) -> list[Outcome]:
        rows = load_if_exists(path)
        groups = ACTION_GROUPS if bucket == "要做事" else UNCHANGED_GROUPS
        return [o for o in rows if group_of(tasks.get(o.task_id, {})) in groups]

    human = {
        "full": merge_override(pick_from(RESULTS / "d15_human_strong.jsonl", "要做事"),
                                 load_if_exists(RESULTS / "d16_human_strong.jsonl"))
                  + pick_from(RESULTS / "d13b_human_strong.jsonl", "不该动"),
        "no_order_info": merge_override(pick_from(RESULTS / "d15_human_weak.jsonl", "要做事"),
                               load_if_exists(RESULTS / "d16_human_weak.jsonl"))
                + pick_from(RESULTS / "d13b_human_weak.jsonl", "不该动"),
    }
    auto = {
        "full": pick_from(RESULTS / "d15_auto_strong.jsonl", "要做事")
                  + pick_from(RESULTS / "d16_auto_strong.jsonl", "要做事")
                  + pick_from(RESULTS / "d13_auto_strong.jsonl", "不该动"),
        "no_order_info": pick_from(RESULTS / "d15_auto_weak.jsonl", "要做事")
                + pick_from(RESULTS / "d16_auto_weak.jsonl", "要做事")
                + pick_from(RESULTS / "d13_auto_weak.jsonl", "不该动"),
    }

    rows: list[dict[str, Any]] = []
    for label, outcomes_by_agent in (("人工集", human), ("自动集", auto)):
        for agent in ("full", "no_order_info"):
            counter: dict[str, int] = {}
            for outcome in outcomes_by_agent[agent]:
                task = tasks.get(outcome.task_id, {})
                counter[outcome.task_id] = counter.get(outcome.task_id, 0) + 1
                forbid = set((task.get("constraints") or {}).get("forbid_tools") or [])
                steps = [
                    {
                        "step": s["step"], "tool": s["tool"],
                        "args": json.dumps(s["args"], ensure_ascii=False),
                        "result": json.dumps(s["result"], ensure_ascii=False),
                        "elapsed_ms": s["elapsed_ms"], "error": s["error"],
                        "forbidden": s["tool"] in forbid,
                    }
                    for s in outcome.trajectory
                ]
                rows.append({
                    "task_id": outcome.task_id,
                    "agent": "{0} / {1} · 第 {2} 次".format(
                        label, agent, counter[outcome.task_id]),
                    "scenario": group_of(task),
                    "instruction": task.get("instruction", ""),
                    "badge": _badge(outcome),
                    "passed": outcome.passed,
                    "steps": steps,
                    "final_state": json.dumps(outcome.final_state or {}, ensure_ascii=False),
                })
    return rows


def _task_rates(outcomes: list[Outcome]) -> dict[str, float]:
    """每条任务的通过率。同一任务跑多次就取均值——扩样本后不能再假设 repeat=1。"""
    grouped: dict[str, list[bool]] = {}
    for o in outcomes:
        grouped.setdefault(o.task_id, []).append(o.passed)
    return {tid: sum(v) / len(v) for tid, v in grouped.items()}


def build_distribution(full: list[Outcome], no_order_info: list[Outcome],
                       tasks: list[dict]) -> dict[str, Any]:
    kind = {t["id"]: type_of(t) for t in tasks}
    f_map, n_map = _task_rates(full), _task_rates(no_order_info)
    buckets = [("0.0", 0.0), ("0.25", 0.25), ("0.5", 0.5), ("0.75", 0.75), ("1.0", 1.0)]
    counts = {b: {"要做事": 0, "不该动": 0} for b, _ in buckets}
    effective = 0
    for tid, rate in f_map.items():
        if tid not in n_map:
            continue
        diff = abs(rate - n_map[tid])
        label = min(buckets, key=lambda b: abs(b[1] - diff))[0]
        counts[label][kind.get(tid, "要做事")] += 1
        if diff > 0:
            effective += 1
    rows = []
    for label, _ in buckets:
        for group, color in (("要做事", COLOR_ACTION), ("不该动", COLOR_UNCHANGED)):
            rows.append({"label": "{0} {1}".format(label, group), "count": counts[label][group],
                         "color": color})
    return {
        "buckets": rows,
        "note": "{0}/{1} 条任务有效（区分度 > 0）。同一任务多次跑取通过率均值，"
                "再看完整与缺订单信息的差。只回答「多少条有效」，不用来看大小。".format(
                    effective, len(f_map)),
    }


def pick(outcomes: list[Outcome], tasks: list[dict], bucket: str) -> list[Outcome]:
    """按任务类型筛执行记录。**扩样本只跑了要做事组，不该动组必须从旧日志里挑**，
    不能把新旧两批混在一起算——那样同一任务会被计两次，N 和通过率都错。"""
    groups = ACTION_GROUPS if bucket == "要做事" else UNCHANGED_GROUPS
    ids = {t["id"] for t in tasks if group_of(t) in groups}
    return [o for o in outcomes if o.task_id in ids]


def build_instruction_following(tasks: list[dict], full_rows: list[Outcome],
                                no_rows: list[Outcome]) -> dict[str, Any]:
    """027 / 028 单独统计「指令遵循」。

    这两条测的是"会不会按用户给的金额提交"，**不是业务正确**：
    按规则退 299 是业务正确的，按指令退 100 才是指令正确。
    混进"业务正确"读者会误读成"模型算错了"。
    """
    wanted = {"task-027", "task-028"}
    by_id = {t["id"]: t for t in tasks if t["id"] in wanted}
    rows: list[dict[str, Any]] = []
    for tid in sorted(by_id):
        task = by_id[tid]
        expect = next((c["expect"] for c in task["success_criteria"]
                       if str(c["path"]).endswith(".amount")), None)
        for label, outcomes in (("full", full_rows), ("no_order_info", no_rows)):
            mine = [o for o in outcomes if o.task_id == tid]
            if not mine:
                continue
            amounts = []
            for o in mine:
                refunds = (o.final_state or {}).get("refunds") or []
                amounts.append(round(float(refunds[-1]["amount"]), 2) if refunds else 0.0)
            ok = all(abs(a - float(expect)) < 0.005 for a in amounts)
            rows.append({
                "task_id": tid, "agent": label, "instruction": task["instruction"],
                "expect": float(expect),
                "actual": " / ".join("{0:.2f}".format(a) for a in amounts),
                "ok": ok,
            })
    return {
        "rows": rows,
        "note": "这一栏是「指令遵循」，不是「业务正确」：模型按规则自己算 299 属于业务正确、"
                "但违反指令。两条任务的期望金额是用户明确给的 100 / 150。",
    }


def build_strength(stages: list[tuple[str, list[Outcome], list[Outcome], list[dict]]],
                   current_gap: float, current_p1: float, current_p2: float) -> dict[str, Any]:
    """「样本量与结论强度」：同一指标在不同样本量下的误差带。

    误差带 = 2 × se(gap)，se 用两比例之差的公式。
    **误差带比 gap 还宽时，那个数没有意义**——不是效果小，是测不出来。
    """
    rows: list[dict[str, Any]] = []
    for label, full_rows, no_rows, tasks in stages:
        report = compare_by_type(full_rows, no_rows, tasks)["要做事"]
        n = report["runs"]
        if n <= 0:
            continue
        variance = report["full"] * (1 - report["full"]) + report["no_order_info"] * (1 - report["no_order_info"])
        se = (variance / n) ** 0.5
        band = 2 * se
        gap = report["gap"]
        # 分档按「误差带是 gap 的几倍」：
        #   ≥1.5 倍 → 真实值可能只有观测值的三分之一，连排序都可能反过来 → 没有意义
        #   ≥1 倍   → 区间含 0，不显著，但符号稳定 → 能看出方向
        #   <1 倍   → 可以判定
        if abs(gap) == 0:
            verdict = "两组完全一致，无差异可谈"
        elif band >= 1.5 * abs(gap):
            verdict = "误差带是 gap 的 {0:.1f} 倍 → 这个数没有意义".format(band / abs(gap))
        elif band >= abs(gap):
            verdict = "区间含 0，不显著；但方向稳定 → 能看出方向"
        else:
            verdict = "gap 大于误差带 → 可以判定"
        rows.append({
            "label": label, "n": n, "gap": round(gap, 4),
            "se": round(se, 4), "band": round(band, 4), "verdict": verdict,
        })
    need = required_n(current_gap, current_p1, current_p2)
    return {
        "rows": rows,
        "need_n": need,
        "note": "分档规则：误差带 ≥ gap 的 1.5 倍 → 这个数没有意义；≥ gap → 能看出方向但不显著；"
                "< gap → 可以判定。「需要 N ≥ {0}」是按观测到的 p₁={1:.3f}、p₂={2:.3f} 反解出来的："
                "令 gap = 2×se，得 n = 4v/gap²（v = p₁(1−p₁)+p₂(1−p₂)）。"
                "如果按最保守的 p=0.5 算，需要 {3}。".format(
                    need, current_p1, current_p2,
                    required_n(current_gap, 0.5, 0.5)),
    }


def build_perf(action: dict[str, dict[str, list[Outcome]]]) -> dict[str, Any] | None:
    """推理性能：读 results/perf.json（D15/D16 跑出来的），加上工具耗时推算比例。

    没有 perf.json 就返回 None，报告里那一节不渲染——**不编数**。
    """
    path = RESULTS / "perf.json"
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    latency = raw["latency"]

    # 工具耗时：从已存轨迹里算（本地函数调用，应该是毫秒级）
    runs = action["自动"]["full"] + action["自动"]["no_order_info"]
    if not runs:
        return None
    tool_ms = sum(s["elapsed_ms"] for o in runs for s in o.trajectory) / len(runs)
    steps = sum(len(o.trajectory) for o in runs) / len(runs)

    ttft = latency["ttft"]["p50"]
    tpot = latency["tpot"]["p50"]
    e2e = latency["e2e"]["p50"]
    tokens = latency["output_tokens"]["p50"]
    share = latency["ttft_share_pct"]["p50"]

    # 一次任务 = 每步一次 LLM 往返；LLM 时间按实测单次往返推算
    llm_ms = steps * e2e
    ratio = llm_ms / tool_ms if tool_ms else 0.0

    # TTFT 归因：优先用**配对测量**（scripts/ttft_pair.py）——
    # 分批测的话两次之间网络会漂，漂移量比要测的效应还大（实测基线 503→750ms）。
    attrib = None
    paired_path = RESULTS / "ttft_paired.json"
    attrib_path = RESULTS / "ttft_attrib.json"
    tokens_path = RESULTS / "ttft_tokens.json"
    if paired_path.exists() and tokens_path.exists():
        paired = json.loads(paired_path.read_text(encoding="utf-8"))
        counts = json.loads(tokens_path.read_text(encoding="utf-8"))
        n_with = counts.get("with_tools_prompt_tokens") or 0
        n_without = counts.get("without_tools_prompt_tokens") or 0
        delta = paired["delta_p50"]
        attrib = {
            "paired": True,
            "with_tools_ttft": paired["with_tools_p50"],
            "with_tools_tokens": n_with,
            "without_tools_ttft": paired["without_tools_p50"],
            "without_tools_tokens": n_without,
            "delta_ttft": delta, "delta_tokens": max(n_with - n_without, 1),
            "ms_per_1k": round(delta / max(n_with - n_without, 1) * 1000, 1),
            "baseline_share": round(paired["without_tools_p50"] / paired["with_tools_p50"] * 100, 1),
            "prefill_share": round(delta / paired["with_tools_p50"] * 100, 1),
            "pairs": paired["pairs"],
            "positive_pairs": paired["delta_positive_pairs"],
            "delta_min": paired["delta_min"], "delta_max": paired["delta_max"],
            # 去掉最大离群点后的均值：一个 1059ms 的离群点能把均值从 2 拉到 108
            "delta_mean_trimmed": round(
                sum(sorted(r["delta"] for r in paired["rows"])[:-1]) / max(paired["pairs"] - 1, 1), 1),
        }
    elif attrib_path.exists() and tokens_path.exists():
        bare = json.loads(attrib_path.read_text(encoding="utf-8"))["without_tools"]
        counts = json.loads(tokens_path.read_text(encoding="utf-8"))
        bare_ttft = bare["ttft"]["p50"]
        n_with = counts.get("with_tools_prompt_tokens") or 0
        n_without = counts.get("without_tools_prompt_tokens") or 0
        delta_ttft = ttft - bare_ttft
        delta_tokens = max(n_with - n_without, 1)
        attrib = {
            "with_tools_ttft": round(ttft, 1), "with_tools_tokens": n_with,
            "without_tools_ttft": round(bare_ttft, 1), "without_tools_tokens": n_without,
            "delta_ttft": round(delta_ttft, 1), "delta_tokens": delta_tokens,
            "ms_per_1k": round(delta_ttft / delta_tokens * 1000, 1),
            "baseline_share": round(bare_ttft / ttft * 100, 1),
            "prefill_share": round(delta_ttft / ttft * 100, 1),
        }

    if share >= 50:
        if attrib:
            bottleneck = (
                "首 token 慢（占端到端 {0:.0f}%），再拆一层。<strong>配对测量</strong>（{1} 轮交替，同轮内一带一不带）："
                "带 schema {2:.0f} ms、不带 {3:.0f} ms，中位差 <strong>{4:+.0f} ms</strong>；"
                "但 {5}/{1} 轮为正、差值从 {6:+.0f} 到 {7:+.0f} ms 乱跳，"
                "去掉一个离群点后均值只有 {8:+.0f} ms。<br>"
                "→ <strong>prefill 的贡献测不出来</strong>（最多 ~{9:.0f}%，落在噪声里）。"
                "TTFT 由服务端排队主导——同一个 prompt 的 TTFT 在 {10:.0f}–{11:.0f} ms 之间跳，"
                "抖动比输入长度的影响大一个量级。<strong>缩短输入对首 token 没用。</strong>"
            ).format(share, attrib["pairs"], attrib["with_tools_ttft"], attrib["without_tools_ttft"],
                     attrib["delta_ttft"], attrib["positive_pairs"],
                     attrib["delta_min"], attrib["delta_max"], attrib["delta_mean_trimmed"],
                     attrib["prefill_share"],
                     min(r["with_tools"] for r in json.loads(paired_path.read_text(encoding="utf-8"))["rows"]),
                     max(r["with_tools"] for r in json.loads(paired_path.read_text(encoding="utf-8"))["rows"]))
        else:
            bottleneck = ("首 token 慢（占端到端 {0:.0f}%）：瓶颈在排队/网络，"
                          "不是模型解码。要提速得从连接复用、降低输入长度、错峰入手。".format(share))
    else:
        bottleneck = ("每 token 慢（首 token 只占 {0:.0f}%，其余 {1:.0f}% 是逐 token 生成）："
                      "瓶颈在解码速度。输出越长越慢，所以要提速得压输出长度"
                      "（少让它写解释、工具返回别太长）。".format(share, 100 - share))

    conclusion = (
        "一次 Agent 任务平均 {0:.1f} 步，每步一次 LLM 往返。工具是本地函数调用，"
        "单次平均 {1:.2f} ms；按实测单次往返 {2:.0f} ms 推算，"
        "LLM 时间 : 工具时间 ≈ <strong>{3:,.0f} : 1</strong>。<br>"
        "拆开看：首 token {4:.0f} ms，之后每个 token {5:.1f} ms × 约 {6:.0f} 个输出 token。"
        "<br><strong>{7}</strong>"
    ).format(steps, tool_ms, e2e, ratio, ttft, tpot, tokens, bottleneck)

    reuse = None
    reuse_path = RESULTS / "reuse_compare.json"
    if reuse_path.exists():
        reuse = json.loads(reuse_path.read_text(encoding="utf-8"))

    return {
        "latency": latency,
        "attribution": attrib,
        "reuse": reuse,
        "concurrency": raw.get("concurrency") or [],
        "steps": round(steps, 2),
        "tool_ms": round(tool_ms, 3),
        "llm_ms": round(llm_ms, 1),
        "ratio": round(ratio, 1),
        "share": round(share, 1),
        "conclusion": conclusion,
        "measured_at": raw.get("measured_at", ""),
        "note": "TTFT = 首个 content/tool_call delta 的时间（role 那种元数据 delta 不算）；"
                "TPOT = (收完 − 首 token) / (输出 token 数 − 1)；端到端 = 发请求到 [DONE]。"
                "串行 20 次、间隔 0.5s；并发扫描每档 12 个请求，429 不重试。",
    }


def build_model_compare() -> dict[str, Any] | None:
    """模型 A / B 的分层指标对比。数据来自 scripts/analyze_layered.py（纯本地重算）。

    这一块存在的理由：**同一个数据集，两个指标给出相反结论**。
    通过率说 B 更好（−0.064，但没过门槛），红线违规率说 B 明显更差（0% vs 6.7%，过门槛）。
    把两个数并排放，才能看出"通过率把做不成和越权混成一个失败"这件事。
    """
    path = RESULTS / "model_compare.json"
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    a, b = raw["A"], raw["B"]

    def row(name, pa, pb, na, nb, how) -> dict[str, Any]:
        gap = round(pa - pb, 4)
        threshold = gap_threshold_two(pa, na, pb, nb)
        return {
            "metric": name, "a": pa, "b": pb, "gap": gap,
            # gap 为 0 时门槛也是 0（方差为 0），abs(0) >= 0 会判成"能区分"——
            # 两边一模一样当然不能区分，单独挡掉
            "threshold": threshold, "ok": abs(gap) >= threshold and gap != 0,
            "na": na, "nb": nb, "how": how,
        }

    rows = [
        row("通过率", a["pass_rate"], b["pass_rate"], a["runs"], b["runs"],
            "判定通过的执行数 / 有效执行数"),
        row("红线违规率", a["red_rate"], b["red_rate"], a["runs"], b["runs"],
            "调用了被禁工具的执行数 / 有效执行数（从轨迹数，不看 category）"),
        row("幻觉率", a["halluc_rate"], b["halluc_rate"], a["runs"], b["runs"],
            "category = hallucinated_result 的执行数 / 有效执行数"),
        row("格式错误率", a["malformed_rate"], b["malformed_rate"], a["runs"], b["runs"],
            "category = tool_call_malformed 的执行数 / 有效执行数"),
    ]
    pass_row = rows[0]
    red_row = rows[1]
    sep = raw["separability"]

    return {
        "rows": rows,
        "cross": raw["cross"],
        "separability": sep,
        "A": a, "B": b,
        "need_n_pass": required_n(abs(pass_row["gap"]) or 0.01, a["pass_rate"], b["pass_rate"]),
        "need_n_red": required_n(abs(red_row["gap"]) or 0.01, a["red_rate"], b["red_rate"]),
        "note": (raw["note"] +
                 "　<strong>这是修完全部 23 条坏题之后的数据（D26）</strong>："
                 "D23 首次测（6 条坏题）→ D24 修完前 6 条 → D25 检查器又挖出 17 条 → D26 全部修完。<br>"
                 "修前（D24）通过率 A 0.856 / B 0.705（gap +0.151，门槛 0.122，能区分）、"
                 "红线 A 2.2% / B 6.8%（gap 0.046，门槛 0.062，分不出）。"),
        "conclusion": (
            "<strong>修完全部坏题后，两个指标都能区分了</strong>："
            "通过率 gap {0:+.3f}（门槛 {1:.3f}）、红线违规率 gap {2:+.3f}（门槛 {3:.3f}）。<br>"
            "<strong>A 的优势从 15.1 pp 扩大到 32.0 pp</strong>——修掉的那 16 条里，"
            "有一批正是 B 靠「不查 VIP、直接按订单金额退全额」蒙对的。<br>"
            "红线违规率也翻倍了（B {5:.1%} vs A {4:.1%}），但要注意："
            "新增的 5 次越权里有 5 次落在 <code>gen-s1-009</code>/<code>016</code> 这两条"
            "<strong>没改过的任务</strong>上，是模型随机波动，不能全记在「任务变难」头上。"
        ).format(pass_row["gap"], pass_row["threshold"], red_row["gap"], red_row["threshold"],
                 a["red_rate"], b["red_rate"]),
        "why": (
            "累计修了 <strong>23 条坏题（22 个任务）</strong>：先人工查出 6 条，"
            "补上三条自动检查后又挖出 17 条。<br>"
            "坏题的共同点：<strong>它们都在奖励「不查 VIP、直接按订单金额退」的模型</strong>。"
            "B 从不调 <code>get_user</code> 和 <code>calc_refund</code>，"
            "在期望写错的那些任务上恰好对；期望改成非 VIP 的 90% 后，B 的金额全错。<br>"
            "<strong>指标没变、模型没变，只有任务定义变对了，A 的优势就从 15 pp 变成 32 pp。</strong>"
            "这说明之前那 15 pp 是被坏题压低的——不是 A 只比 B 强一点，是 A 强得多但被拖住了。"
        ),
    }


def main() -> None:
    diff = build_diff()

    human_tasks = load_tasks("tasks/human40.jsonl")
    # 用**全量** 100 条，不能用 30 条抽样：D15 跑的是 57 条要做事组，
    # 拿抽样集去筛只会匹配到 16 条，N 直接腰斩、门槛被抬高。
    auto_tasks = load_tasks("tasks/gen_s2.jsonl")

    # D15 扩样本：只跑了要做事组（57 条 × 2 次 / 22 条 × 3 次）
    d15 = {
        "人工": {"full": load_if_exists(RESULTS / "d15_human_strong.jsonl"),
                 "no_order_info": load_if_exists(RESULTS / "d15_human_weak.jsonl")},
        "自动": {"full": load_if_exists(RESULTS / "d15_auto_strong.jsonl"),
                 "no_order_info": load_if_exists(RESULTS / "d15_auto_weak.jsonl")},
    }
    # D16：重跑 task-035（改了 instruction）+ 自动集最后一次加样本
    d16 = {
        "人工": {"full": load_if_exists(RESULTS / "d16_human_strong.jsonl"),
                 "no_order_info": load_if_exists(RESULTS / "d16_human_weak.jsonl")},
        "自动": {"full": load_if_exists(RESULTS / "d16_auto_strong.jsonl"),
                 "no_order_info": load_if_exists(RESULTS / "d16_auto_weak.jsonl")},
    }
    # 人工集：task-035 用新记录替换旧的；自动集：两批样本**相加**（不同 run，不重复）
    action = {
        "人工": {k: merge_override(d15["人工"][k], d16["人工"][k]) for k in ("full", "no_order_info")},
        "自动": {k: d15["自动"][k] + d16["自动"][k] for k in ("full", "no_order_info")},
    }
    # 不该动组没重跑，从 D13 全量日志里挑
    old = {
        "人工": {"full": load_outcomes(RESULTS / "d13b_human_strong.jsonl"),
                 "no_order_info": load_outcomes(RESULTS / "d13b_human_weak.jsonl")},
        "自动": {"full": load_outcomes(RESULTS / "d13_auto_strong.jsonl"),
                 "no_order_info": load_outcomes(RESULTS / "d13_auto_weak.jsonl")},
    }
    tasks_of = {"人工": human_tasks, "自动": auto_tasks}
    sources = {
        ("人工", "要做事"): "D15 22 条 × 3 次 + D16 重跑 task-035",
        ("人工", "不该动"): "D13 全量：20 条 × 1 次",
        ("自动", "要做事"): "D15 57 条 × 2 次 + D16 57 条 × 1 次",
        ("自动", "不该动"): "D13 抽样：14 条 × 1 次",
    }

    matrix: list[dict[str, Any]] = []
    by_type: dict[str, dict[str, Any]] = {}
    for set_name in ("人工", "自动"):
        for bucket in ("要做事", "不该动"):
            pool = action[set_name] if bucket == "要做事" else old[set_name]
            s_rows = pick(pool["full"], tasks_of[set_name], bucket)
            w_rows = pick(pool["no_order_info"], tasks_of[set_name], bucket)
            report = compare_by_type(s_rows, w_rows, tasks_of[set_name])
            row = report[bucket]
            by_type[set_name + bucket] = row
            matrix.append({
                "set": set_name, "bucket": bucket, "full": row["full"], "no_order_info": row["no_order_info"],
                "gap": row["gap"], "n": row["runs"], "threshold": row["threshold"],
                "direction": row["direction"], "source": sources[(set_name, bucket)],
            })

    action_row = by_type["自动要做事"]
    audit = json.loads((RESULTS / "audit_s2.json").read_text(encoding="utf-8"))
    critical = sum(1 for o in action["自动"]["full"]
                   if o.category in ("forbidden_tool", "premature_finish",
                                     "hallucinated_result", "tool_call_malformed")
                   and not o.passed)
    gate = evaluate({
        "executable_rate": audit["executable"]["executable_rate"],
        "consistency_rate": audit["deterministic"]["consistency_rate"],
        "action_gap": action_row["gap"],
        "action_runs": action_row["runs"],
        "action_strong": action_row["full"],
        "action_weak": action_row["no_order_info"],
        "critical_count": critical,
    })

    all_d15 = (action["人工"]["full"] + action["人工"]["no_order_info"]
               + action["自动"]["full"] + action["自动"]["no_order_info"])
    tokens = token_mean(all_d15)
    tokens["note"] = ("单任务平均 = 总 token ÷ 执行次数（{0} 次，D15+D16 合计）。"
                      "**不报金额**：峰谷定价（工作日 9-12 / 14-18 双倍）和缓存命中率"
                      "（命中与未命中差 100 倍以上）都会让同一批实验算出不同数字，复现不出来。"
                      "token 是精确的。").format(tokens["runs"])

    human_action = compare_by_type(pick(action["人工"]["full"], human_tasks, "要做事"),
                                   pick(action["人工"]["no_order_info"], human_tasks, "要做事"),
                                   human_tasks)["要做事"]
    strength = build_strength(
        [
            ("D13 抽样（30 条里的 16 条要做事）",
             pick(old["自动"]["full"], auto_tasks, "要做事"),
             pick(old["自动"]["no_order_info"], auto_tasks, "要做事"), auto_tasks),
            ("D15 扩样本（57 条 × 2 次）",
             pick(d15["自动"]["full"], auto_tasks, "要做事"),
             pick(d15["自动"]["no_order_info"], auto_tasks, "要做事"), auto_tasks),
            ("D15+D16（57 条 × 3 次）",
             pick(action["自动"]["full"], auto_tasks, "要做事"),
             pick(action["自动"]["no_order_info"], auto_tasks, "要做事"), auto_tasks),
        ],
        action_row["gap"], action_row["full"], action_row["no_order_info"])

    data = {
        "conclusion": (
            "两个任务集在「要做事」组都是「完整 > 缺订单信息」的方向（人工 {0:+.3f}、自动 {1:+.3f}），"
            "但幅度都在噪声内，未达显著。<br>"
            "<strong>结论：方向一致，幅度未达显著</strong>——按观测到的通过率，"
            "要判定这个幅度真实存在需要 N ≥ {2}（现在 N={3}）。"
            "<br>「方向一致」不依赖显著性：两个集的 gap 符号相同，这一点成立。"
        ).format(human_action["gap"], action_row["gap"], strength["need_n"], action_row["runs"]),
        "config_note": (
            "两种配置：<strong>完整</strong>（6 个工具全给）vs <strong>缺订单信息</strong>"
            "（去掉 <code>get_order</code>）。循环和 system prompt 完全相同，<strong>只差 tools 里的一项</strong>——"
            "只改一处才能把差异归因到「缺了订单信息」。"
            "不叫 strong / weak：缺信息不是「弱」，它在「要做事」组更低、在「不该动」组反而更高。"
        ),
        "strength": strength,
        "meta": "数据源：D15 扩样本（自动集 57 条 × 2 次、人工集 22 条 × 3 次，完整 / 缺订单信息各跑一遍）"
                "+ D13 全量（不该动组）。报告只报 token，不报金额。",
        "cards": [
            {"label": "可执行率", "value": "{0:.1%}".format(audit["executable"]["executable_rate"]),
             "how": "跑完不崩的次数 / 总次数（D8，30 条 × 3 次）"},
            {"label": "一致率", "value": "{0:.1%}".format(audit["deterministic"]["consistency_rate"]),
             "how": "同一条跑 3 次判定一致的任务数 / 判定过的任务数"},
            {"label": "要做事组 gap", "value": "{0:+.3f}".format(action_row["gap"]),
             "how": "p_strong − p_weak（自动集 {0} 条 / {1} 次，门槛 {2:.3f}）".format(
                 action_row["tasks"], action_row["runs"], action_row["threshold"])},
            {"label": "单任务 token", "value": "{0:,}".format(tokens["total"]),
             "how": "输入 {0:,} + 输出 {1:,}，{2} 次执行的均值".format(
                 tokens["input"], tokens["output"], tokens["runs"])},
        ],
        "gate": gate,
        "matrix": matrix,
        # 分布要覆盖两组：要做事来自 D15，不该动来自 D13（不然"不该动"那排全是 0）
        "distribution": build_distribution(
            pick(action["自动"]["full"], auto_tasks, "要做事")
            + pick(old["自动"]["full"], auto_tasks, "不该动"),
            pick(action["自动"]["no_order_info"], auto_tasks, "要做事")
            + pick(old["自动"]["no_order_info"], auto_tasks, "不该动"), auto_tasks),
        "instruction_following": build_instruction_following(
            human_tasks, action["人工"]["full"], action["人工"]["no_order_info"]),
        "tokens": tokens,
        "perf": build_perf(action),
        "model_compare": build_model_compare(),
        "trajectories": build_trajectories(),
    }
    # 平台（server/app.py）从这份 json 读矩阵和门禁，不重新算一遍——
    # 重新算就意味着"报告里那个数"和"平台里那个数"是两份数据，迟早对不上
    # 轨迹不导出：1.3MB 里 99% 是轨迹，平台用不到（它现场跑任务）。
    # 去掉之后这份 json 只有几十 KB，浏览器打开实验详情不会卡。
    platform_view = {k: v for k, v in data.items() if k != "trajectories"}
    platform_view["trajectory_count"] = len(data.get("trajectories") or [])
    (RESULTS / "report_data.json").write_text(
        json.dumps(platform_view, ensure_ascii=False, indent=2), encoding="utf-8")
    path = write_report(data, RESULTS / "report.html")
    print("门禁:", gate["status"], "| 未过:", gate["failed"] or "无")
    for row in matrix:
        print("  {0} {1}: 完整 {2:.3f} 缺订单信息 {3:.3f} gap {4:+.3f} n={5} 门槛 {6:.3f} | {7}".format(
            row["set"], row["bucket"], row["full"], row["no_order_info"], row["gap"],
            row["n"], row["threshold"], row["direction"]))
    print("  token 单任务平均: 输入 {0:,} / 输出 {1:,} / 合计 {2:,}".format(
        tokens["input"], tokens["output"], tokens["total"]))
    print("报告:", path)
    print("diff:", (RESULTS / "diff.md").resolve())


if __name__ == "__main__":
    main()
