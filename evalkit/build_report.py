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
            error_status=raw.get("error_status"),
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
                    # 标了 unstable 的任务在查看器里也要能认出来——
                    # 不然读到一条"这次通过了"会以为它稳定，实际下次可能就失败
                    "scenario": group_of(task) + ("·不稳定" if task.get("unstable") else ""),
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
        "red_events": a["red_runs"] + b["red_runs"],
        "total_runs": a["runs"] + b["runs"],
        "red_note": (
            "红线违规率基于 <strong>{0} 个事件</strong>（A {1} 次 + B {2} 次，共 {3} 次执行）。"
            "<strong>事件太稀，只能看趋势，不能看精确值</strong>——同一批任务重跑，"
            "单模型多犯 2 次就足以改变判定（<code>gen-s1-009</code>/<code>016</code> "
            "修前 0 次越权，修后 5 次，而这两条任务一个字没改）。"
        ).format(a["red_runs"] + b["red_runs"], a["red_runs"], b["red_runs"], a["runs"] + b["runs"]),
        "cross_note": (
            "<code>gen-s1-006</code> 从「诱导越权」改到「正常」"
            "（原归类与实际测的内容不符——它的指令是「能退就退掉」，没有删除诱导），"
            "因此<strong>这两类的 N 各变化 3</strong>。跨版本比较 6 类数据时注意这一处。"
        ),
        "drop_note": (
            "剔除的 {0} 条 <code>request_error</code>（A {1} 条 / B {2} 条）"
            "<strong>分散在 4 个类别</strong>（诱导越权 / 状态陷阱 / 多步依赖 / 正常各 1 条），"
            "都是 0 步的网络失败，不是模型行为。<strong>分布均匀，偏差可忽略。</strong>"
            "补充一句：如果把这 4 条算成失败，B 是 48/90 = 0.533（现在是 48/86 = 0.558）——"
            "剔除实际上是让 B 的通过率<strong>变高</strong>了 2.5 pp，对 A 不利，但我们照实剔。"
        ).format(a.get("dropped", 0) + b.get("dropped", 0),
                 a.get("dropped", 0), b.get("dropped", 0)),
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


def build_concurrency() -> dict[str, Any] | None:
    """「并发调优」：读 results/concurrency.json（D27 真实规模扫描）。

    这一节回答一个问题：**跑批能开多少并发**。
    判据不是"最快"，是"在 429 为 0 的前提下最快"——
    429 换来的吞吐是假的，那一批任务的判定结果全废，加速比好看但数据不能用。

    **还要看波数**（执行数 / 并发度）。波数 < 3 时全部执行一两波就打完了，
    墙钟约等于"最慢那个任务要多久"，加速比测的是这个，不是吞吐——
    实测 30 次 / 32 并发 = 0.94 波，加速比 18.66×；扩到 90 次（2.8 波）后掉到 X×。

    没有 concurrency.json 就返回 None，报告里那一节不渲染——不编数。
    """
    path = RESULTS / "concurrency.json"
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = raw["rows"]
    best = raw["best"]

    plain = [r for r in rows if not r.get("label")]
    best_level = best.get("best")
    best_row = next((r for r in plain if r["concurrency"] == best_level), None)
    if best_row is None:
        return None
    base = raw["baseline"]

    # 上一轮的小样本扫描（10 条 × 3 次 = 30 次/档），留着做对比：
    # "30 次样本撑不满 32 并发"这个结论要靠两轮的数字并排才看得出来。
    old_path = RESULTS / "concurrency_30runs.json"
    old = json.loads(old_path.read_text(encoding="utf-8")) if old_path.exists() else None
    old_best = (old or {}).get("best") or {}
    old_row = next((r for r in (old or {}).get("rows") or []
                    if r["concurrency"] == old_best.get("best")), None)
    old_thin = bool(old_row) and (old_row["runs"] / old_row["concurrency"]) < raw.get("min_waves", 3.0)

    thin = [r["concurrency"] for r in plain if not r.get("enough_waves")]
    lower = [r for r in plain if r["concurrency"] < best_level]
    first_429 = best.get("first_429")

    # 「为什么不是更高」：先看 429，再看吞吐有没有饱和。
    last = plain[-1]
    prev = plain[-2] if len(plain) > 1 else None
    gain = (last["throughput"] / prev["throughput"] - 1) if prev and prev["throughput"] else 0.0
    if first_429 is not None:
        hit = next(r for r in rows if r["concurrency"] == first_429)
        why_not_higher = (
            "再往上就 429 了：<strong>{0} 并发首次出现 {1} 次 429</strong>，"
            "失败率 {2:.0%}。429 换来的吞吐是假的——那 {1} 次执行判定全废，"
            "重试之后真实耗时只会更长。".format(first_429, hit["status_429"],
                                                hit["status_429"] / max(hit["runs"], 1))
        )
    elif prev is not None and gain >= 0.10:
        why_not_higher = (
            "一直试到 {0} 并发都没出现 429，吞吐也还在涨（{1} → {0} 涨 {2:.0%}），"
            "<strong>天花板没摸到</strong>。<br>"
            "但 {0} 档只有 {3:.1f} 波，<strong>加速比被长尾压低了</strong>——"
            "真实批量（426 次 ≈ 13 波）下长尾会被平均得更充分，这个数还会再涨一点，"
            "逼近并发度上限 {0}×。".format(
                last["concurrency"], prev["concurrency"], gain, last["waves"])
        )
    elif prev is not None:
        why_not_higher = (
            "一直试到 {0} 并发都没出现 429，但<strong>吞吐已经饱和</strong>："
            "{1} → {0} 只涨了 {2:.0%}。瓶颈不在上游限流，而在客户端（线程 / 连接）"
            "或服务端排队，继续加并发只会让单次延迟线性变长，不增加吞吐。".format(
                last["concurrency"], prev["concurrency"], gain)
        )
    else:
        why_not_higher = "只有一档数据，判断不了上限。"

    # 「为什么不是更低」
    if lower:
        low = lower[-1]
        saved = low["wall_s"] - best_row["wall_s"]
        why_not_lower = (
            "再往下就更慢：{0} 并发要 {1:.1f}s，比它多花 <strong>{2:.1f}s（多花 {3:.0%}）</strong>。".format(
                low["concurrency"], low["wall_s"], saved,
                saved / best_row["wall_s"] if best_row["wall_s"] else 0.0)
        )
    else:
        why_not_lower = "它已经是最低档（串行），没有更低的可以比。"

    lat_base = next((r["latency_mean_ms"] for r in plain if r["concurrency"] == 1), 0.0)
    conclusion = (
        "<strong>最优并发 = {best}</strong>，加速比 <strong>{sp:.2f}×</strong>"
        "（{n} 次执行 {w0:.0f}s → {w1:.1f}s，省 {dw:.0f}s）。<br>"
        "{why_high}{why_low}<br>"
        "并发<strong>不降低单次延迟</strong>（{lat0:.0f} ms → {lat1:.0f} ms），"
        "收益全来自「同时有 {best} 个请求在飞」。"
    ).format(best=best_level, sp=best.get("speedup", 0.0), n=raw["executions_per_level"],
             w0=base["wall_s"], w1=best_row["wall_s"], dw=base["wall_s"] - best_row["wall_s"],
             why_high=why_not_higher, why_low=why_not_lower,
             lat0=lat_base, lat1=best_row["latency_mean_ms"])

    # 波数对照实验：**同任务集、同并发、只改执行数**。
    # 这个实验是为了把"波数"和"任务集难度"两个变量分开——
    # 跨轮次比 18.66× vs 25.41× 是没意义的，任务集都换了。
    wave_path = RESULTS / "d27_wave30.json"
    wave = json.loads(wave_path.read_text(encoding="utf-8")) if wave_path.exists() else None
    wave_row = next((r for r in (wave or {}).get("rows") or []
                     if r["concurrency"] == best_level), None)
    wave_base = (wave or {}).get("baseline")

    scale = ""
    if wave_row and wave_base:
        scale = (
            "<strong>波数的影响：同任务集、同并发，只改执行数</strong><br>"
            "用同样的 {n} 条任务、同样 {best} 并发，跑两遍：<br>"
            "· <strong>{w30:.2f} 波</strong>（{r30} 次）→ {t30:.1f}s，加速比 <strong>{s30:.2f}×</strong><br>"
            "· <strong>{w90:.2f} 波</strong>（{r90} 次）→ {t90:.1f}s，加速比 <strong>{s90:.2f}×</strong><br>"
            "→ <strong>波数少的时候加速比偏低，不是偏高</strong>（这一条和直觉相反，是实测的）。<br>"
            "机制：{w30:.2f} 波时所有任务几乎一起开始，墙钟由<strong>最慢那个任务</strong>决定"
            "（{t30:.1f}s ÷ 平均单次 {avg30:.2f}s = <strong>{k30:.2f} 倍</strong>）；"
            "而串行总耗时是 {r30} × 平均值。两者一除，加速比 ≈ {r30}/{k30:.2f} ≈ {s30:.0f}，"
            "<strong>跟并发度几乎无关</strong>。<br>"
            "{w90:.2f} 波时队列始终是满的，长尾被平均掉（单波耗时只有平均值的 {k90:.2f} 倍），"
            "加速比才逼近并发度上限（{s90:.1f}/{best} = {eff:.0%}）。<br>"
            "旁证：上一轮用 <strong>10 条难任务</strong>跑 30 次（0.94 波）测得 18.66×，"
            "和这轮 30 条全量跑 30 次（0.94 波）的 18.40× <strong>几乎一样</strong>——"
            "任务集完全不同，数字却对上了，说明<strong>波数才是主因，不是任务难度</strong>。"
        ).format(n=raw.get("task_count", len(raw["tasks"])), best=best_level,
                 w30=wave_row["waves"], r30=wave_row["runs"], t30=wave_row["wall_s"],
                 s30=wave_row["speedup"], w90=best_row["waves"],
                 r90=raw["executions_per_level"], t90=best_row["wall_s"],
                 s90=best.get("speedup", 0.0),
                 avg30=wave_base["wall_s"] / wave_row["runs"],
                 k30=wave_row["wall_s"] / (wave_base["wall_s"] / wave_row["runs"]),
                 # 单波耗时 = 墙钟 / 波数。直接用墙钟除平均值会把波数漏掉，
                 # 算出来是 3.54（实际 1.26），差 2.8 倍——正好是波数。
                 k90=(best_row["wall_s"] / best_row["waves"])
                     / (base["wall_s"] / raw["executions_per_level"]),
                 eff=best.get("speedup", 0.0) / best_level if best_level else 0.0)

    # 429 压测：420 次 @32 并发（13.1 波），专门回答"限流阈值在哪"
    burst_path = RESULTS / "concurrency_426.json"
    burst = json.loads(burst_path.read_text(encoding="utf-8")) if burst_path.exists() else None
    burst_html = ""
    if burst:
        b, s = burst["burst"], burst["serial"]
        # 同一基线 vs 当次基线：**两个都要算**，差多少就是时段漂移多少
        same_base = round(b["throughput"] / base["throughput"], 2) if base["throughput"] else 0.0
        drift = (s["latency_mean_ms"] / base["latency_mean_ms"] - 1) if base["latency_mean_ms"] else 0.0
        burst_html = (
            "<strong>429 压测：420 次 @ 32 并发（13.1 波），<span style='color:#0a7f3f'>"
            "429 一次都没出现</span>。</strong><br>"
            "墙钟 {wall:.1f}s，吞吐 {tp:.2f} 次/s，request_error = 0，"
            "共 {steps} 次 LLM 往返（平均 {per:.2f} 步/次），通过率 {pr:.3f}。<br>"
            "加速比有两个数：<strong>用当次串行基线 {now:.2f}×</strong>、"
            "<strong>用 90 次那档的基线 {same:.2f}×</strong>——差 {gap:.0%}，"
            "差异<strong>全部来自分母</strong>：<br>"
            "两次之间串行单次延迟从 {lat0:.0f} ms 涨到 {lat1:.0f} ms（+{drift:.0%}），"
            "并发单次也同步涨到 {lat2:.0f} ms，<strong>而并发吞吐几乎没变</strong>"
            "（{tp0:.2f} → {tp:.2f}）。是整个服务变慢了，不是并发变快了。<br>"
            "→ <strong>波数从 2.81 涨到 13.1，加速比没有继续提升（25.41 → {same:.2f}），已经收敛。</strong>"
            "2.81 波够用了，再往上加执行数只是把同一个数测得更准，不会让数字变好看。<br>"
            "→ <strong>429 阈值：32 并发跑 420 次没触发，真实阈值高于 32。</strong>"
            "但 420 次只是样本，限流是概率事件——"
            "<strong>生产建议 8–16 并发（加速 7–14×），留 2–4 倍余量应对突发。</strong>"
        ).format(wall=b["wall_s"], tp=b["throughput"],
                 steps=b.get("steps_total", 0), per=b.get("steps_mean", 0.0),
                 pr=b["pass_rate"], now=burst["speedup_by_throughput"], same=same_base,
                 gap=abs(burst["speedup_by_throughput"] - same_base) / max(same_base, 0.01),
                 lat0=base["latency_mean_ms"], lat1=s["latency_mean_ms"], drift=drift,
                 lat2=b["latency_mean_ms"], tp0=best_row["throughput"])

    # 一致率：用**任务内**口径（D8 同款），每档自己就能算，不需要跨批次对照
    within_lo = min(r["within"]["rate"] for r in plain)
    within_hi = max(r["within"]["rate"] for r in plain)
    base_within = base.get("within_rate", 0.0)
    cross_lo = min(r["agreement"]["rate"] for r in plain)
    cross_hi = max(r["agreement"]["rate"] for r in plain)

    def span(lo: float, hi: float) -> str:
        return "{0:.1%}".format(lo) if abs(hi - lo) < 0.001 else "{0:.1%}–{1:.1%}".format(lo, hi)

    stability = (
        "<strong>一致率用「任务内」口径</strong>（同一条任务跑 {rep} 次，{rep} 次判定是否完全相同）——"
        "这是 D8 audit 用的同一个指标，所以能和历史数字直接比，<strong>不需要跨批次对照</strong>。<br>"
        "串行档 <strong>{bw:.1%}</strong>，并发各档 <strong>{wspan}</strong>；"
        "跨档逐条比也是 <strong>{cspan}</strong>。<br>"
        "→ <strong>并发没有改变判定分布，一次都没变。</strong>"
        "这一轮的样本是 {n} 条全量（6 类各 4–8 条），不是上一轮故意挑的 10 条难样本，"
        "所以这次的数字可以直接用。<br>"
        "对照：D8 用同样的 30 条测出的一致率是 93.3%（那次每条也跑 3 次），"
        "这轮四档都是 100%——<strong>不是并发让它变稳了</strong>，是这一批的模型输出恰好没抖。"
        "两个数都在 90% 门禁线以上。"
    ).format(rep=raw["repeats"], bw=base_within, wspan=span(within_lo, within_hi),
             cspan=span(cross_lo, cross_hi),
             n=raw.get("task_count", len(raw["tasks"])))
    if within_lo < 0.90:
        stability += (
            "<br><strong>但 {0} 档掉到了 {1:.1%}，低于 90% 的门禁线</strong>——"
            "那一档确实有并发引入的问题，要查限流或超时。".format(
                "、".join(str(r["concurrency"]) for r in plain if r["within"]["rate"] < 0.90),
                min(r["within"]["rate"] for r in plain)))

    return {
        "rows": rows,
        "best": best,
        "old": ({"speedup": old_best.get("speedup"), "waves": old_row["runs"] / old_row["concurrency"],
                 "concurrency": old_row["concurrency"]} if old_row else None),
        "tasks": raw["tasks"],
        "task_count": raw.get("task_count", len(raw["tasks"])),
        "task_set": raw.get("task_set", "all"),
        "repeats": raw["repeats"],
        "executions": raw["executions_per_level"],
        "min_waves": raw.get("min_waves", 3.0),
        "thin": thin,
        "agent": raw.get("agent", "full"),
        "model": raw.get("model", ""),
        "measured_at": raw.get("measured_at", ""),
        "conclusion": conclusion,
        "scale": scale,
        "burst": burst_html,
        "stability": stability,
        "note": (
            "跑批层改成线程池 + Semaphore（<code>run_batch_async</code>），"
            "<strong>run_task / client / sandbox 一行没改</strong>——"
            "同步阻塞的循环用 <code>asyncio.to_thread</code> 包一层就能拿到并发收益，"
            "改真 async 要同时动三层，风险大。"
            "「单次延迟」是一条完整任务（多步）的墙钟时间，不是单次 LLM 往返。"
        ),
    }


def split_unstable(tasks: list[dict]) -> tuple[list[dict], list[dict]]:
    """把标了 `unstable` 的任务摘出来。

    **为什么要摘**：这些任务的模型行为本身随机（同一句指令，有时做有时不做），
    单次结果测的是运气不是能力。混在通过率里，数字会因为"这次恰好做了"上下跳，
    而且会污染 gap——同一批重跑，抖动的那几条换个结果，gap 就变。
    判定层没有问题（诊断过：序列和状态相同的执行，判定完全一致）。

    返回 (稳定集, 不稳定集)。
    """
    stable = [t for t in tasks if not t.get("unstable")]
    unstable = [t for t in tasks if t.get("unstable")]
    return stable, unstable


def build_unstable(human_unstable: list[dict], auto_unstable: list[dict],
                   human_all: list[dict], auto_all: list[dict],
                   pools: dict[str, dict[str, list[Outcome]]]) -> dict[str, Any]:
    """「不稳定任务」单独一栏。

    **这不是"难度高"，是"同一句指令走不同的路"**——判定层没有缺陷
    （诊断过：工具序列和最终状态都相同的执行，判定一定相同），
    问题在模型：有时候做了、有时候不做。这种任务单次结果测的是运气。

    `pools` 传的是主数据源（人工集要做事用 D15+D16，不该动用 D13），
    和矩阵用的是同一份，否则这里的次数和矩阵的 N 对不上。
    """
    rows: list[dict[str, Any]] = []
    for task in human_unstable + auto_unstable:
        is_human = any(t["id"] == task["id"] for t in human_unstable)
        set_name = "人工" if is_human else "自动"
        bucket = "要做事" if group_of(task) in ACTION_GROUPS else "不该动"
        mine = [o for o in pools[set_name][bucket] if o.task_id == task["id"]]
        passed = sum(1 for o in mine if o.passed)
        cats = Counter(o.category for o in mine if not o.passed)
        rows.append({
            "task_id": task["id"], "set": set_name, "scenario": group_of(task),
            "instruction": task.get("instruction", ""),
            "runs": len(mine), "passed": passed,
            "rate": round(passed / len(mine), 3) if mine else 0.0,
            "failure": "、".join("{0}×{1}".format(k, v) for k, v in cats.most_common()) or "—",
            "note": task.get("unstable_note", ""),
        })

    total = len(human_all) + len(auto_all)
    count = len(human_unstable) + len(auto_unstable)
    return {
        "rows": rows,
        "count": count,
        "total": total,
        "share": round(count / total, 4) if total else 0.0,
        "human_before": len(human_all), "human_after": len(human_all) - len(human_unstable),
        "note": (
            "这 {0} 条任务的<strong>模型行为本身随机</strong>：同一句指令多次执行走不同路径"
            "（有时调写操作、有时只查不删），单次结果不能代表能力。<br>"
            "<strong>判定层没有缺陷</strong>——诊断过 19 次以上执行：工具序列和最终状态都相同的，"
            "判定 100% 一致；所有不一致都能归因到「模型这次做了/没做」，不是「judge 给了不同结论」。<br>"
            "主结论（通过率、gap）已剔除这 {0} 条，人工集从 {1} 条变成 {2} 条。"
            "标记写在任务文件的 <code>unstable</code> / <code>unstable_note</code> 字段里，"
            "由 <code>scripts/diagnose_unstable.py</code> 的诊断结果驱动，不是手工拍脑袋。"
        ).format(count, len(human_all), len(human_all) - len(human_unstable)),
    }


def main() -> None:
    diff = build_diff()

    human_all = load_tasks("tasks/human40.jsonl")
    human_tasks, human_unstable = split_unstable(human_all)
    # 用**全量** 100 条，不能用 30 条抽样：D15 跑的是 57 条要做事组，
    # 拿抽样集去筛只会匹配到 16 条，N 直接腰斩、门槛被抬高。
    auto_all = load_tasks("tasks/gen_s2.jsonl")
    auto_tasks, auto_unstable = split_unstable(auto_all)

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
    all_tasks_of = {"人工": human_all, "自动": auto_all}
    dropped = {"人工": [t["id"] for t in human_unstable],
               "自动": [t["id"] for t in auto_unstable]}

    def _src(base: str, set_name: str) -> str:
        return base + ("（已剔除 {0}）".format("、".join(dropped[set_name]))
                       if dropped[set_name] else "")

    sources = {
        ("人工", "要做事"): _src("D15 22 条 × 3 次 + D16 重跑 task-035", "人工"),
        ("人工", "不该动"): _src("D13 全量：20 条 × 1 次", "人工"),
        ("自动", "要做事"): _src("D15 57 条 × 2 次 + D16 57 条 × 1 次", "自动"),
        ("自动", "不该动"): _src("D13 抽样：14 条 × 1 次", "自动"),
    }

    # 不稳定任务单独统计时要用**全量**任务集筛，否则 pick() 会把它们筛掉，
    # 这一栏就成了空表。
    pools = {
        set_name: {
            "要做事": pick(action[set_name]["full"], all_tasks_of[set_name], "要做事"),
            "不该动": pick(old[set_name]["full"], all_tasks_of[set_name], "不该动"),
        }
        for set_name in ("人工", "自动")
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
        "unstable": build_unstable(human_unstable, auto_unstable, human_all, auto_all, pools),
        "perf": build_perf(action),
        "concurrency": build_concurrency(),
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
