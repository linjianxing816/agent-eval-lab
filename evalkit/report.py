"""D14 报告页：生成一个双击就能打开的单文件 HTML（file:// 协议，不依赖服务）。

**所有数据在构建时内联进 HTML**，页面里没有任何 fetch——
`file://` 下 fetch 会被 CORS 拦，双击打开就是白屏。

七块：四个数字 / 门禁明细 / 矩阵 / 区分度分布 / 指令遵循 / Token 消耗 / 轨迹查看器。
**不报金额，只报 token**——金额受峰谷定价和缓存命中影响，复现不出同一个数。
样式：浅色底、中文、数字等宽、无渐变无动画。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CSS = """
:root{--ink:#1a1a1a;--sub:#5b6470;--line:#dfe3e8;--bg:#f7f8fa;--card:#fff;
--ok:#0a7f3f;--bad:#b3261e;--warn:#8a5a00;--accent:#1f5fbf}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.65 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:28px 20px 60px}
h1{font-size:22px;margin:0 0 4px}
h2{font-size:16px;margin:32px 0 12px;padding-bottom:6px;border-bottom:1px solid var(--line)}
.meta{color:var(--sub);font-size:12px;margin-bottom:20px}
.num,.mono{font-family:ui-monospace,SFMono-Regular,Consolas,"Courier New",monospace;
font-variant-numeric:tabular-nums}
.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:14px 16px}
.card .k{font-size:12px;color:var(--sub)}
.card .v{font-size:26px;font-weight:600;margin:4px 0 2px}
.card .h{font-size:11px;color:var(--sub);line-height:1.5}
table{width:100%;border-collapse:collapse;background:var(--card);font-size:13px}
th,td{border:1px solid var(--line);padding:7px 10px;text-align:left}
th{background:#eef1f5;font-weight:600}
td.n{text-align:right;font-family:ui-monospace,Consolas,monospace;font-variant-numeric:tabular-nums}
.verdict{display:inline-block;padding:2px 10px;border-radius:4px;font-weight:600;font-size:13px}
.verdict.pass{background:#e3f4e9;color:var(--ok);border:1px solid #b6e0c5}
.verdict.fail{background:#fdeaea;color:var(--bad);border:1px solid #f3c2c2}
.banner{padding:12px 16px;border-radius:6px;margin:14px 0 0;font-size:13px}
.conclusion{background:#fff;border:1px solid var(--line);border-left:4px solid var(--accent);
border-radius:6px;padding:14px 16px;margin:0 0 14px;font-size:13.5px;line-height:1.75}
.banner.pass{background:#e3f4e9;border:1px solid #b6e0c5}
.banner.fail{background:#fdeaea;border:1px solid #f3c2c2}
.banner ul{margin:6px 0 0 18px;padding:0}
.viewer{display:grid;grid-template-columns:330px 1fr;gap:14px;align-items:start}
.list{background:var(--card);border:1px solid var(--line);border-radius:6px;
max-height:560px;overflow:auto}
.item{padding:8px 10px;border-bottom:1px solid var(--line);cursor:pointer;font-size:12px}
.item:hover{background:#f2f5f9}
.item.on{background:#e8f0fb;border-left:3px solid var(--accent)}
.item .tid{font-family:ui-monospace,Consolas,monospace;font-size:12px}
.badge{display:inline-block;padding:1px 6px;border-radius:3px;font-size:11px;margin-left:6px}
.badge.p{background:#e3f4e9;color:var(--ok)}
.badge.f{background:#fdeaea;color:var(--bad)}
.badge.c{background:#fbe4e4;color:#8c1111;font-weight:600}
.detail{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:14px 16px;
min-height:300px}
.step{padding:9px 0;border-bottom:1px dashed var(--line)}
.step:last-child{border-bottom:none}
.step .t{font-family:ui-monospace,Consolas,monospace;font-weight:600}
.step .kv{color:var(--sub);font-size:12px;font-family:ui-monospace,Consolas,monospace;
word-break:break-all}
.step.bad{background:#fdf2f2;border-left:3px solid var(--bad);padding-left:10px;margin-left:-10px}
.note{color:var(--bad);font-size:12px;font-weight:600}
.bars{margin-top:8px}
.bar{display:flex;align-items:center;gap:8px;margin:3px 0;font-size:12px}
.bar .lab{width:92px;color:var(--sub)}
.bar .track{flex:1;height:14px;background:#eef1f5;border:1px solid var(--line)}
.bar .fill{height:100%}
.bar .val{width:44px;text-align:right;font-family:ui-monospace,Consolas,monospace}
.legend{font-size:12px;color:var(--sub);margin-top:8px}
.empty{color:var(--sub);font-size:13px}
"""


def _fmt(value: float, digits: int = 3) -> str:
    return "{0:.{1}f}".format(value, digits)


def build_html(data: dict[str, Any]) -> str:
    gate = data["gate"]
    cards = data["cards"]
    matrix = data["matrix"]
    dist = data["distribution"]
    tokens = data.get("tokens") or {"input": 0, "output": 0, "total": 0, "runs": 0, "note": ""}
    following = data.get("instruction_following") or {"rows": [], "note": ""}
    tasks = data["trajectories"]

    status_cls = "pass" if gate["status"] == "PASS" else "fail"
    banner = ['<div class="banner {0}"><strong>门禁 {1}</strong>'.format(status_cls, gate["status"])]
    if gate["failed"]:
        banner.append("<ul>" + "".join("<li>{0} 未过门槛</li>".format(x) for x in gate["failed"]) + "</ul>")
    for warning in gate["warnings"]:
        banner.append("<div style='margin-top:6px;color:#8a5a00'>警告：{0}</div>".format(warning))
    banner.append("</div>")

    gate_rows = "".join(
        "<tr><td>{0}</td><td class='n'>{1}</td><td class='n'>{2}</td>"
        "<td>{3}</td><td class='h'>{4}</td></tr>".format(
            c["name"], _fmt(c["value"]), _fmt(c["threshold"]),
            "通过" if c["ok"] else "<span style='color:#b3261e;font-weight:600'>未过</span>",
            c["how"])
        for c in gate["checks"]
    )

    matrix_rows = "".join(
        "<tr><td>{0}</td><td>{1}</td><td class='n'>{2}</td><td class='n'>{3}</td>"
        "<td class='n'>{4}</td><td class='n'>{5}</td><td class='n'>{6}</td>"
        "<td>{7}</td><td>{8}</td></tr>".format(
            row["set"], row["bucket"], _fmt(row["full"]), _fmt(row["no_order_info"]),
            "{0:+.3f}".format(row["gap"]), row.get("n", ""), _fmt(row.get("threshold", 0.0)),
            row["direction"], row.get("source", ""))
        for row in matrix
    )

    max_count = max([d["count"] for d in dist["buckets"]] + [1])
    bars = "".join(
        '<div class="bar"><span class="lab">{0}</span>'
        '<span class="track"><span class="fill" style="width:{1:.1f}%;background:{2}"></span></span>'
        '<span class="val">{3}</span></div>'.format(
            b["label"], b["count"] / max_count * 100, b["color"], b["count"])
        for b in dist["buckets"]
    )

    token_rows = "".join(
        "<tr><td>{0}</td><td class='n'>{1}</td><td>{2}</td></tr>".format(name, "{0:,}".format(value), how)
        for name, value, how in (
            ("输入", tokens["input"], "prompt_tokens_mean，单任务平均，取整"),
            ("输出", tokens["output"], "completion_tokens_mean，单任务平均，取整"),
            ("合计", tokens["total"], "两者相加；{0} 次执行的均值".format(tokens["runs"])),
        )
    )

    follow_rows = "".join(
        "<tr><td>{0}</td><td>{1}</td><td>{2}</td><td class='n'>{3}</td><td class='n'>{4}</td>"
        "<td>{5}</td></tr>".format(
            r["task_id"], r["agent"], r["instruction"], "{0:.2f}".format(r["expect"]),
            r["actual"], "按指令" if r["ok"] else "<span style='color:#b3261e;font-weight:600'>按规则自己算</span>")
        for r in following["rows"]
    )
    follow_block = ""
    if follow_rows:
        follow_block = """<h2>七、指令遵循（027 / 028）</h2>
<table><thead><tr><th>任务</th><th>Agent</th><th>指令</th><th>期望金额</th><th>实际提交</th><th>结论</th></tr></thead>
<tbody>{rows}</tbody></table>
<div class="legend">{note}</div>""".format(rows=follow_rows, note=following["note"])

    perf = data.get("perf")
    perf_block = ""
    if perf:
        lat = perf["latency"]
        metric_rows = "".join(
            "<tr><td>{0}</td><td class='n'>{1}</td><td class='n'>{2}</td>"
            "<td class='n'>{3}</td><td class='n'>{4}</td></tr>".format(
                name, _fmt(row["p50"], digits), _fmt(row["p95"], digits),
                _fmt(row["min"], digits), _fmt(row["max"], digits))
            for name, row, digits in (
                ("首 token TTFT（ms）", lat["ttft"], 1),
                ("每 token TPOT（ms）", lat["tpot"], 1),
                ("端到端（ms）", lat["e2e"], 1),
                ("输出 token（个）", lat["output_tokens"], 0),
            )
        )
        conc_rows = "".join(
            "<tr><td class='n'>{0}</td><td class='n'>{1:.2f}</td><td class='n'>{2:.0f}</td>"
            "<td class='n'>{3:.0f}</td><td class='n'{4}>{5}</td></tr>".format(
                r["concurrency"], r["throughput"], r["p50"], r["p95"],
                " style='color:#b3261e;font-weight:600'" if r["status_429"] else "",
                r["status_429"])
            for r in perf["concurrency"]
        )
        reuse = perf.get("reuse")
        reuse_html = ""
        if reuse:
            lows, highs = min(reuse["after_runs"]), max(reuse["after_runs"])
            reuse_html = """<div class="legend" style="margin-top:10px">连接复用：
每次新建 <code>httpx.Client</code> 时 <strong>{b:.0f} ms</strong> → 复用后 <strong>{lo:.0f}–{hi:.0f} ms</strong>
（两次会话）。差值 23–81 ms，落在 TTFT 的会话波动里，<strong>不能说省了多少</strong>；
能说的是<strong>握手不是主要成分</strong>——如果 TLS 握手占 200 ms 以上，复用后必然大幅下降，实际没有。</div>""".format(
                b=reuse["before_p50"], lo=lows, hi=highs)

        attrib = perf.get("attribution")
        attrib_html = ""
        if attrib:
            attrib_html = """<div class="legend" style="margin-top:14px">TTFT 归因：同一个 prompt 带 / 不带 6 个工具 schema，<strong>配对交替</strong>测 {pairs} 轮。</div>
<table><thead><tr><th>场景</th><th>输入 token</th><th>TTFT P50（ms）</th><th>说明</th></tr></thead><tbody>
<tr><td>带 6 个工具 schema</td><td class='n'>{wt}</td><td class='n'>{wttft}</td><td class="hint">评测时的真实输入</td></tr>
<tr><td>不带 schema</td><td class='n'>{nt}</td><td class='n'>{nttft}</td><td class="hint">只剩 system prompt + 用户问题</td></tr>
<tr><td><strong>中位差（配对）</strong></td><td class='n'>+{dt}</td><td class='n'>+{dttft}</td>
<td class="hint">{msp} ms / 千 token</td></tr>
</tbody></table>
<div class="legend">拆开看：<strong>prefill 只占 {ps:.0f}%</strong>、基线占 {bs:.0f}%。
配对是为了控住漂移：同一 prompt 的 TTFT 在不同时刻能差 1 秒以上，
分批测出来的差（上一版算出 37%）其实是时段差异，不是输入长度的影响。</div>
{reuse_html}""".format(
                wt=attrib["with_tools_tokens"], wttft=_fmt(attrib["with_tools_ttft"], 1),
                nt=attrib["without_tools_tokens"], nttft=_fmt(attrib["without_tools_ttft"], 1),
                dt=attrib["delta_tokens"], dttft=_fmt(attrib["delta_ttft"], 1),
                msp=_fmt(attrib["ms_per_1k"], 0),
                ps=attrib["prefill_share"], bs=attrib["baseline_share"],
                pairs=attrib.get("pairs", 10), reuse_html=reuse_html)

        perf_block = """<h2>九、推理性能</h2>
<div class="legend">单请求 {samples} 次（串行，间隔 0.5s），同一个 prompt + 6 个工具 schema，和评测时一致。</div>
<table><thead><tr><th>指标</th><th>P50</th><th>P95</th><th>min</th><th>max</th></tr></thead>
<tbody>{metric_rows}</tbody></table>
<div class="legend">首 token 占端到端 <strong>{share:.0f}%</strong>（P50）——剩下 {rest:.0f}% 是逐 token 生成。</div>
<div class="legend" style="margin-top:14px">并发扫描：每档 12 个请求，429 不重试（429 本身就是结论）。</div>
<table><thead><tr><th>并发</th><th>吞吐（req/s）</th><th>P50（ms）</th><th>P95（ms）</th><th>429</th></tr></thead>
<tbody>{conc_rows}</tbody></table>
{attrib_html}
<div class="conclusion" style="margin-top:14px">{conclusion}</div>
<div class="legend" style="margin-top:10px;padding:10px 12px;background:#fff;border:1px solid #dfe3e8">
<strong>这一节的数能用到什么程度</strong><br>
· <strong>TTFT 绝对值不可跨会话比较</strong>：同一个 prompt 的 P50 在 405–1586 ms 之间波动，
报告里的 700 ms 只代表测量那一次会话。<br>
· <strong>prefill 的贡献测不出来</strong>（配对 10 轮，中位差 +42 ms，5/10 轮为正）——
不是没有，是被会话间抖动淹没了。要测准需要固定时段、并发 1、50 轮以上配对。<br>
· 可用的结论只有一条：<strong>单次请求的延迟不可控</strong>（服务端排队），
能控制的只有请求次数。所以 <code>max_steps</code> 是时间和成本的主要杠杆。
</div>
<div class="legend">{perf_note}　实测时间 {measured}</div>""".format(
            samples=lat["ttft"]["n"], metric_rows=metric_rows, share=perf["share"],
            rest=100 - perf["share"], conc_rows=conc_rows, attrib_html=attrib_html,
            conclusion=perf["conclusion"], perf_note=perf["note"],
            measured=perf["measured_at"])

    conc = data.get("concurrency")
    conc_block = ""
    if conc:
        best_level = conc["best"].get("best")
        min_waves = conc.get("min_waves", 3.0)
        conc_rows = "".join(
            "<tr{cls}><td class='n'>{label}</td><td class='n'>{wall:.1f}s</td>"
            "<td class='n'>{tp:.2f}</td><td class='n'>{sp:.2f}×</td><td class='n'{wcls}>{waves:.2f}</td>"
            "<td class='n'{e429}>{n429}</td><td class='n'>{err}</td>"
            "<td class='n'>{lat:.0f}</td><td class='n'>{within:.0%}</td>"
            "<td class='n'>{cross:.0%}</td><td>{tag}</td></tr>".format(
                cls=" style='background:#eef7f0'" if r["concurrency"] == best_level and not r.get("label") else "",
                label=str(r["concurrency"]) + ("（复测）" if r.get("label") else ""),
                wall=r["wall_s"], tp=r["throughput"], sp=r["speedup"],
                waves=r.get("waves", 0.0),
                wcls=" style='color:#8a5a00;font-weight:600'" if not r.get("enough_waves", True) else "",
                n429=r["status_429"],
                e429=" style='color:#b3261e;font-weight:600'" if r["status_429"] else "",
                err=r["request_error"], lat=r["latency_mean_ms"],
                within=r["within"]["rate"], cross=r["agreement"]["rate"],
                tag=("最优" if r["concurrency"] == best_level and not r.get("label")
                     else ("<span style='color:#5b6470'>基线（和自己比，恒 100%）</span>"
                           if r["concurrency"] == 1 and not r.get("label")
                           else ("<span style='color:#8a5a00'>样本不足，加速比偏高</span>"
                                 if not r.get("enough_waves", True) else ""))),
            )
            for r in conc["rows"]
        )
        conc_block = """<h2>{no}、并发调优</h2>
<div class="conclusion">{conclusion}</div>
<div class="legend" style="margin:0 0 12px;padding:10px 12px;background:#fff;border-left:4px solid #c77700;border-radius:6px">{scale}</div>
<div class="legend" style="margin:0 0 12px;padding:10px 12px;background:#fff;border-left:4px solid #0a7f3f;border-radius:6px">{burst}</div>
<table><thead><tr><th>并发</th><th>总耗时</th><th>吞吐（执行/s）</th><th>加速比</th><th>波数</th>
<th>429</th><th>request_error</th><th>单次延迟（ms）</th><th>任务内一致率</th><th>跨档一致率</th><th></th></tr></thead>
<tbody>{rows}</tbody></table>
<div class="legend" style="margin-top:10px;padding:10px 12px;background:#fff;border-left:4px solid #1f5fbf;border-radius:6px">{stability}</div>
<div class="legend" style="margin-top:10px;padding:10px 12px;background:#fff;border:1px solid #dfe3e8">
<strong>这一节的数怎么读</strong><br>
· <strong>波数 = 执行数 ÷ 并发度</strong>，是这个表里最容易被忽略的一列。
波数 &lt; {minw:.0f} 时全部执行一两波就打完了，墙钟约等于「最慢那个任务要多久」，
<strong>加速比测的是这个，不是吞吐，所以偏低</strong>（不是偏高——实测 0.94 波 18.40×、2.81 波 25.41×）。<br>
· <strong>两个一致率口径不同</strong>：任务内 = 同一条跑 3 次判定是否相同（D8 同款，每档自己算）；
跨档 = 和串行档逐条比（混着模型自己的随机性）。<strong>前者才是「并发有没有影响」的答案。</strong><br>
· <strong>单次延迟会变长</strong>，这是正常的——{best} 个请求同时在排队。收益来自并发数，不来自单次变快。<br>
· {note}
</div>""".format(no=("十" if perf else "九"), conclusion=conc["conclusion"],
                 rows=conc_rows, scale=conc.get("scale", ""),
                 burst=conc.get("burst", ""),
                 stability=conc.get("stability", ""), note=conc["note"],
                 minw=min_waves, best=conc["best"].get("best"))

    # 不稳定任务单独一栏。**不是"难度高"，是"同一句指令走不同的路"**——
    # 放在矩阵下面，因为读者看到 N 变小会先问"为什么"。
    unstable = data.get("unstable") or {"rows": [], "count": 0, "share": 0.0,
                                        "total": 0, "note": ""}
    unstable_block = ""
    if unstable.get("rows"):
        unstable_rows = "".join(
            "<tr><td>{0}</td><td>{1}</td><td>{2}</td><td class='hint'>{3}</td>"
            "<td class='n'>{4}</td><td class='n'>{5}</td><td class='n'>{6:.0%}</td>"
            "<td class='hint'>{7}</td></tr>".format(
                r["task_id"], r["set"], r["scenario"], r["instruction"],
                r["runs"], r["passed"], r["rate"], r["failure"])
            for r in unstable["rows"]
        )
        unstable_block = """<div class="legend" style="margin-top:18px;padding:12px 14px;background:#fff;
border-left:4px solid #c77700;border-radius:6px">
<strong>不稳定任务：{count} 条，占全部 {total} 条的 {share:.0%}</strong><br>{note}</div>
<table style="margin-top:10px"><thead><tr><th>任务</th><th>集</th><th>类别</th><th>指令</th>
<th>主数据源里的执行</th><th>通过</th><th>通过率</th><th>失败原因</th></tr></thead>
<tbody>{rows}</tbody></table>
<div class="legend">注意「执行次数」这一列：主数据源里每条只有 1–3 次，
<strong>根本不够判稳</strong>——真正的证据是诊断脚本跑的 19 次以上执行。
这一栏的通过率只用来表示"当前数据里它落在哪"，不要当准确值看。</div>""".format(
            count=unstable["count"], total=unstable["total"], share=unstable["share"],
            note=unstable["note"], rows=unstable_rows)

    strength = data.get("strength") or {"rows": [], "note": "", "need_n": 0}
    strength_rows = "".join(
        "<tr><td>{0}</td><td class='n'>{1}</td><td class='n'>{2:+.4f}</td>"
        "<td class='n'>±{3:.4f}</td><td class='n'>±{4:.4f}</td><td>{5}</td></tr>".format(
            r["label"], r["n"], r["gap"], r["se"], r["band"], r["verdict"])
        for r in strength["rows"]
    )

    mc = data.get("model_compare")
    model_block = ""
    if mc:
        metric_rows = "".join(
            "<tr><td>{0}</td><td class='n'>{1}</td><td class='n'>{2}</td>"
            "<td class='n'>{3:+.3f}</td><td class='n'>{4:.3f}</td>"
            "<td>{5}</td><td class='hint'>{6}</td></tr>".format(
                r["metric"], _fmt(r["a"], 3), _fmt(r["b"], 3), r["gap"], r["threshold"],
                "<span style='color:#b3261e;font-weight:600'>能区分</span>" if r["ok"] else "分不出",
                r["how"])
            for r in mc["rows"]
        )
        cross_rows = "".join(
            "<tr><td>{0}</td><td class='n'>{1}</td><td class='n'>{2}</td>"
            "<td class='n'>{3}</td><td class='n'>{4}</td><td class='n'>{5}</td>"
            "<td>{6}</td></tr>".format(
                row["scenario"],
                _fmt(row["A"]["pass"], 3), row["A"]["n"],
                _fmt(row["B"]["pass"], 3), row["B"]["n"],
                row["A"]["red"] + row["B"]["red"],
                "A 更好" if (row["A"]["pass"] or 0) > (row["B"]["pass"] or 0)
                else ("B 更好" if (row["A"]["pass"] or 0) < (row["B"]["pass"] or 0) else "持平"))
            for row in mc["cross"]
        )
        sep = mc["separability"]
        a_win = "、".join(sep["a_full_win"]) or "无"
        b_win = "、".join(sep["b_full_win"]) or "无"
        model_block = """<h2>六、指标分层（模型 A vs B）</h2>
<div class="legend">同一个数据集、同一套判定，只换指标口径。
<strong>修完之后两个指标结论一致（都能区分）</strong>；但在修之前它们曾经<strong>完全相反</strong>——
通过率说 B 更好、红线违规率说 B 更差。<strong>那个矛盾本身就是坏题的证据</strong>：
只有任务定义出错时，两个指标才会指向相反方向。</div>
<table><thead><tr><th>指标</th><th>A</th><th>B</th><th>gap（A−B）</th><th>门槛</th><th>能区分吗</th><th>怎么算的</th></tr></thead>
<tbody>{metric_rows}</tbody></table>
<div class="legend" style="margin-top:10px;padding:10px 12px;background:#fff;border:1px solid #dfe3e8">{red_note}</div>
<div class="legend">{note}</div>
<div class="legend" style="margin-top:8px">{drop_note}</div>
<div class="conclusion" style="margin-top:12px">{why}</div>

<div class="legend" style="margin-top:16px">六类任务 × 两个模型。<strong>每格都带 N</strong>——30 条分 6 类，每类只有 4–8 条，比例很容易被一条带偏。</div>
<table><thead><tr><th>类别</th><th>A 通过率</th><th>A 的 N</th><th>B 通过率</th><th>B 的 N</th>
<th>越权次数（A+B）</th><th>谁更好</th></tr></thead>
<tbody>{cross_rows}</tbody></table>
<div class="legend">{cross_note}</div>

<div class="legend" style="margin-top:16px"><strong>逐条任务比通过率</strong>（每条跑 3 次，所以有 3/3 vs 1/3 这种中间态）。
两档口径都列，因为宽松那档会虚高：</div>
<table><thead><tr><th>口径</th><th>任务数</th><th>占 30 条</th><th>判定规则</th></tr></thead><tbody>
<tr><td><strong>严格：完全分离</strong></td><td class='n'><strong>{strict}</strong></td><td class='n'><strong>{strict_pct:.0%}</strong></td>
<td class="hint">一边 3/3、另一边 0/3，没有歧义</td></tr>
<tr><td>宽松：通过率不等</td><td class='n'>{sep}</td><td class='n'>{sep_pct:.0%}</td>
<td class="hint">1/3 vs 0/2 也算——可两个数都表示「基本做不成」，差异没意义</td></tr>
<tr><td>两个模型表现完全相同</td><td class='n'>{same}</td><td class='n'>{same_pct:.0%}</td>
<td class="hint">都过 或 都不过，对区分没有贡献</td></tr>
</tbody></table>
<table style="margin-top:10px"><thead><tr><th>完全分离的任务</th><th>哪几条</th></tr></thead><tbody>
<tr><td>A 赢（A 3/3、B 0/3）</td><td class="hint">{a_win}</td></tr>
<tr><td>B 赢（B 3/3、A 0/3）</td><td class="hint">{b_win}</td></tr>
</tbody></table>
<div class="legend">「<strong>{strict}/{total}</strong>」这个数比 gap 直观：它回答<strong>这套题里有多少道题真的能分出好坏</strong>。
30 条里 {same} 条两个模型都做对或都做错，对区分毫无贡献——这就是天花板效应的具体形状。
<br>{uneven_note}</div>
<div class="conclusion" style="margin-top:12px">{conclusion}</div>
""".format(metric_rows=metric_rows, cross_rows=cross_rows, note=mc["note"],
           why=mc["why"], same=sep["same"], same_pct=sep["same"] / sep["total"],
           ab=sep["a_better"], ab_pct=sep["a_better"] / sep["total"],
           bb=sep["b_better"], bb_pct=sep["b_better"] / sep["total"],
           sep=sep["separable"], sep_pct=sep["separable"] / sep["total"],
           strict=sep["separable_strict"], strict_pct=sep["separable_strict"] / sep["total"],
           uneven_note=("注：{0} 这条任务 B 侧因剔除 request_error 只剩 2 次观测，"
                        "它的通过率分母和别人不同，读的时候要打折扣。"
                        ).format("、".join(sep.get("uneven") or []))
                       if sep.get("uneven") else "",
           a_ids="、".join(sep["a_better_ids"]) or "—",
           b_ids="、".join(sep["b_better_ids"]) or "—",
           a_win=a_win, b_win=b_win, total=sep["total"],
           conclusion=mc["conclusion"], red_note=mc.get("red_note", ""),
           cross_note=mc.get("cross_note", ""), drop_note=mc.get("drop_note", ""))

    cards_html = "".join(
        '<div class="card"><div class="k">{0}</div><div class="v num">{1}</div>'
        '<div class="h">{2}</div></div>'.format(c["label"], c["value"], c["how"])
        for c in cards
    )

    payload = json.dumps(tasks, ensure_ascii=False).replace("</", "<\\/")
    return """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>LLM 评测平台 · 报告</title><style>{css}</style></head>
<body><div class="wrap">
<h1>LLM Agent 评测 · 报告</h1>
<div class="meta">{meta}</div>
<div class="conclusion">{conclusion}</div>
<div class="legend" style="margin:0 0 12px;padding:10px 12px;background:#fff;border:1px solid #dfe3e8">{config_note}</div>
{banner}

<h2>一、四个数字</h2>
<div class="cards">{cards}</div>

<h2>二、门禁明细</h2>
<table><thead><tr><th>项</th><th>实际</th><th>门槛</th><th>结果</th><th>怎么算的</th></tr></thead>
<tbody>{gate_rows}</tbody></table>
<div class="legend" style="margin-top:10px;padding:10px 12px;background:#fff;border:1px solid #dfe3e8">
<strong>门槛公式已于 2026-10-25 修正</strong>：要做事组 gap 是「两个比例之差」，
标准误用 <code>sqrt(p₁(1−p₁)/n + p₂(1−p₂)/n)</code>。
早期版本用单比例的 <code>sqrt(0.25/n)</code>，<strong>偏松 √2 倍</strong>——
N=171 时旧门槛 0.0765、新门槛 0.1056，gap 0.0877 从「过」变成「未过」。
<strong>用错公式不是"多一点误差"，是让不该过的 gate 过。</strong>
</div>

<h2>三、矩阵</h2>
<table><thead><tr><th>集</th><th>组</th><th>完整</th><th>缺订单信息</th><th>gap</th>
<th>N（观测）</th><th>门槛</th><th>方向</th><th>数据源</th></tr></thead>
<tbody>{matrix_rows}</tbody></table>
<div class="legend">要做事组 gap = p_完整 − p_缺订单信息（正数=缺订单信息的 Agent 做不成事）；
不该动组 gap = p_缺订单信息 − p_完整（正数=缺订单信息的 Agent 更保守）。两个数不合并。</div>
{unstable_block}
<h2>四、样本量与结论强度</h2>
<table><thead><tr><th>阶段</th><th>N（单 Agent 观测）</th><th>gap</th>
<th>标准误 ±</th><th>误差带（2×se）</th><th>这个数能说明什么</th></tr></thead>
<tbody>{strength_rows}</tbody></table>
<div class="legend">{strength_note}</div>
<div class="legend" style="margin-top:10px;padding:10px 12px;background:#fff;border:1px solid #dfe3e8">
<strong>不是效果小，是样本不够。</strong>这句话本身是结论——
误差带比 gap 还宽的时候，报"gap 是 0.088"和报"gap 是 0"没有区别。</div>

<h2>五、区分度分布</h2>
<div class="bars">{bars}</div>
<div class="legend">{dist_note}</div>

{model_block}

{follow_block}

<h2>八、Token 消耗（单任务平均）</h2>
<table><thead><tr><th>项</th><th>值</th><th>怎么算的</th></tr></thead>
<tbody>{token_rows}</tbody></table>
<div class="legend">{token_note}</div>

{perf_block}

{conc_block}

<h2>{traj_no}、轨迹查看器</h2>
<div class="viewer">
  <div class="list" id="list"></div>
  <div class="detail" id="detail"><div class="empty">左边点一条任务看轨迹</div></div>
</div>
</div>
<script>
const DATA = {payload};
const listEl = document.getElementById('list');
const detailEl = document.getElementById('detail');
DATA.forEach((t, i) => {{
  const div = document.createElement('div');
  div.className = 'item';
  const cls = t.badge === '严重失败' ? 'c' : (t.passed ? 'p' : 'f');
  div.innerHTML = '<span class="tid">' + t.task_id + '</span>' +
    '<span class="badge ' + cls + '">' + t.badge + '</span><br>' +
    '<span style="color:#5b6470">' + t.agent + ' · ' + t.scenario + '</span>';
  div.onclick = () => show(i, div);
  listEl.appendChild(div);
}});
function esc(s) {{ return String(s).replace(/[&<>]/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;'}}[c])); }}
function show(i, el) {{
  document.querySelectorAll('.item').forEach(e => e.classList.remove('on'));
  el.classList.add('on');
  const t = DATA[i];
  let html = '<div><strong>' + t.task_id + '</strong>　<span style="color:#5b6470">' +
    t.agent + ' · ' + t.scenario + ' · 判定 ' + t.badge + '</span></div>' +
    '<div class="kv" style="margin:6px 0 12px">指令：' + esc(t.instruction) + '</div>';
  if (!t.steps.length) {{
    html += '<div class="empty">这条没有轨迹（跑崩了）</div>';
  }}
  t.steps.forEach(s => {{
    html += '<div class="step' + (s.forbidden ? ' bad' : '') + '">' +
      '<div class="t">step ' + s.step + '　' + esc(s.tool) + '</div>' +
      '<div class="kv">参数　' + esc(s.args) + '</div>' +
      '<div class="kv">返回　' + esc(s.result) + '</div>' +
      '<div class="kv">耗时　' + s.elapsed_ms + ' ms' +
      (s.error ? '　错误　' + esc(s.error) : '') + '</div>' +
      (s.forbidden ? '<div class="note">← 这步不该发生：' + esc(s.tool) + ' 在 forbid_tools 里</div>' : '') +
      '</div>';
  }});
  html += '<div class="kv" style="margin-top:12px">最终状态　' + esc(t.final_state) + '</div>';
  detailEl.innerHTML = html;
}}
if (DATA.length) {{ document.querySelectorAll('.item')[0].click(); }}
</script>
</body></html>""".format(css=CSS, meta=data["meta"], banner="".join(banner), cards=cards_html,
                         gate_rows=gate_rows, matrix_rows=matrix_rows, bars=bars,
                         unstable_block=unstable_block,
                         dist_note=dist["note"], payload=payload,
                         follow_block=follow_block, token_rows=token_rows,
                         token_note=tokens["note"], conclusion=data.get("conclusion", ""),
                         config_note=data.get("config_note", ""),
                         strength_rows=strength_rows, strength_note=strength["note"],
                         perf_block=perf_block, model_block=model_block,
                         conc_block=conc_block,
                         traj_no=("十一" if (perf and conc) else ("十" if (perf or conc) else "九")))


def write_report(data: dict[str, Any], out_path: str | Path) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_html(data), encoding="utf-8")
    return path.resolve()
