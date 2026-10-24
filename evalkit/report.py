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
            row["set"], row["bucket"], _fmt(row["strong"]), _fmt(row["weak"]),
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
        follow_block = """<h2>五、指令遵循（027 / 028）</h2>
<table><thead><tr><th>任务</th><th>Agent</th><th>指令</th><th>期望金额</th><th>实际提交</th><th>结论</th></tr></thead>
<tbody>{rows}</tbody></table>
<div class="legend">{note}</div>""".format(rows=follow_rows, note=following["note"])

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
{banner}

<h2>一、四个数字</h2>
<div class="cards">{cards}</div>

<h2>二、门禁明细</h2>
<table><thead><tr><th>项</th><th>实际</th><th>门槛</th><th>结果</th><th>怎么算的</th></tr></thead>
<tbody>{gate_rows}</tbody></table>

<h2>三、矩阵</h2>
<table><thead><tr><th>集</th><th>组</th><th>strong</th><th>weak</th><th>gap</th>
<th>N（观测）</th><th>门槛</th><th>方向</th><th>数据源</th></tr></thead>
<tbody>{matrix_rows}</tbody></table>
<div class="legend">要做事组 gap = p_strong − p_weak（正数=弱 Agent 做不成事）；
不该动组 gap = p_weak − p_strong（正数=弱 Agent 更保守）。两个数不合并。</div>

<h2>四、区分度分布</h2>
<div class="bars">{bars}</div>
<div class="legend">{dist_note}</div>

{follow_block}

<h2>六、Token 消耗（单任务平均）</h2>
<table><thead><tr><th>项</th><th>值</th><th>怎么算的</th></tr></thead>
<tbody>{token_rows}</tbody></table>
<div class="legend">{token_note}</div>

<h2>七、轨迹查看器</h2>
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
                         dist_note=dist["note"], payload=payload,
                         follow_block=follow_block, token_rows=token_rows,
                         token_note=tokens["note"])


def write_report(data: dict[str, Any], out_path: str | Path) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_html(data), encoding="utf-8")
    return path.resolve()
