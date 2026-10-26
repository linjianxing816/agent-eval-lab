"""缺订单信息的 Agent + 区分度测量（D10 / D11）。

【缺订单信息的 Agent 怎么削弱：删掉 `get_order`】
三条原则逐条对照：

1. **绕不过去**：`get_order` 是拿到 `amount` 和 `status` 的唯一入口。
   `get_user` 只给 `name` / `vip`，`calc_refund` 必须先知道订单号且内部会读状态——
   但算出来的金额也拿不回来（它返回 refund_amount，嗯，这条要单独说，见下）。
   所以拿不到金额和状态，就没有替代路径。

2. **影响能力不是运气**：金额只能从五个值里猜（0.01 / 88.8 / 158.0 / 299.0 / 999999.0），
   猜中概率接近 0；状态不知道就没法判断该不该先取消发货。这是**信息缺失**导致的系统性错误，
   不是加温度那种随机波动。

3. **只改一处**：工具集少一个，system prompt 一个字没改。归因干净。

**已知代价**（报告里要写）：有 40% 的任务断言是"状态没变"，缺订单信息的 Agent 什么都不做也能过。
所以它的通过率会被这类任务抬高，区分度被低估——这是任务设计带来的，不是测量误差。
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evalkit.agent.loop import run_task
from evalkit.agent.full import FULL_AGENT, SYSTEM_PROMPT
from evalkit.client import ChatError, ModelClient
from evalkit.env.sandbox import Sandbox
from evalkit.env.state import EnvState
from evalkit.env.tools import TOOL_SCHEMAS
from evalkit.gate import gap_threshold
from evalkit.judge import judge_task

def _without(tool_name: str) -> list[dict]:
    return [t for t in TOOL_SCHEMAS if t["function"]["name"] != tool_name]


NO_ORDER_INFO_AGENT: dict[str, Any] = {          # 第 1/2 轮用的：删 get_order
    "name": "no_order_info",
    "system_prompt": SYSTEM_PROMPT,
    "tools": _without("get_order"),
}
# 第 3 轮的两个候选。删 get_order 绕得过去，因为 calc_refund 是 oracle
# （返回 refund_amount + basis），所以这轮直接打 oracle 和 VIP 信息来源。
NO_CALC_AGENT: dict[str, Any] = {
    "name": "no_calc",
    "system_prompt": SYSTEM_PROMPT,
    "tools": _without("calc_refund"),
}
NO_USER_AGENT: dict[str, Any] = {
    "name": "no_user",
    "system_prompt": SYSTEM_PROMPT,
    "tools": _without("get_user"),
}
# 契约第七节：区分度必须按这两组分开算，不许合并。
# 「不该动」组的正确行为是"什么都不做"，所以公式要反过来——
# 少给信息 → 更保守 → 通过率更高，用同一方向减会读反。
ACTION_GROUPS = ("多步依赖", "边界值", "正常")
UNCHANGED_GROUPS = ("诱导越权", "信息不足", "状态陷阱")


def group_of(task: dict) -> str:
    """取任务类别。

    自动集有 `scenario` 字段直接用；**人工集没有**，按 id 段推
    （1-7 越权 / 8-12 信息不足 / 13-18 状态陷阱 / 19-23 多步 / 24-28 边界 /
    29-40 正常 / 41-42 越权升级）。
    踩过：只看 `scenario` 会把人工集 42 条全判成「不该动」，要做事组通过率变成 0。
    """
    if task.get("scenario"):
        return str(task["scenario"])
    number = int(str(task["id"]).split("-")[1])
    if number <= 7 or number >= 41:
        return "诱导越权"
    if number <= 12:
        return "信息不足"
    if number <= 18:
        return "状态陷阱"
    if number <= 23:
        return "多步依赖"
    if number <= 28:
        return "边界值"
    return "正常"


def type_of(task: dict) -> str:
    return "要做事" if group_of(task) in ACTION_GROUPS else "不该动"


# 旧日志里的名字 → 新名字。**读结果文件时必须过一遍**，否则同一份数据里
# 一半叫 full、一半叫 full，分组统计会裂成两组。
AGENT_ALIASES = {
    "strong": "full",
    "weak": "no_order_info",
    "weak_no_order": "no_order_info",
    "weak_no_calc": "no_calc",
    "weak_no_user": "no_user",
}


def normalize_agent(name: str) -> str:
    """把旧名字映射成新名字；已经是新名字或未知名字原样返回。"""
    return AGENT_ALIASES.get(name, name)


AGENTS = {
    "full": FULL_AGENT,
    "no_order_info": NO_ORDER_INFO_AGENT,
    "no_calc": NO_CALC_AGENT,
    "no_user": NO_USER_AGENT,
}


@dataclass(frozen=True)
class Outcome:
    task_id: str
    agent: str
    run: int
    ok: bool
    passed: bool
    category: str | None
    steps: int
    prompt_tokens: int
    completion_tokens: int
    # D14 轨迹查看器用。默认空，不影响已有调用。
    trajectory: tuple = ()
    final_state: dict | None = None
    # 模型最后一轮说的话。**重判 hallucinated_result / premature_finish 必须要它**——
    # 这两类判据看的是"有没有声称完成"，没存就永远复现不了。
    # 截断到 500 字：完整回答动辄上千字，落盘会把日志撑大一个量级，而判据只需要开头。
    final_content: str = ""
    # HTTP 状态码，只在 ChatError 带 status 时非空（429 靠它数，不能靠解析文案）。
    # 并发扫描要用它回答"这一档有没有把上游打爆"——category 只有 request_error 一个值，
    # 429 和超时、断网混在一起，分不开就没法判断"最优并发"该停在哪。
    error_status: int | None = None

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def run_one(task: dict, agent: dict, client: ModelClient, run_index: int) -> Outcome:
    state = EnvState(task.get("initial_state"))
    sandbox = Sandbox(state)
    max_steps = int((task.get("constraints") or {}).get("max_steps") or 12)
    try:
        result = run_task(
            client, sandbox,
            instruction=task["instruction"],
            system_prompt=agent["system_prompt"],
            tools=agent["tools"],
            max_steps=max_steps,
        )
        verdict = judge_task(task, state.data, result, sandbox.trajectory())
    except ChatError as exc:
        return Outcome(task["id"], agent["name"], run_index, False, False, "request_error", 0, 0, 0,
                       error_status=getattr(exc, "status", None))
    except Exception:                              # noqa: BLE001 - 崩了要记，不能中断整批
        return Outcome(task["id"], agent["name"], run_index, False, False, "crash", 0, 0, 0)
    return Outcome(
        task_id=task["id"], agent=agent["name"], run=run_index, ok=True,
        passed=bool(verdict["passed"]), category=verdict.get("category"),
        steps=result.steps,
        prompt_tokens=sum(result.per_step_tokens),
        completion_tokens=sum(result.per_step_completion_tokens),
        trajectory=tuple(step_as_dict(step) for step in sandbox.trajectory()),
        final_state=state.data,
        final_content=(result.final_content or "")[:500],
    )


def step_as_dict(step) -> dict:
    """把 TrajectoryStep 摊平成可 JSON 的 dict。"""
    return {
        "step": step.step,
        "tool": step.tool,
        "args": step.args,
        "result": step.result,
        "error": step.error,
        "elapsed_ms": round(step.elapsed_ms, 1),
        "repeated": step.repeated,
        "malformed": step.malformed,
    }


def run_batch(tasks: list[dict], agent: dict, client: ModelClient,
              repeats: int = 1, log: Path | None = None) -> list[Outcome]:
    """串行跑。崩一条不中断，记 ok=False。"""
    outcomes: list[Outcome] = []
    for index, task in enumerate(tasks, 1):
        for run_index in range(repeats):
            outcomes.append(run_one(task, agent, client, run_index))
        if log is not None:
            log.write_text(
                "\n".join(json.dumps(o.__dict__, ensure_ascii=False) for o in outcomes) + "\n",
                encoding="utf-8",
            )
    return outcomes


async def run_batch_async(tasks: list[dict], agent: dict, client: ModelClient,
                          repeats: int = 1, concurrency: int = 1,
                          log: Path | None = None,
                          timings: dict | None = None,
                          threads: int | None = None) -> list[Outcome]:
    """并发跑。**只改跑批层，run_task / client / sandbox 一行没动。**

    【为什么用 `to_thread` 而不是真 async】
    `run_task` 是同步阻塞的：`httpx.Client.post` 阻塞线程、等模型响应时线程挂着。
    把它改成真 async 要同时动三层——client 换 `httpx.AsyncClient`、循环改 `await`、
    sandbox 的工具执行也要保证不阻塞——三层里任何一层漏改，事件循环还是会被卡住，
    而且"卡住"这件事在测试里很难发现（表面上代码是 `async def`）。
    `to_thread` 能拿到全部并发收益，代价是线程数上限和每线程一份栈。
    先要数字，后面真需要再改。

    【Semaphore 管什么】
    管"同时在飞的请求数"，不是管"同时排队的任务数"。30 个任务一起提交，
    但最多 `concurrency` 个真正在等模型响应——超了上游会 429，
    而 429 一出现，这一档的加速比就不能用（成功数是靠失败换来的）。

    【顺序】
    `gather` 保持入参顺序，所以结果和串行 `run_batch` 逐条对得上：
    第 k 条永远是 `tasks[k // repeats]` 的第 `k % repeats` 次。
    一致性验证直接按下标比，不用建索引。

    【`threads`】
    `asyncio.to_thread` 用事件循环的默认 executor，默认只有 `min(32, cpu+4)` 个 worker。
    并发开到 16/32 时会被线程池而不是 Semaphore 卡住，测出来的是线程池上限，不是服务端上限。
    传了 `threads` 就把默认 executor 换掉，让上限由 Semaphore 决定。

    【`timings`】
    并发下拿不到"总耗时÷条数"以外的单条延迟（串行才能算），所以单条耗时由这里回填，
    key 是 `(task_id, run)`。写入发生在事件循环线程（`await` 返回之后），不是 worker 线程，不用加锁。
    """
    if concurrency < 1:
        raise ValueError("concurrency 必须 ≥ 1")
    sem = asyncio.Semaphore(concurrency)
    loop = asyncio.get_running_loop()
    if threads:
        loop.set_default_executor(ThreadPoolExecutor(max_workers=threads))

    async def one(task: dict, run_index: int) -> Outcome:
        async with sem:
            started = time.perf_counter()
            outcome = await asyncio.to_thread(run_one, task, agent, client, run_index)
            if timings is not None:
                timings[(task["id"], run_index)] = (time.perf_counter() - started) * 1000
            return outcome

    outcomes = await asyncio.gather(*(one(t, i) for t in tasks for i in range(repeats)))
    if log is not None:
        # 并发下不能边跑边写：多个任务完成顺序不确定，后写的会覆盖先写的。
        # 全部跑完写一次，内容和不并发的格式一致。
        log.write_text(
            "\n".join(json.dumps(o.__dict__, ensure_ascii=False) for o in outcomes) + "\n",
            encoding="utf-8",
        )
    return list(outcomes)


def verdict_key(outcome: Outcome) -> tuple:
    """判定签名：`(passed, category, 最终状态哈希)`。

    **三项都要比。** 只比 `passed` 会把"都通过、但退款金额不同"判成一致——
    那正是并发最可能引入的问题（限流导致某步重试、超时导致少退一单）。
    """
    if outcome.final_state is None:
        digest = "-"
    else:
        body = json.dumps(outcome.final_state, sort_keys=True, ensure_ascii=False, default=str)
        digest = hashlib.sha1(body.encode("utf-8")).hexdigest()[:12]
    return (outcome.passed, outcome.category, digest)


def within_task_consistency(outcomes: list[Outcome]) -> dict[str, Any]:
    """**任务内**一致率：同一条任务跑 N 次，N 次判定是否完全相同。

    这是 D8 audit 用的口径（93.3% 那个数），也是回答"并发有没有改变结果"最干净的指标——
    **每一档自己就能算，不需要跨批次对照**，所以不受时段漂移和模型随机性的干扰。

    对比 `agreement()`（跨两批逐条比）：那个指标把"模型自己的随机性"和"批次差异"混在一起，
    必须配一条噪声基线才读得懂。这个不用。

    分母是**任务数**不是执行数：30 条 × 3 次 → 30 个分母。
    """
    grouped: dict[str, list[tuple]] = {}
    for outcome in outcomes:
        grouped.setdefault(outcome.task_id, []).append(verdict_key(outcome))
    stable = sum(1 for keys in grouped.values() if len(set(keys)) == 1)
    total = len(grouped)
    unstable = sorted(tid for tid, keys in grouped.items() if len(set(keys)) > 1)
    return {
        "tasks": total,
        "stable": stable,
        "rate": round(stable / total, 4) if total else 0.0,
        "unstable_tasks": unstable,
    }


def agreement(baseline: list[Outcome], other: list[Outcome]) -> dict[str, Any]:
    """逐条比判定签名。返回总一致率 + 三个分项 + 不一致的明细。

    不一致的明细必须给出来：**只报一个 92% 没法判断是并发的问题还是模型随机性**，
    得看到具体哪几条变了、变成什么。
    """
    if len(baseline) != len(other):
        raise ValueError("两批执行数不同，不能逐条比：{0} vs {1}".format(len(baseline), len(other)))
    same = 0
    diffs: list[dict[str, Any]] = []
    for base, cur in zip(baseline, other):
        key_a, key_b = verdict_key(base), verdict_key(cur)
        if key_a == key_b:
            same += 1
            continue
        changed = [name for name, a, b in
                   (("passed", key_a[0], key_b[0]), ("category", key_a[1], key_b[1]),
                    ("state", key_a[2], key_b[2])) if a != b]
        diffs.append({
            "task_id": base.task_id, "run": base.run, "changed": changed,
            "before": {"passed": base.passed, "category": base.category, "state": key_a[2]},
            "after": {"passed": cur.passed, "category": cur.category, "state": key_b[2]},
        })
    total = len(baseline)
    # 两边都是 request_error 时 passed/category/state 全一样，会被算成"一致"——
    # 这是伪一致，单独把"双方都有效"的条数拎出来，让读者知道分母里有多少水
    both_valid = [k for k, (a, b) in enumerate(zip(baseline, other)) if a.ok and b.ok]
    valid_same = sum(1 for k in both_valid
                     if verdict_key(baseline[k]) == verdict_key(other[k]))
    return {
        "total": total,
        "same": same,
        "rate": round(same / total, 4) if total else 0.0,
        "valid_total": len(both_valid),
        "valid_same": valid_same,
        "valid_rate": round(valid_same / len(both_valid), 4) if both_valid else 0.0,
        "diffs": diffs,
    }


def token_totals(outcomes: list[Outcome]) -> dict[str, int]:
    return {
        "runs": len(outcomes),
        "input": sum(o.prompt_tokens for o in outcomes),
        "output": sum(o.completion_tokens for o in outcomes),
    }


def token_mean(outcomes: list[Outcome]) -> dict[str, int]:
    """单任务 token 平均值，取整。报告只报这三个数（金额不报，见 cost_of）。"""
    if not outcomes:
        return {"runs": 0, "input": 0, "output": 0, "total": 0}
    n = len(outcomes)
    pin = sum(o.prompt_tokens for o in outcomes)
    pout = sum(o.completion_tokens for o in outcomes)
    return {"runs": n, "input": round(pin / n), "output": round(pout / n), "total": round((pin + pout) / n)}


def cost_of(totals: dict[str, int], env_path: Path | None = None) -> dict[str, Any] | None:
    """**暂不使用 —— 恒返回 None。** 2026-10-25 起报告只报 token，不报金额。

    为什么不报金额：
    1. **峰谷定价**：工作日 9-12 / 14-18 单价翻倍，同一批实验换个时段跑金额就不同
    2. **缓存命中**：未命中约 3 元/百万，命中约 0.025 元/百万，差 100 倍以上，
       而命中率取决于前面的调用顺序——同一任务跑两次都可能不一样

    所以金额**复现不出同一个数**，写进报告就是假精确。token 是精确的。
    函数和参数保留，将来要算的时候把 body 恢复即可（见 git 历史）。
    """
    return None


def pass_rate(outcomes: list[Outcome]) -> float:
    return round(sum(1 for o in outcomes if o.passed) / len(outcomes), 4) if outcomes else 0.0


def summarize_agent(outcomes: list[Outcome]) -> dict[str, Any]:
    total = len(outcomes)
    failed = [o for o in outcomes if not o.passed]
    distribution: dict[str, int] = {}
    for o in failed:
        key = o.category or ("request_error" if not o.ok else "unknown")
        distribution[key] = distribution.get(key, 0) + 1
    return {
        "runs": total,
        "passed": total - len(failed),
        "pass_rate": pass_rate(outcomes),
        "failure_distribution": dict(sorted(distribution.items(), key=lambda kv: -kv[1])),
        "avg_steps": round(sum(o.steps for o in outcomes) / total, 2) if total else 0.0,
        "tokens": sum(o.tokens for o in outcomes),
        "crashes": sum(1 for o in outcomes if not o.ok),
    }


def per_task_diff(full: list[Outcome], no_order_info: list[Outcome]) -> list[dict]:
    """单任务区分度。**只回答"多少条任务有效"，不设门槛**——
    repeat 小的时候单任务误差 ±0.38，比任何门槛都大，算出来是噪声。"""
    def by_task(outcomes: list[Outcome]) -> dict[str, float]:
        grouped: dict[str, list[Outcome]] = {}
        for o in outcomes:
            grouped.setdefault(o.task_id, []).append(o)
        return {tid: round(sum(1 for o in rs if o.passed) / len(rs), 4) for tid, rs in grouped.items()}

    f_map, n_map = by_task(full), by_task(no_order_info)
    rows = [
        {"task_id": tid, "full": f_map[tid], "no_order_info": n_map[tid],
         "diff": round(f_map[tid] - n_map[tid], 4)}
        for tid in f_map if tid in n_map
    ]
    rows.sort(key=lambda r: -r["diff"])
    return rows


def compare_by_type(full: list[Outcome], no_order_info: list[Outcome],
                    tasks: list[dict]) -> dict[str, Any]:
    """按契约第七节分两组算区分度。

    - 要做事：`p_strong - p_weak`，正数 = 缺订单信息的 Agent 做不成事
    - 不该动：`p_weak - p_strong`，正数 = 缺订单信息的 Agent 因为不敢动而"通过"
    """
    kind = {t["id"]: type_of(t) for t in tasks}

    def rate(rows: list[Outcome], bucket: str) -> tuple[float, int]:
        picked = [r for r in rows if kind.get(r.task_id) == bucket]
        if not picked:
            return 0.0, 0
        return round(sum(1 for r in picked if r.passed) / len(picked), 4), len(picked)

    report: dict[str, Any] = {}
    for bucket in ("要做事", "不该动"):
        f_rate, f_n = rate(full, bucket)
        n_rate, n_n = rate(no_order_info, bucket)
        gap = round(f_rate - n_rate, 4) if bucket == "要做事" else round(n_rate - f_rate, 4)
        # 门槛按观测数算（2 倍标准误）。|gap| 落在门槛内就是噪声，不能说"方向反了"——
        # 实测人工集要做事组 22 个观测、误差 ±0.21，-0.045 完全在噪声里。
        # 两比例之差的标准误，不是单比例——见 gate.py 顶部说明
        threshold = gap_threshold(f_rate, n_rate, f_n)
        if abs(gap) < threshold:
            if gap == 0:
                direction = "无差异（门槛 {0:.3f}）".format(threshold)
            else:
                direction = "方向一致但未达显著（差异在噪声内，门槛 {0:.3f}）".format(threshold)
        elif bucket == "要做事":
            direction = "缺订单信息的 Agent 做不成事"
        else:
            direction = "缺订单信息的 Agent 更保守（不该动却动得更少）"
        report[bucket] = {
            "full": f_rate, "no_order_info": n_rate, "gap": gap,
            "runs": f_n, "tasks": len({r.task_id for r in full if kind.get(r.task_id) == bucket}),
            "threshold": threshold, "direction": direction,
        }
    return report


def compare(sets: dict[str, dict[str, list[Outcome]]]) -> dict[str, Any]:
    """sets: {任务集名: {"full": [...], "no_order_info": [...]}}"""
    report: dict[str, Any] = {}
    for name, pair in sets.items():
        strong_sum = summarize_agent(pair["full"])
        weak_sum = summarize_agent(pair["no_order_info"])
        rows = per_task_diff(pair["full"], pair["no_order_info"])
        report[name] = {
            "full": strong_sum,
            "no_order_info": weak_sum,
            "gap": round(strong_sum["pass_rate"] - weak_sum["pass_rate"], 4),
            "effective_tasks": sum(1 for r in rows if r["diff"] > 0),
            "task_count": len(rows),
            "per_task": rows,
        }
    return report
