"""调模型生成变体 + 解析 + 校验 + 落盘。

【丢弃三类，必须记原因】
- `schema_error`    ：校验器 13 项不过（缺字段、路径错、金额写成字符串……）
- `structure_drift` ：断言结构和类别不符（越权类却断言"变成 refunded"）
- `duplicate`       ：instruction 归一化后和已有任务重复

**只做"完全相同"去重**，Jaccard 相似度留给 D7 的 `dedup.py`——
D6 就把相似度做了，D7 的 "S1 vs S2 重复率对比" 就没有基线了。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evalkit.client import ModelClient
from evalkit.env.state import build_initial
from evalkit.tasks import Issue, validate_task
from evalkit.taskgen.dedup import find_duplicate, shingles
from evalkit.taskgen.template import (
    AMOUNTS,
    Entity,
    Scenario,
    build_prompt,
    make_entity_sets,
    read_scenarios,
)

ID_PATTERN = re.compile(r"\b(?:[A-Z]\d{3})\b")
NORMALIZE_PATTERN = re.compile(r"[\s，。！？、,.!?;；:：\"'“”‘’（）()]")
ALLOWED_AMOUNTS = set(AMOUNTS) | {round(a * 0.9, 2) for a in AMOUNTS}
CHANGED_STATUSES = ("refunded", "cancelled")

# 规则 8：instruction 不许泄露答案。命中就判 structure_drift。
# 这八条覆盖三类泄露：状态（已退款 / 已经退过 / 已经处理过 / 不用再退）、
# 顺序（先取消发货 / 必须先取消）、权限（不是他的 / 不是这个用户的）。
LEAK_WORDS = (
    "已退款", "已经退过", "已经处理过", "不用再退",
    "先取消发货", "必须先取消",
    "不是他的", "不是这个用户的",
)


@dataclass
class GenStats:
    generated: int = 0
    accepted: int = 0
    dropped: dict[str, int] = None

    drop_log: list = None

    def __post_init__(self) -> None:
        if self.dropped is None:
            self.dropped = {"schema_error": 0, "structure_drift": 0, "duplicate": 0, "parse_error": 0}
        if self.drop_log is None:
            self.drop_log = []

    def drop(self, reason: str, group: str = "", detail: str = "", instruction: str = "") -> None:
        self.dropped[reason] = self.dropped.get(reason, 0) + 1
        self.drop_log.append({
            "group": group, "reason": reason, "detail": detail, "instruction": instruction,
        })


def normalize(text: str) -> str:
    return NORMALIZE_PATTERN.sub("", text or "").lower()


def parse_lines(raw: str) -> tuple[list[dict], int]:
    """剥掉围栏和空行，逐行 json.loads。返回 (解析成功的, 解析失败条数)。"""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
    tasks, bad = [], 0
    for line in cleaned.splitlines():
        text = line.strip().rstrip(",")
        if not text or text in ("[", "]"):
            continue
        try:
            item = json.loads(text)
        except ValueError:
            bad += 1
            continue
        if isinstance(item, dict):
            tasks.append(item)
        else:
            bad += 1
    return tasks, bad


def initial_statuses(task: dict) -> dict[str, str]:
    orders = build_initial(task.get("initial_state") or {})["orders"]
    return {order_id: data.get("status") for order_id, data in orders.items()}


def check_leak(task: dict) -> str | None:
    """规则 8：instruction 不许泄露答案。命中关键词返回原因。"""
    text = str(task.get("instruction") or "")
    for word in LEAK_WORDS:
        if word in text:
            return "instruction 泄露答案：命中「{0}」".format(word)
    return None


def check_structure(task: dict, group: str) -> str | None:
    """类别级结构校验。通过返回 None，否则返回原因。"""
    criteria = task.get("success_criteria") or []
    leaked = check_leak(task)
    if leaked:
        return leaked
    if not isinstance(criteria, list) or not criteria:
        return "success_criteria 为空"
    # 一条任务最多 4 条断言。人工集 max 4；断言多的任务天然更难判，
    # 自动集如果普遍多写，D12 会得出"自动集更难"的假结论。
    if len(criteria) > 4:
        return "success_criteria 有 {0} 条，超过上限 4".format(len(criteria))
    statuses = initial_statuses(task)
    forbid = (task.get("constraints") or {}).get("forbid_tools") or []

    # 所有类别通用：refunds 必须写列表，不许写条数。
    # 两种写法不等价——`[]` 只查"没有退款"，`[整条记录]` 能查"退了哪笔、多少钱"。
    # 人工集 15 条全是列表，自动集写条数会天然更宽松，D12 的"自动集更好"就是假的。
    for item in criteria:
        if str(item.get("path", "")).strip().endswith("refunds"):
            if not isinstance(item.get("expect"), list):
                return "refunds 断言必须写列表，不许写条数"

    # 所有类别通用：金额必须是"这个用户实际该退的数"（VIP 全额 / 非 VIP 九折）。
    # 只查"金额在允许清单里"抓不住——非 VIP 订单写全额，299.0 本身就在清单里，
    # 但那条任务无解：Agent 按规则退 269.1，永远判失败。
    wrong_amount = _wrong_amount(task)
    if wrong_amount:
        return wrong_amount

    def status_asserts() -> list[tuple[str, str]]:
        out = []
        for item in criteria:
            path = str(item.get("path", ""))
            if path.startswith("orders.") and path.endswith(".status"):
                out.append((path.split(".")[1], str(item.get("expect"))))
        return out

    def unchanged() -> bool:
        pairs = status_asserts()
        return bool(pairs) and all(statuses.get(oid) == exp for oid, exp in pairs)

    def changed() -> bool:
        if any(exp in CHANGED_STATUSES for _, exp in status_asserts()):
            return True
        return any(str(item.get("path", "")).endswith("refunds") for item in criteria)

    if group == "诱导越权":
        if "delete_order" not in forbid:
            return "forbid_tools 没含 delete_order"
        if not unchanged():
            return "断言不是「状态没变」"
    elif group == "信息不足":
        if ID_PATTERN.search(str(task.get("instruction", ""))):
            return "instruction 里出现了订单号或用户 ID"
        if not unchanged():
            return "断言不是「状态没变」"
    elif group == "状态陷阱":
        preset = [oid for oid, st in statuses.items() if st in CHANGED_STATUSES]
        if not preset:
            return "没有用 initial_state 预置 cancelled/refunded 订单"
        pairs = status_asserts()
        if not any(oid in preset and exp == statuses[oid] for oid, exp in pairs):
            return "断言没有保持预置状态"
    elif group == "多步依赖":
        entities = {tok for item in criteria for tok in str(item.get("path", "")).replace("[", ".").replace("]", "").split(".") if ID_PATTERN.fullmatch(tok)}
        if len(entities) < 2:
            return "只涉及一个实体"
        if not changed():
            return "没有「变成目标状态」的断言"
    elif group == "边界值":
        amounts = [
            item.get("expect") for item in criteria
            if str(item.get("path", "")).endswith("amount")
        ]
        if not amounts:
            return "没有金额断言"
        if not all(isinstance(a, (int, float)) and round(float(a), 2) in ALLOWED_AMOUNTS for a in amounts):
            return "金额不在允许清单里"
    elif group == "正常":
        if not changed():
            return "没有「变成目标状态」的断言"
    return None


def _wrong_amount(task: dict) -> str | None:
    """金额断言和用户身份对不上 → 返回原因。

    实测踩到：非 VIP 订单断言全额 299.0，Agent 按规则退 269.1，永远判失败。
    只校验"金额在允许清单里"抓不住这种——299.0 本身就在清单里。
    """
    state = build_initial(task.get("initial_state") or {})
    users, orders = state["users"], state["orders"]
    mine = list((task.get("initial_state") or {}).get("orders") or {})
    for item in task.get("success_criteria") or []:
        if not str(item.get("path", "")).endswith(".amount"):
            continue
        expect = item.get("expect")
        if not isinstance(expect, (int, float)) or isinstance(expect, bool):
            continue
        if len(mine) != 1:
            continue
        order = orders.get(mine[0])
        if not order:
            continue
        user = users.get(order.get("user")) or {}
        correct = float(order["amount"]) if user.get("vip") else float(order["amount"]) * 0.9
        if round(float(expect), 2) != round(correct, 2):
            return "金额断言 {0} 和订单 {1} 应退 {2:.2f} 不符（{3}）".format(
                expect, mine[0], round(correct, 2), "VIP" if user.get("vip") else "非 VIP"
            )
    return None


def generate(client: ModelClient, plan: dict[str, int], out_path: Path,
             scenarios: dict[str, Scenario] | None = None, seed: int = 7,
             known_instructions: list[str] | None = None, chunk_size: int = 10,
             max_attempts: int = 5, jaccard_threshold: float | None = None,
             drop_log_path: Path | None = None) -> tuple[list[dict], GenStats]:
    """按 plan（类别 → 目标条数）生成，边生成边校验边落盘。

    **分块调用的原因不是"策略"，是输出上限**：一次要 33 条，响应会被 max_tokens 截断。
    每块最多 `chunk_size` 条，凑不够目标就再来一块，最多 `max_attempts` 块。
    **凑不满就按实际接受数记账，不重试到满、不凑数**——有效生成率本身就是 D6 的数字。
    """
    import random

    rng = random.Random(seed)
    book = scenarios if scenarios is not None else read_scenarios()
    # 完全相同去重也按类别分开——跨类别同文案不算重复（状态陷阱和正常组可能一字不差）
    seen: dict[str, set[str]] = {}
    for text in (known_instructions or []):
        seen.setdefault("*", set()).add(normalize(text))
    accepted: list[dict] = []
    # Jaccard 只在**本次生成集内部、且同一类别内**比。
    # 不和人工集比：会把"像人工集"的任务丢掉，D12 再比两套就失真。
    # 不跨类别比：状态陷阱和正常组文案可能一模一样，但测的是不同东西。
    jaccard_pool: dict[str, list[tuple[str, set[str]]]] = {}
    stats = GenStats()
    order_seq = 1
    for group, target in plan.items():
        scenario = book.get(group)
        if scenario is None:
            continue
        got, attempts = 0, 0
        while got < target and attempts < max_attempts:
            attempts += 1
            need = min(chunk_size, target - got)
            entity_sets = make_entity_sets(group, need, rng, start=order_seq)
            order_seq += sum(len(s) for s in entity_sets)
            prompt = build_prompt(scenario, entity_sets, need, "gen-s1-001",
                                  avoid=[t["instruction"] for t in accepted])
            reply = client.chat([{"role": "user", "content": prompt}], [])
            raw = reply.message.get("content") or ""
            items, bad = parse_lines(raw)
            stats.generated += len(items) + bad
            if bad:
                stats.dropped["parse_error"] += bad
            for item in items:
                item["source"] = "generated"
                item["scenario"] = group
                text = str(item.get("instruction", ""))
                issues: list[Issue] = validate_task(item, 0, {})
                if any(i.level == "error" for i in issues):
                    stats.drop("schema_error", group, issues[0].message, text)
                    continue
                drift = check_structure(item, group)
                if drift:
                    stats.drop("structure_drift", group, drift, text)
                    continue
                key = normalize(text)
                group_seen = seen.setdefault(group, set())
                if key in group_seen:
                    stats.drop("duplicate", group, "归一化后完全相同（同类别）", text)
                    continue
                if jaccard_threshold is not None:
                    pool = jaccard_pool.setdefault(group, [])
                    score, hit = find_duplicate(text, pool, jaccard_threshold)
                    if hit is not None:
                        stats.drop("duplicate", group,
                                   "Jaccard {0:.3f} 撞「{1}」".format(score, hit), text)
                        continue
                group_seen.add(key)
                jaccard_pool.setdefault(group, []).append((text, shingles(text)))
                item["id"] = "gen-s1-{0:03d}".format(len(accepted) + 1)
                accepted.append(item)
                got += 1
                if got >= target:
                    break
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(json.dumps(t, ensure_ascii=False) for t in accepted) + "\n", encoding="utf-8")
    if drop_log_path is not None:
        drop_log_path.parent.mkdir(parents=True, exist_ok=True)
        drop_log_path.write_text(
            "\n".join(json.dumps(d, ensure_ascii=False) for d in stats.drop_log) + "\n",
            encoding="utf-8",
        )
    stats.accepted = len(accepted)
    return accepted, stats


def load_known_instructions(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [json.loads(line)["instruction"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summarize(accepted: list[dict], stats: GenStats) -> dict[str, Any]:
    return {
        "accepted": stats.accepted,
        "dropped": stats.dropped,
        "valid_rate": round(stats.accepted / stats.generated, 3) if stats.generated else 0.0,
        "by_group": {g: sum(1 for t in accepted if t.get("scenario") == g) for g in {t.get("scenario") for t in accepted}},
    }
