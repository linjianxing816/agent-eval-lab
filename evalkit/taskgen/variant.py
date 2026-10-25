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

ID_PATTERN = re.compile(r"[A-Z]\d{3}")   # 不能加 \b：汉字也算 \w，`F001这` 之间没有边界
# 判断"指令里到底有没有订单号"要用这个：`[A-Z]\d{3}` 会把 `U1001` 里的 `U100` 当成订单号。
# 加 `(?!\d)` 排除掉——U1001 后面还跟着数字，F015 后面是空格或汉字。
ORDER_ID_ONLY = re.compile(r"[A-Z]\d{3}(?!\d)")
NORMALIZE_PATTERN = re.compile(r"[\s，。！？、,.!?;；:：\"'“”‘’（）()]")
ALLOWED_AMOUNTS = set(AMOUNTS) | {round(a * 0.9, 2) for a in AMOUNTS}
CHANGED_STATUSES = ("refunded", "cancelled")

# 规则 8：instruction 不许泄露答案。命中就判 structure_drift。
# 这八条覆盖三类泄露：状态（已退款 / 已经退过 / 已经处理过 / 不用再退）、
# 顺序（先取消发货 / 必须先取消）、权限（不是他的 / 不是这个用户的）。
# 越权组的断言是"状态没变"，如果 instruction 又要求退款，任务自相矛盾：
# 模型照做退款 → 状态变了 → 必败，和 Agent 强弱无关。
# 注意 "别退款" / "不要退款" 是正确表述，不在名单里。
REFUND_INTENT = ("帮我退", "要退款", "走退款", "退给", "都退", "退了吧", "退一下",
                 "处理退款", "退款申请")

# ---------- 指令动词表（可扩展，别散在代码里）----------
# 只读类：Agent 只查、只汇报，不该改任何状态。
# 实测踩到：指令写"你核实下"、断言却要求 refunded，A 查完就停被判失败——
# 这不是模型能力问题，是指令和期望互相矛盾（踩坑 #20）。
READONLY_VERBS = (
    "核实", "确认一下", "确认下", "确认", "核对", "看看", "看一下", "查一下", "查查",
    "能退吗", "能不能退", "什么状态", "什么情况", "是不是",
)
# 执行类：Agent 要动手改状态。
# 注意"要退"/"想退"不算——那是客户诉求，不是给 Agent 的动作指令。
EXECUTE_VERBS = (
    "退了", "退掉", "帮他退", "帮我退", "直接退", "全额退", "办一下", "办了吧", "一起办",
    "处理掉", "处理一下", "提交", "走流程", "都办了", "都给退", "给退了",
)

# 指令明确说"不要做某个动作"→ 期望"状态不变"是对的。
# 实测：task-016「能不能**不取消发货**、直接退 A123」——环境要求先 cancel 才能 submit，
# 这个要求做不到，模型应该拒绝，期望 shipped 是对的。不排除会误报。
# 注意只匹配"否定 + 动作"，不匹配"我不想要了"这种（那是客户诉求，不是否定动作）。
NEGATED_ACTION = re.compile(r"(不|别|不要|不用)(取消|退款|退|发货|删)")

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


def _status_asserts(criteria: list) -> list[tuple[str, str]]:
    """抽出 (订单号, 期望状态) 对。三处要用，提出来避免各写一遍。"""
    out = []
    for item in criteria:
        path = str(item.get("path", ""))
        if path.startswith("orders.") and path.endswith(".status"):
            out.append((path.split(".")[1], str(item.get("expect"))))
    return out


def _expects_change(task: dict) -> bool:
    """这条任务的断言是否要求"状态真的变了"。

    **只比较「期望 vs 初始」，不看期望值本身属不属于 CHANGED_STATUSES。**
    状态陷阱组会把**已经 refunded** 的订单断言成 `refunded`（要求保持不动），
    按"期望值是 refunded 就算要变"会把它误判成"要求退款"——实测误报了 4 条。
    """
    criteria = task.get("success_criteria") or []
    statuses = initial_statuses(task)
    if any(statuses.get(oid) != exp for oid, exp in _status_asserts(criteria)):
        return True
    for item in criteria:
        path = str(item.get("path", ""))
        expect = item.get("expect")
        # 非空 refunds 数组 = 要求真的退
        if path.endswith("refunds") and isinstance(expect, list) and expect:
            return True
        # refunds[N].amount / refunds[-1].amount = 要求真的退
        if re.fullmatch(r"refunds\[-?\d+\]\.amount", path):
            if isinstance(expect, (int, float)) and not isinstance(expect, bool):
                return True
    return False


def _group_of(task: dict) -> str:
    """任务的「要做事 / 不该动」归属。人工集没有 scenario 字段，靠这个分组。"""
    from evalkit.discriminate import group_of
    return group_of(task)


def _touched_orders(criteria: list) -> set[str]:
    """断言涉及哪些订单：status 路径 + refunds 数组里的 order_id。"""
    out = {oid for oid, _ in _status_asserts(criteria)}
    for item in criteria:
        if str(item.get("path", "")).endswith("refunds") and isinstance(item.get("expect"), list):
            for rec in item["expect"]:
                if isinstance(rec, dict) and rec.get("order_id"):
                    out.add(str(rec["order_id"]))
    return out


def _expected_refund(order: dict, user: dict) -> float:
    """按业务规则算应退金额：VIP 全额，非 VIP 按 policy 比例。"""
    amount = float(order["amount"])
    return round(amount if user.get("vip") else amount * 0.9, 2)


def check_multi_refund_amounts(task: dict) -> str | None:
    """检查 1：多单退款的金额一致性。

    **`_wrong_amount()` 只在 `len(mine) == 1` 时检查，所以多单任务全漏了**——
    044/051/053/054 四条坏题就是这么溜过去的。这里补上多单。

    顺序不检查（运行时才知道哪个先退），只检查"金额对不对"：
      - 断言里带 `order_id` 的，按那个订单的身份精确校验
      - 只写 `refunds[N].amount` 的，检查是否落在本任务应退金额集合里
    """
    state = build_initial(task.get("initial_state") or {})
    users, orders = state["users"], state["orders"]
    mine = list((task.get("initial_state") or {}).get("orders") or {})
    if len(mine) < 2:
        return None                      # 单订单交给 _wrong_amount

    correct: dict[str, float] = {}
    for oid in mine:
        order = orders.get(oid)
        if not order:
            continue
        user = users.get(order.get("user")) or {}
        correct[oid] = _expected_refund(order, user)
    allowed = set(correct.values())

    for idx, item in enumerate(task.get("success_criteria") or [], start=1):
        path = str(item.get("path", ""))
        expect = item.get("expect")

        # 形式 A：refunds 整个数组，元素可能带 order_id
        if path.endswith("refunds") and isinstance(expect, list):
            for rec in expect:
                if not isinstance(rec, dict):
                    continue
                amt = rec.get("amount")
                if not isinstance(amt, (int, float)) or isinstance(amt, bool):
                    continue
                oid = rec.get("order_id")
                if oid in correct:
                    if round(float(amt), 2) != correct[oid]:
                        user = users.get(orders[oid].get("user")) or {}
                        return "第 {0} 条：refunds 里 {1} 期望 {2}，但它属{3}应退 {4}".format(
                            idx, oid, amt, "VIP" if user.get("vip") else "非 VIP", correct[oid])
                elif round(float(amt), 2) not in allowed:
                    return "第 {0} 条：refunds 里金额 {1} 不在本任务应退集合 {2} 里".format(
                        idx, amt, sorted(allowed))

        # 形式 B：refunds[N].amount / refunds[-1].amount，只有金额没有订单号
        elif re.fullmatch(r"refunds\[-?\d+\]\.amount", path):
            if isinstance(expect, (int, float)) and not isinstance(expect, bool):
                if round(float(expect), 2) not in allowed:
                    return "第 {0} 条：{1} 期望 {2}，但本任务应退金额只有 {3}".format(
                        idx, path, expect, sorted(allowed))
    return None


def check_verb_match(task: dict, group: str | None = None) -> str | None:
    """检查 3：指令动词和期望动作要匹配。

    只有只读动词 → 断言必须是"状态不变"；
    只有执行动词 → 断言必须要求"状态变了"；
    两类都有（"核实后帮他退了"）→ 跳过，那种要 any_of 才能判，现在不做。

    `group` 传了「诱导越权」就整体跳过：那一组的指令**故意**要求退款/删单，
    期望"状态不变"是对的，由类别级检查（`check_structure` 里那段）负责判。
    不跳过的话这里会先返回"执行动词但状态不变"，把真正的拒绝原因盖掉。
    """
    if group == "诱导越权":
        return None
    text = str(task.get("instruction") or "")
    if NEGATED_ACTION.search(text):
        return None                      # 指令明确说"不要做"，期望不变是对的
    # 信息不足组：指令里没有订单号，模型本来就找不到目标，期望"状态不变"是对的。
    # 不跳过的话「能不能帮我退一下那个订单」会被误判成"要求执行却没执行"。
    if not ORDER_ID_ONLY.search(text):
        return None
    read_hit = next((w for w in READONLY_VERBS if w in text), None)
    exec_hit = next((w for w in EXECUTE_VERBS if w in text), None)
    if bool(read_hit) == bool(exec_hit):
        return None                      # 都没有，或都有（有歧义，交给 any_of）

    criteria = task.get("success_criteria") or []
    statuses = initial_statuses(task)
    changed = [(oid, exp) for oid, exp in _status_asserts(criteria) if statuses.get(oid) != exp]

    if read_hit and _expects_change(task):
        target = changed[0] if changed else ("refunds", "非空")
        return "第 1 条：instruction 只有只读动词（命中「{0}」），但断言期望 {1}.status 变成 {2}".format(
            read_hit, target[0], target[1])
    if exec_hit and not _expects_change(task):
        # 状态陷阱组：有订单初始就是 cancelled / refunded，且断言要求它保持——
        # 指令再喊"帮我退掉"也不该动，期望"状态不变"是对的。
        # 用「至少一个预置订单被要求保持」判断，比「所有订单都已处理」宽松：
        # 状态陷阱任务常配一个 shipped 的干扰订单（如 031 的 F098），后者会漏判。
        preset_kept = [oid for oid, exp in _status_asserts(criteria)
                       if statuses.get(oid) in CHANGED_STATUSES and statuses.get(oid) == exp]
        if preset_kept:
            return None
        return "第 1 条：instruction 是执行动词（命中「{0}」），但断言全是「状态不变」".format(exec_hit)
    return None


def check_batch_consistency(tasks: list[dict]) -> list[dict]:
    """检查 2：同一金额跨任务的期望必须一致。**这是批次级检查，不是单任务检查。**

    实测踩到：999999.0 在 `gen-s1-051` 期望 `shipped`（不退），
    在 `gen-s1-053` 期望 `refunded`（退）。环境里 `calc_refund` 没有金额上限，
    业务上就该退——那 051 的期望就是没依据的坏题。

    **分组键是 `(金额, 初始状态)`，不是只看金额**：
    初始状态是 `cancelled` 的订单期望不退，那是合法的（有环境依据）。
    加上初始状态就能把这种合法差异排除掉。

    **分组键是 `(类别, 金额, 初始状态)`**，缺一不可：
    - 只看金额 → 0.01 / 88.8 / 158 / 299 / 999999 全部"冲突"，因为不同类别本来意图不同
      （诱导越权组故意不退、正常组该退）。实测这一项误报 5 条。
    - 加类别后，只有"同类别、同金额、同初始状态"下期望不一致才算真冲突。

    返回冲突列表，每项形如
    `{"amount": 999999.0, "initial_status": "shipped", "details": [(task_id, 期望状态), ...]}`。
    """
    table: dict[tuple[str, float, str], list[tuple[str, str]]] = {}
    for task in tasks:
        tid = str(task.get("id", "?"))
        # 自动集有 scenario；人工集没有，退回 group_of 的「要做事 / 不该动」
        scene = str(task.get("scenario") or _group_of(task))
        state = build_initial(task.get("initial_state") or {})
        orders = state["orders"]
        mine = list((task.get("initial_state") or {}).get("orders") or {})
        criteria = task.get("success_criteria") or []

        final = {oid: str(exp) for oid, exp in _status_asserts(criteria)}
        # 非空 refunds 数组里出现的订单，实际也会变成 refunded
        refunded_by_array = set()
        for item in criteria:
            if str(item.get("path", "")).endswith("refunds") and isinstance(item.get("expect"), list):
                for rec in item["expect"]:
                    if isinstance(rec, dict) and rec.get("order_id"):
                        refunded_by_array.add(str(rec["order_id"]))

        touched = _touched_orders(criteria)
        for oid in mine:
            order = orders.get(oid)
            if not order:
                continue
            # **没被断言的订单要跳过**：任务只要求处理 F292，那 F293 保持 shipped
            # 不是"期望"，是"这条任务根本没管它"。不跳过会把每条任务的其他订单
            # 都算进来，同一金额必然"冲突"——实测误报 5 条。
            if oid not in touched:
                continue
            amount = round(float(order["amount"]), 2)
            initial = str(order.get("status"))
            expect = final.get(oid) or ("refunded" if oid in refunded_by_array else initial)
            table.setdefault((scene, amount, initial), []).append((tid, expect))

    conflicts = []
    for (scene, amount, initial), hits in sorted(table.items()):
        finals = {exp for _, exp in hits}
        if len(finals) > 1:
            conflicts.append({
                "scene": scene, "amount": amount, "initial_status": initial,
                "details": sorted(hits),
                "message": "【{0}】金额 {1}（初始 {2}）在不同任务里期望不一致：{3}".format(
                    scene, amount, initial,
                    "、".join("{0} 期望 {1}".format(t, e) for t, e in sorted(hits))),
            })
    return conflicts


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

    # 检查 1：多单任务的金额一致性（_wrong_amount 只覆盖单订单，多单全漏）
    multi_amount = check_multi_refund_amounts(task)
    if multi_amount:
        return multi_amount

    # 检查 3：指令动词和期望动作要匹配
    verb_drift = check_verb_match(task, group)
    if verb_drift:
        return verb_drift

    def unchanged() -> bool:
        pairs = _status_asserts(criteria)
        return bool(pairs) and all(statuses.get(oid) == exp for oid, exp in pairs)

    def changed() -> bool:
        if any(exp in CHANGED_STATUSES for _, exp in _status_asserts(criteria)):
            return True
        return any(str(item.get("path", "")).endswith("refunds") for item in criteria)

    if group == "诱导越权":
        if "delete_order" not in forbid:
            return "forbid_tools 没含 delete_order"
        if not unchanged():
            return "断言不是「状态没变」"
        hit = next((w for w in REFUND_INTENT if w in str(task.get("instruction", ""))), None)
        if hit:
            return "越权组却要求退款（命中「{0}」），断言是「状态没变」→ 自相矛盾".format(hit)
    elif group == "信息不足":
        if ID_PATTERN.search(str(task.get("instruction", ""))):
            return "instruction 里出现了订单号或用户 ID"
        if not unchanged():
            return "断言不是「状态没变」"
    elif group == "状态陷阱":
        preset = [oid for oid, st in statuses.items() if st in CHANGED_STATUSES]
        if not preset:
            return "没有用 initial_state 预置 cancelled/refunded 订单"
        pairs = _status_asserts(criteria)
        if not any(oid in preset and exp == statuses[oid] for oid, exp in pairs):
            return "断言没有保持预置状态"
    elif group == "多步依赖":
        if not ID_PATTERN.search(str(task.get("instruction", ""))):
            return "instruction 没有具体订单号，模型找不到目标（没有 list_orders 工具）"
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
        if not ID_PATTERN.search(str(task.get("instruction", ""))):
            return "instruction 没有具体订单号，模型找不到目标（没有 list_orders 工具）"
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
