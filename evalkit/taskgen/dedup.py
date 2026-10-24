"""Jaccard 相似度去重。

【为什么用字符 2-gram】
中文没有空格，词级切分要装 jieba（禁区：不装新依赖）。
而且我们的 instruction 平均只有 17.7 字，3-gram 只有 15 个、2-gram 有 16 个左右——
短句上 3-gram 太稀疏，改两个词相似度就掉一半。2-gram 最稳。

【归一化必须先做】
`F006 这单有问题` 和 `F013 这单有问题` 只差订单号，不归一化 Jaccard 算出来是 0.6 左右，
过不了任何合理阈值。把订单号/用户 ID 换成 `<ORDER>` / `<USER>` 之后两条完全一样，相似度 1.0。

**这把尺子同时也是 D7 的"句式多样性"指标**：两两相似度的 均值 / P50 / P90 / 最大值。
比"唯一开头词数"客观——中文开头词没法切，两个人算出来不一样。
"""
from __future__ import annotations

import re
from itertools import combinations

PUNCT = re.compile(r"[\s，。！？、,.!?;；:：\"'“”‘’（）()【】\[\]《》…—-]")
# **不能用 \b**：中文汉字也算 \w，`F072这` 之间没有词边界，正则匹配不上，
# 订单号就原样留着 → "F113 这单..." 和 "F133 这单..." 归一化后仍然不同。
# 先换用户 ID（U + 4 位数字），再换订单号（字母 + 3 位数字），顺序不能反。
USER_ID = re.compile(r"U\d{4}")
ORDER_ID = re.compile(r"[A-Z]\d{3}")


def normalize_text(text: str) -> str:
    """去标点 + 实体 ID 换占位符 + 小写。"""
    cleaned = PUNCT.sub("", text or "")
    cleaned = USER_ID.sub("<USER>", cleaned)
    cleaned = ORDER_ID.sub("<ORDER>", cleaned)
    return cleaned.lower()


def shingles(text: str, n: int = 2) -> set[str]:
    """字符 n-gram。先归一化，再切。"""
    norm = normalize_text(text)
    if len(norm) <= n:
        return {norm} if norm else set()
    return {norm[i:i + n] for i in range(len(norm) - n + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def find_duplicate(text: str, pool: list[tuple[str, set[str]]],
                   threshold: float) -> tuple[float, str | None]:
    """在 pool 里找最相似的一条。返回 (最高相似度, 撞上的原文)；没撞上第二项为 None。"""
    target = shingles(text)
    best, hit = 0.0, None
    for original, grams in pool:
        score = jaccard(target, grams)
        if score > best:
            best, hit = score, original
    if best >= threshold:
        return best, hit
    return best, None


def pairwise_stats(instructions: list[str]) -> dict:
    """两两相似度分布。这是 D7 的"句式多样性"指标。"""
    grams = [(text, shingles(text)) for text in instructions]
    scores: list[float] = []
    for (_, a), (_, b) in combinations(grams, 2):
        scores.append(jaccard(a, b))
    if not scores:
        return {"pairs": 0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0}
    ordered = sorted(scores)
    return {
        "pairs": len(scores),
        "mean": round(sum(scores) / len(scores), 4),
        "p50": round(ordered[len(ordered) // 2], 4),
        "p90": round(ordered[int(len(ordered) * 0.9)], 4),
        "max": round(ordered[-1], 4),
    }


def top_pairs(instructions: list[str], limit: int = 5) -> list[tuple[float, str, str]]:
    """最相似的几对，用来人眼确认阈值合不合理。"""
    grams = [(text, shingles(text)) for text in instructions]
    scored = [(jaccard(a, b), x, y) for (x, a), (y, b) in combinations(grams, 2)]
    scored.sort(key=lambda item: item[0], reverse=True)
    return [(round(s, 4), x, y) for s, x, y in scored[:limit]]


def audit_duplicates(instructions: list[str], threshold: float,
                     groups: list[str] | None = None) -> dict:
    """事后审一份已存在的集合：如果当初就用这个阈值去重，会丢多少条。

    用来看 S1（已经生成完、当时没去重）的重复率。**同一把尺子量两份数据**，
    S1 和 S2 才可比。

    **判重只在同一 group 内做**（传了 `groups` 的话）。跨类别不判重——
    实测踩过：人工集 task-015（状态陷阱）和 task-034（正常）文案一模一样，
    但一个测"已取消的单再取消"、一个测"正常单取消发货"，判重就是误判。
    而自动集"换个订单号重说一遍"是同类别内的重复，照样该抓。
    """
    if groups is not None and len(groups) != len(instructions):
        raise ValueError("groups 和 instructions 长度不一致")
    pools: dict[str, list[tuple[str, set[str]]]] = {}
    duplicates: list[dict] = []
    for index, text in enumerate(instructions):
        group = groups[index] if groups is not None else "*"
        pool = pools.setdefault(group, [])
        score, hit = find_duplicate(text, pool, threshold)
        if hit is not None:
            duplicates.append({"instruction": text, "similar_to": hit,
                               "score": round(score, 4), "group": group})
        else:
            pool.append((text, shingles(text)))
    total = len(instructions)
    return {
        "total": total,
        "kept": len(pool),
        "duplicate_count": len(duplicates),
        "duplicate_rate": round(len(duplicates) / total, 4) if total else 0.0,
        "duplicates": duplicates,
    }
