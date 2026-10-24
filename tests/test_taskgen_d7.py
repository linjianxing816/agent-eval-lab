"""D7 去重：归一化 / 2-gram / Jaccard / 事后审计。全部离线。"""
from __future__ import annotations

from evalkit.taskgen.dedup import (
    audit_duplicates,
    find_duplicate,
    jaccard,
    normalize_text,
    pairwise_stats,
    shingles,
    top_pairs,
)


def test_normalize_replaces_ids_without_word_boundary_bug():
    # `\b` 在中文里不成立（汉字也算 \w），所以不能用 \b 包住 ID
    assert normalize_text("F113 这单帮我退一下") == "<order>这单帮我退一下"
    assert normalize_text("帮 U1001 退 A123") == "帮<user>退<order>"
    # 两个只差订单号的句子，归一化后必须完全一样
    assert normalize_text("F072 这单客户又来问了") == normalize_text("F080 这单客户又来问了")


def test_user_id_replaced_before_order_id():
    # U1001 如果先按订单号规则处理，会被切成 "U100"
    assert normalize_text("U1001") == "<user>"


def test_shingles_are_character_bigrams():
    assert shingles("这单太麻烦") == {"这单", "单太", "太麻", "麻烦"}


def test_jaccard_bounds():
    assert jaccard({"a", "b"}, {"a", "b"}) == 1.0
    assert jaccard({"a"}, {"b"}) == 0.0
    assert jaccard(set(), set()) == 1.0


def test_same_sentence_different_order_id_is_duplicate():
    pool = [("F113 这单帮我退一下", shingles("F113 这单帮我退一下"))]
    score, hit = find_duplicate("F133 这单帮我退一下", pool, 0.85)
    assert score == 1.0 and hit is not None


def test_different_intent_is_not_duplicate():
    pool = [("F113 这单帮我退一下", shingles("F113 这单帮我退一下"))]
    score, hit = find_duplicate("F133 的信息乱了，清掉重新录", pool, 0.85)
    assert hit is None and score < 0.85


def test_audit_counts_duplicates_and_keeps_first():
    texts = [
        "F001 这单帮我退一下",
        "F002 的信息乱了，清掉重新录",
        "F003 这单帮我退一下",          # 和第一条只差订单号
    ]
    result = audit_duplicates(texts, 0.85)
    assert result["total"] == 3
    assert result["duplicate_count"] == 1
    assert result["kept"] == 2
    assert result["duplicates"][0]["score"] == 1.0


def test_pairwise_stats_shape():
    stats = pairwise_stats(["退 A123", "取消 A123 发货", "查一下 B456"])
    assert stats["pairs"] == 3
    assert 0.0 <= stats["mean"] <= 1.0
    assert stats["max"] >= stats["p50"]


def test_top_pairs_sorted_desc():
    pairs = top_pairs(["F001 退一下", "F002 退一下", "完全不同的一句话"], 2)
    assert len(pairs) == 2
    assert pairs[0][0] >= pairs[1][0]
