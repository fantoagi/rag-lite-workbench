# -*- coding: utf-8 -*-
"""Unit tests for rag_lite.eval_judge.

Default config is tightened for finance CJK corpora. Use
``EvalJudgeConfig.legacy()`` when asserting historical loose behavior.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rag_lite.eval_judge import (  # noqa: E402
    ANSWER_NO_CONTEXT,
    ANSWER_RETRIEVAL_ONLY,
    _ABSTAIN_PHRASES,
    EvalJudgeConfig,
    chunk_match_diagnostics,
    chunk_match_metrics,
    is_abstain_answer,
    judge_case,
    text_match_loose,
    texts_match_expected_chunk,
)


# ----- text_match_loose -----------------------------------------------------


def test_text_match_loose_rejects_near_duplicate_cjk_company_names():
    # 默认收紧后：招商证券 vs 浙商证券 不得算 chunk 命中
    assert text_match_loose("招商证券", "浙商证券") is False


def test_text_match_loose_legacy_still_accepts_near_duplicate():
    assert text_match_loose("招商证券", "浙商证券", EvalJudgeConfig.legacy()) is True


def test_text_match_loose_preserves_substring_and_legitimate_cjk():
    assert text_match_loose("招商证券2024年年度报告", "招商证券2024年年度报告全文") is True
    a = "招商证券股份有限公司总部位于深圳"
    b = "招商证券股份有限公司总部位于深圳市"
    assert text_match_loose(a, b) is True
    long_a = "the quick brown fox jumps over the lazy dog and runs into the forest"
    long_b = "the quick brown fox jumps over the lazy dog and runs into the woods"
    assert text_match_loose(long_a, long_b) is True
    assert text_match_loose("abc", "abcdef") is True


def test_text_match_loose_disjoint_returns_false():
    assert text_match_loose("招商证券", "中国银行") is False
    assert text_match_loose("", "招商证券") is False
    assert text_match_loose("招商证券", "") is False
    assert text_match_loose("", "") is False


def test_text_match_loose_cjk_bigram_gate():
    a = "招商证券"
    b = "浙商证券"
    # 门禁开启 + 高 ratio 门槛：拒绝近音公司名
    assert text_match_loose(a, b) is False
    # legacy：ratio 分支仍可通过
    assert text_match_loose(a, b, EvalJudgeConfig.legacy()) is True


def test_text_match_loose_near_containment_in_long_chunk():
    # 金标与长片段仅差少量字符时，应按近包含命中，而不是被长文本 bigram 稀释拒掉
    expect = "中国最早的圆形铸币是战国中期的圜钱亦称环钱流通全国的则是秦始皇为统一中国货币而铸造的秦半两"
    actual = (
        "货币形态逐渐趋于统一的圆形。因为圆形便于携带不易磨损。"
        + expect
        + "这种铸币为圆形中间有方孔一直沿用到清末。"
    )
    assert text_match_loose(expect, actual) is True


def test_texts_match_expected_chunk_multipart_and_joined():
    part_a = "扩大的价值形式虽然相对于简单的偶然的价值形式有了进一步的发展"
    part_b = "扩大的价值形式的缺点是商品价值未能获得共同的统一的表现形式"
    assert texts_match_expected_chunk(
        f"{part_a}...{part_b}",
        [f"前文。{part_a}。后文。", f"开头。{part_b}。结尾。"],
    ) is True
    assert texts_match_expected_chunk(
        f"{part_a}...{part_b}",
        [f"只有第一段：{part_a}"],
    ) is False


# ----- judge_case -----------------------------------------------------------


def _resolved(file_names):
    return {
        "raw_values": list(file_names),
        "resolved_files": list(file_names),
        "unresolved_files": [],
        "ambiguous_files": [],
        "suggestions": [],
        "has_unresolved": False,
        "has_ambiguous": False,
    }


def test_judge_case_ok_path():
    case = {
        "question": "招商证券总部在哪里？",
        "expected_file_names": ["招商证券年报.pdf"],
        "expected_chunk_content": "招商证券总部位于深圳",
        "expected_answer": "深圳",
    }
    vector_diag = [{"file_name": "招商证券年报.pdf", "chunk": "招商证券总部位于深圳福田"}]
    sources = [{"file_name": "招商证券年报.pdf", "chunk": "招商证券总部位于深圳福田"}]
    answer = "招商证券总部位于深圳。"
    result = judge_case(
        case,
        vector_diag,
        sources,
        answer,
        expected_file_info=_resolved(["招商证券年报.pdf"]),
    )
    assert result["error_type"] == "OK"
    assert result["candidate_hit"] is True
    assert result["context_hit"] is True
    assert result["chunk_hit"] is True
    assert result["answer_hit"] is True


def test_judge_case_no_context_sentinel_is_target_file_miss():
    case = {
        "question": "招商证券总部在哪里？",
        "expected_file_names": ["招商证券年报.pdf"],
        "expected_chunk_content": "招商证券总部位于深圳",
        "expected_answer": "深圳",
    }
    result = judge_case(
        case,
        [],
        [],
        ANSWER_NO_CONTEXT,
        expected_file_info=_resolved(["招商证券年报.pdf"]),
        no_context=True,
    )
    assert result["error_type"] == "TARGET_FILE_MISS"
    assert result["abstain_with_target_context"] is False
    assert "TARGET_FILE_MISS" in result["issue_codes"]
    assert "ABSTAIN_WHEN_HAS_TARGET_CONTEXT" not in result["issue_codes"]


def test_judge_case_no_context_with_allow_abstain_is_ok():
    case = {
        "question": "资料里没有的问题？",
        "expected_file_names": [],
        "allow_abstain": True,
    }
    result = judge_case(
        case,
        [],
        [],
        ANSWER_NO_CONTEXT,
        expected_file_info=_resolved([]),
        no_context=True,
    )
    assert result["error_type"] == "OK"
    assert result["abstain_actual"] is True
    assert result["abstain_correct"] is True


def test_judge_case_retrieval_only_skips_answer_abstain():
    case = {
        "question": "招商证券总部在哪里？",
        "expected_file_names": ["招商证券年报.pdf"],
        "expected_answer": "深圳",
        "allow_abstain": True,
    }
    sources = [{"file_name": "招商证券年报.pdf", "chunk": "招商证券总部位于深圳"}]
    result = judge_case(
        case,
        sources,
        sources,
        ANSWER_RETRIEVAL_ONLY,
        expected_file_info=_resolved(["招商证券年报.pdf"]),
        generation_mode="retrieval_only",
    )
    assert result["answer_hit"] is None
    assert result["abstain_correct"] is None
    assert result["error_type"] == "OK"
    assert result["retrieval_only"] is True


def test_judge_case_target_file_miss():
    case = {
        "question": "招商证券总部在哪里？",
        "expected_file_names": ["招商证券年报.pdf"],
        "expected_chunk_content": "招商证券总部位于深圳",
        "expected_answer": "深圳",
    }
    vector_diag = [{"file_name": "中国银行年报.pdf", "chunk": "中国银行总部位于北京"}]
    sources = []
    answer = "中国银行总部位于北京。"
    result = judge_case(
        case,
        vector_diag,
        sources,
        answer,
        expected_file_info=_resolved(["招商证券年报.pdf"]),
    )
    assert result["error_type"] == "TARGET_FILE_MISS"
    assert "TARGET_FILE_MISS" in result["issue_codes"]


def test_judge_case_answered_when_should_abstain():
    case = {
        "question": "招商证券海外子公司名单？",
        "expected_file_names": [],
        "allow_abstain": True,
    }
    ok = judge_case(case, [], [], "无法回答。", expected_file_info=_resolved([]))
    assert ok["error_type"] == "OK"
    assert ok["abstain_correct"] is True

    bad = judge_case(
        case,
        [],
        [],
        "根据资料，海外子公司有 A、B、C。",
        expected_file_info=_resolved([]),
    )
    assert bad["error_type"] == "ANSWERED_WHEN_SHOULD_ABSTAIN"


def test_judge_case_abstain_when_has_target_context():
    case = {
        "question": "招商证券总部在哪里？",
        "expected_file_names": ["招商证券年报.pdf"],
        "expected_answer": "深圳",
    }
    sources = [{"file_name": "招商证券年报.pdf", "chunk": "招商证券总部位于深圳"}]
    result = judge_case(
        case,
        sources,
        sources,
        "无法回答，未检索到任何文档片段。",
        expected_file_info=_resolved(["招商证券年报.pdf"]),
    )
    assert result["error_type"] == "ABSTAIN_WHEN_HAS_TARGET_CONTEXT"
    assert result["abstain_with_target_context"] is True


def test_judge_case_rerank_drop():
    case = {
        "question": "招商证券总部在哪里？",
        "expected_file_names": ["招商证券年报.pdf"],
        "expected_chunk_content": "招商证券总部位于深圳",
    }
    vector_diag = [{"file_name": "招商证券年报.pdf", "chunk": "招商证券总部位于深圳"}]
    final_ranked = [{"file_name": "中国银行年报.pdf", "chunk": "中国银行总部位于北京"}]
    sources = [{"file_name": "中国银行年报.pdf", "chunk": "中国银行总部位于北京"}]
    result = judge_case(
        case,
        vector_diag,
        sources,
        "中国银行总部位于北京。",
        final_ranked_sources=final_ranked,
        use_rerank=True,
        expected_file_info=_resolved(["招商证券年报.pdf"]),
    )
    assert result["error_type"] == "RERANK_DROP"
    assert "RERANK_DROP" in result["issue_codes"]


def test_chunk_match_diagnostics_shape():
    diag = chunk_match_diagnostics(
        expected_chunk="招商证券总部位于深圳",
        candidates=[
            {"file_name": "a.pdf", "chunk": "招商证券总部位于深圳福田", "score": 0.9, "score_kind": "vector"},
            {"file_name": "b.pdf", "chunk": "完全无关的内容", "score": 0.1, "score_kind": "vector"},
        ],
        sources=[
            {"file_name": "a.pdf", "chunk": "招商证券总部位于深圳福田", "score": 0.9, "score_kind": "hybrid"},
        ],
    )
    assert diag["has_expected_chunk"] is True
    assert diag["best"]["file_name"] == "a.pdf"
    assert diag["best"]["hit"] is True


def test_chunk_match_metrics_basic():
    m = chunk_match_metrics("招商证券总部位于深圳", "招商证券总部位于深圳福田")
    assert m["hit"] is True
    m2 = chunk_match_metrics("招商证券总部", "中国银行总部")
    assert m2["hit"] is False


def test_is_abstain_answer_patterns():
    assert is_abstain_answer("无法回答") is True
    assert is_abstain_answer("未检索到任何文档片段") is True
    assert is_abstain_answer("招商证券总部位于深圳") is False
    assert is_abstain_answer(ANSWER_NO_CONTEXT) is False
    assert is_abstain_answer(ANSWER_RETRIEVAL_ONLY) is False


def test_abstain_phrases_match_historical():
    assert "\u65e0\u6cd5\u56de\u7b54" in _ABSTAIN_PHRASES
    assert "\u672a\u68c0\u7d22\u5230\u4efb\u4f55\u6587\u6863\u7247\u6bb5" in _ABSTAIN_PHRASES
