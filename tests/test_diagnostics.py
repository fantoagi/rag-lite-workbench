"""分数口径：不导入 LlamaIndex / 重排模型。"""
from rag_lite.diagnostics import (
    apply_rerank_metadata,
    format_diag_score,
    resolve_retrieval_score_kind,
    score_kind_label,
)


def test_score_kind_label_distinguishes_hybrid_from_vector():
    assert "混合" in score_kind_label("hybrid")
    assert "向量" in score_kind_label("vector")
    assert "关键词" in score_kind_label("keyword_fallback")
    assert "重排" in score_kind_label("rerank")


def test_vector_failure_keeps_rerank_score_kind_when_rerank_ran():
    # 重排后的数字是概率。降级不能再把 score_kind 改成 keyword_fallback，
    # 否则界面会把 0.8 读成 BM25。
    kind = resolve_retrieval_score_kind(
        vector_failed=True,
        keyword_enabled=True,
        has_final_nodes=True,
        rerank_applied=True,
        requested_mode="hybrid",
    )
    assert kind == "rerank"
    kind_plain = resolve_retrieval_score_kind(
        vector_failed=True,
        keyword_enabled=True,
        has_final_nodes=True,
        rerank_applied=False,
        requested_mode="hybrid",
    )
    assert kind_plain == "keyword_fallback"
    assert (
        resolve_retrieval_score_kind(
            vector_failed=True,
            keyword_enabled=True,
            has_final_nodes=False,
            rerank_applied=False,
            requested_mode="hybrid",
        )
        == "vector_error"
    )
    assert (
        resolve_retrieval_score_kind(
            vector_failed=False,
            keyword_enabled=True,
            has_final_nodes=True,
            rerank_applied=False,
            requested_mode="hybrid",
        )
        == "hybrid"
    )


def test_rerank_does_not_overwrite_vector_score_with_rrf():
    md = apply_rerank_metadata(
        {
            "rag_vector_score": 0.81,
            "rag_keyword_score": 2.4,
            "rag_merged_score": 0.0164,
            "rag_retrieval_sources": ["vector", "keyword"],
        },
        pre_score=0.0164,
        relevance_prob=0.73,
    )
    assert md["rag_vector_score"] == 0.81
    assert md["rag_pre_rerank_score"] == 0.0164
    assert md["rag_rerank_score"] == 0.73
    assert md["rag_keyword_score"] == 2.4


def test_vector_only_path_records_pre_score_as_vector_score():
    md = apply_rerank_metadata({}, pre_score=0.42, relevance_prob=0.9)
    assert md["rag_vector_score"] == 0.42
    keyword_only = apply_rerank_metadata(
        {"rag_retrieval_sources": ["keyword"], "rag_keyword_score": 3.2},
        pre_score=3.2,
        relevance_prob=0.2,
    )
    assert "rag_vector_score" not in keyword_only
    assert keyword_only["rag_pre_rerank_score"] == 3.2


def test_format_diag_score_names_hybrid_and_keeps_vector_channel():
    text = format_diag_score(
        {
            "score": 0.0164,
            "score_kind": "hybrid",
            "vector_score": 0.81,
            "keyword_score": 2.4,
        }
    )
    assert "混合" in text
    assert "0.0164" in text
    assert "向量 0.8100" in text
    assert "关键词 2.4000" in text
    vector_only = format_diag_score({"score": 0.81, "score_kind": "vector", "vector_score": 0.81})
    assert "向量检索" in vector_only
    assert vector_only.count("0.8100") == 1
