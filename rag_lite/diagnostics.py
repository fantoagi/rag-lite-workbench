"""检索诊断里与模型无关的口径：分数标签、重排前后分数、降级后的 score_kind。

对话和评测都靠这些字段判断「向量分 / 混合分 / 重排分」是不是同一种数。
放在这里是为了不导入 LlamaIndex / Ollama 也能单测。
"""
from __future__ import annotations

from typing import Any


def score_kind_label(kind: str | None) -> str:
    """把 score_kind 译成界面上的分数名称。未知值按向量分处理，避免空白徽章。"""
    k = str(kind or "").strip().lower()
    if k == "rerank":
        return "Cross-Encoder 重排（相关性 0–1）"
    if k in {"keyword", "keyword_fallback"}:
        return "关键词/BM25 得分"
    if k == "hybrid":
        return "混合召回得分（RRF）"
    if k == "vector_error":
        return "检索失败"
    return "向量检索得分"


def resolve_retrieval_score_kind(
    *,
    vector_failed: bool,
    keyword_enabled: bool,
    has_final_nodes: bool,
    rerank_applied: bool,
    requested_mode: str,
) -> str:
    """向量阶段失败时不要再标成 hybrid。

    重排若已经跑完，分数是相关性概率，score_kind 保持 rerank；
    降级事实另由 retrieval_degraded / vector_error 表达。
    """
    if vector_failed:
        if keyword_enabled and has_final_nodes:
            return "rerank" if rerank_applied else "keyword_fallback"
        return "vector_error"
    if rerank_applied:
        return "rerank"
    mode = str(requested_mode or "").strip().lower()
    return mode or "vector"


def apply_rerank_metadata(
    metadata: dict[str, Any] | None,
    pre_score: float | None,
    relevance_prob: float,
) -> dict[str, Any]:
    """写入重排分，且不把混合 RRF 分覆盖成「向量分」。

    纯向量路径没有 rag_retrieval_sources，当前 score 就是向量分，可以补记。
    hybrid 合并后 score 已是 RRF，向量分应留在已有的 rag_vector_score 里。
    """
    md = dict(metadata or {})
    sources = [str(x) for x in (md.get("rag_retrieval_sources") or [])]
    if pre_score is not None:
        md["rag_pre_rerank_score"] = float(pre_score)
        if md.get("rag_vector_score") is None and "keyword" not in sources:
            md["rag_vector_score"] = float(pre_score)
    md["rag_rerank_score"] = float(relevance_prob)
    return md


def format_diag_score(row: dict[str, Any] | None) -> str:
    """诊断表「得分」单元格：主分数带口径，并附上另一种召回分（若有）。"""
    row = row or {}
    score = row.get("score")
    try:
        score_s = "—" if score is None else f"{float(score):.4f}"
    except (TypeError, ValueError):
        score_s = "—"
    kind = str(row.get("score_kind") or "").strip().lower()
    label = score_kind_label(kind) if kind else ""
    head = f"{score_s}（{label}）" if label else score_s
    extras: list[str] = []
    vector_score = row.get("vector_score")
    keyword_score = row.get("keyword_score")
    try:
        if vector_score is not None and kind not in {"vector", ""}:
            extras.append(f"向量 {float(vector_score):.4f}")
    except (TypeError, ValueError):
        pass
    try:
        if keyword_score is not None and kind not in {"keyword", "keyword_fallback"}:
            extras.append(f"关键词 {float(keyword_score):.4f}")
    except (TypeError, ValueError):
        pass
    if extras:
        return head + " / " + " / ".join(extras)
    return head
