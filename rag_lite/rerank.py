from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from llama_index.core.schema import NodeWithScore

_cross_encoder = None
_cross_model_name: str | None = None


def _logits_to_relevance_prob(logits: np.ndarray) -> np.ndarray:
    """
    sentence-transformers CrossEncoder（如 BAAI/bge-reranker-*）对 query–passage 输出的是 **logits**，
    数值常在 [-10, 10] 附近、可为接近 0 的小数；**与向量余弦相似度（常显示为 0.x）不是同一量纲**。
    映射到 (0,1) 便于界面展示；sigmoid 单调，不改变候选间的排序。
    """
    x = np.asarray(logits, dtype=np.float64)
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def get_cross_encoder(model_name: str, device: str | None):
    global _cross_encoder, _cross_model_name
    from sentence_transformers import CrossEncoder

    if _cross_encoder is None or _cross_model_name != model_name:
        kwargs: dict[str, Any] = {}
        if device:
            kwargs["device"] = device
        _cross_encoder = CrossEncoder(model_name, **kwargs)
        _cross_model_name = model_name
    return _cross_encoder


def rerank_nodes(
    query: str,
    nodes: list[NodeWithScore],
    model_name: str,
    device: str | None,
    top_k: int,
) -> list[NodeWithScore]:
    if not nodes:
        return []
    ce = get_cross_encoder(model_name, device)
    pairs = [(query, n.node.get_content(metadata_mode="none")) for n in nodes]
    logits = np.asarray(ce.predict(pairs), dtype=np.float64)
    ranked = sorted(zip(logits, nodes), key=lambda x: float(x[0]), reverse=True)
    out: list[NodeWithScore] = []
    for logit, nws in ranked[:top_k]:
        vec_score = nws.score
        prob = float(_logits_to_relevance_prob(np.array([logit], dtype=np.float64))[0])
        # 供界面与 node_to_source_dict 对比展示：重排前的向量检索分
        md = dict(nws.node.metadata or {})
        if vec_score is not None:
            md["rag_vector_score"] = float(vec_score)
        nws.node.metadata = md
        nws.score = prob
        out.append(nws)
    return out
