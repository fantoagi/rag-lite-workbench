from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from llama_index.core.schema import NodeWithScore

_cross_encoder = None
_cross_model_name: str | None = None


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
    scores = ce.predict(pairs)
    ranked = sorted(zip(scores, nodes), key=lambda x: float(x[0]), reverse=True)
    out: list[NodeWithScore] = []
    for score, nws in ranked[:top_k]:
        nws.score = float(score)
        out.append(nws)
    return out
