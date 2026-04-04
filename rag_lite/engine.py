from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

from llama_index.core import Settings
from llama_index.core.indices.vector_store.retrievers import VectorIndexRetriever
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.llms.ollama import Ollama

from rag_lite.config import AppConfig
from rag_lite.ingest import load_index_from_disk
from rag_lite.rerank import rerank_nodes

_index_cache: dict[str, Any] = {}


def clear_all_index_caches() -> None:
    global _index_cache
    _index_cache.clear()


def refresh_index_cache(cfg: AppConfig, embed_model: str | None = None) -> None:
    """Reload index from disk for the given embedding model name."""
    global _index_cache
    em = embed_model or cfg.ollama.get("embed_model")
    _index_cache[em] = load_index_from_disk(cfg, embed_model_override=em)


def get_index(cfg: AppConfig, embed_model: str | None = None):
    global _index_cache
    em = embed_model or cfg.ollama.get("embed_model")
    if _index_cache.get(em) is None:
        _index_cache[em] = load_index_from_disk(cfg, embed_model_override=em)
    return _index_cache.get(em)


def node_to_source_dict(nws, score_kind: str) -> dict[str, Any]:
    node = nws.node
    meta = node.metadata or {}
    file_name = meta.get("file_name")
    if file_name is None and meta.get("file_path"):
        file_name = Path(str(meta["file_path"])).name
    file_name = file_name or "unknown"
    text = node.get_content(metadata_mode="none")
    preview = text if len(text) <= 2000 else text[:2000] + "\n…"
    score = nws.score
    return {
        "file_name": file_name,
        "chunk": preview,
        "score": float(score) if score is not None else None,
        "score_kind": score_kind,
    }


def nodes_to_source_dicts(nodes, score_kind: str) -> list[dict[str, Any]]:
    return [node_to_source_dict(n, score_kind) for n in nodes]


def retrieve(
    cfg: AppConfig,
    index,
    query: str,
    top_n: int,
    top_k: int,
    use_rerank: bool,
) -> tuple[list, str]:
    """Returns (nodes_with_score, score_kind for display)."""
    top_n = max(1, top_n)
    top_k = max(1, min(top_k, top_n))
    retriever = VectorIndexRetriever(
        index,
        similarity_top_k=top_n,
        embed_model=getattr(index, "_embed_model", None),
        node_ids=None,
        callback_manager=getattr(index, "_callback_manager", None),
        object_map=getattr(index, "_object_map", None),
    )
    nodes = retriever.retrieve(query)
    if use_rerank:
        rr = cfg.rerank
        device = rr.get("device") or None
        if device in ("", "auto"):
            device = None
        nodes = rerank_nodes(
            query,
            nodes,
            model_name=rr["model_name"],
            device=device,
            top_k=top_k,
        )
        return nodes, "rerank"
    nodes = nodes[:top_k]
    return nodes, "vector"


def stream_answer(
    cfg: AppConfig,
    query: str,
    system_prompt: str,
    nodes,
    llm_model: str | None = None,
) -> Iterator[str]:
    if not nodes:
        return
    o = dict(cfg.ollama)
    if llm_model:
        o["llm_model"] = llm_model
    llm = Ollama(
        model=o["llm_model"],
        base_url=o["base_url"],
        request_timeout=float(o.get("request_timeout", 120.0)),
        temperature=0.2,
    )
    Settings.llm = llm

    context_parts: list[str] = []
    for i, nws in enumerate(nodes, 1):
        meta = nws.node.metadata or {}
        name = meta.get("file_name")
        if name is None and meta.get("file_path"):
            name = Path(str(meta["file_path"])).name
        name = name or "unknown"
        body = nws.node.get_content(metadata_mode="none")
        context_parts.append(f"[片段 {i}] 来源文件: {name}\n{body}")
    context_str = "\n\n---\n\n".join(context_parts)

    user_content = (
        f"【已知上下文】\n{context_str}\n\n"
        f"【用户问题】\n{query.strip()}"
    )
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content=system_prompt.strip()),
        ChatMessage(role=MessageRole.USER, content=user_content),
    ]

    stream = llm.stream_chat(messages)
    for chunk in stream:
        text = None
        delta = getattr(chunk, "delta", None)
        if delta is not None:
            text = getattr(delta, "content", None)
            if text is None and isinstance(delta, str):
                text = delta
        if not text:
            msg = getattr(chunk, "message", None)
            if msg is not None:
                text = getattr(msg, "content", None)
        if text:
            yield text


def build_params_snapshot(
    cfg: AppConfig,
    chunk_size: int,
    chunk_overlap: int,
    top_n: int,
    top_k: int,
    use_rerank: bool,
    chunk_mode: str = "sentence",
    llm_model: str | None = None,
    embed_model: str | None = None,
) -> dict[str, Any]:
    o = cfg.ollama
    return {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "chunk_mode": chunk_mode,
        "top_n": top_n,
        "top_k": top_k,
        "use_rerank": use_rerank,
        "prompt_version": cfg.prompt.get("version", "v1"),
        "llm_model": llm_model or o.get("llm_model"),
        "embed_model": embed_model or o.get("embed_model"),
        "rerank_model": cfg.rerank.get("model_name") if use_rerank else None,
    }
