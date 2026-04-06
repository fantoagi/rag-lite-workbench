from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import httpx
from ollama import Client as OllamaSdkClient

from llama_index.core import Settings
from llama_index.core.indices.vector_store.retrievers import VectorIndexRetriever
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.llms.ollama import Ollama

from rag_lite.config import AppConfig
from rag_lite.ingest import load_index_from_disk
from rag_lite.rerank import rerank_nodes

_index_cache: dict[str, Any] = {}


def _ollama_context_window_for_llm(o: dict[str, Any]) -> int:
    """
    LlamaIndex Ollama 会把 context_window 映射为请求里的 num_ctx。
    若 Modelfile 把 num_ctx 拉到极大（如 262144），KV 显存占满后常见整段回退 CPU，
    ollama ps 会显示 100% CPU；工作台侧用合理 num_ctx 可更易留在 GPU。
    未配置 ollama.num_ctx 时默认 16384；若显式写 -1 则交给模型/服务端默认。
    """
    if "num_ctx" not in o:
        return 16384
    try:
        v = int(o["num_ctx"])
    except (TypeError, ValueError):
        return 16384
    return v


def _ollama_llm_sdk_client(o: dict[str, Any]) -> OllamaSdkClient:
    """
    为 LlamaIndex Ollama 提供底层 ollama.Client。
    仅传 float 给 httpx 时，流式场景下「首包/两次 token 间隔」仍可能触发 ReadTimeout；
    这里显式设置 connect / read，并默认放大 read（本地 CPU 慢或上下文长时更安全）。
    """
    base = str(o.get("base_url") or "http://127.0.0.1:11434")
    read_sec = float(o.get("request_timeout", 600.0))
    conn_sec = float(o.get("connect_timeout", 60.0))
    # pool：连接池等待；write：与 Ollama 交互一般同 read 量级即可
    timeout = httpx.Timeout(
        connect=conn_sec,
        read=read_sec,
        write=max(read_sec, 120.0),
        pool=conn_sec,
    )
    return OllamaSdkClient(host=base, timeout=timeout)


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
    md_now = node.metadata
    vec_raw = md_now.get("rag_vector_score") if isinstance(md_now, dict) else None
    return {
        "file_name": file_name,
        "chunk": preview,
        "score": float(score) if score is not None else None,
        "score_kind": score_kind,
        # 开启重排时由 rerank.py 写入，便于与重排分对照
        "vector_score": float(vec_raw) if vec_raw is not None else None,
    }


def nodes_to_source_dicts(nodes, score_kind: str) -> list[dict[str, Any]]:
    return [node_to_source_dict(n, score_kind) for n in nodes]


def vector_retrieve(cfg: AppConfig, index, query: str, top_n: int):
    """仅向量初筛（Top-N），供 UI 分阶段展示「检索中」进度。"""
    top_n = max(1, top_n)
    retriever = VectorIndexRetriever(
        index,
        similarity_top_k=top_n,
        embed_model=getattr(index, "_embed_model", None),
        node_ids=None,
        callback_manager=getattr(index, "_callback_manager", None),
        object_map=getattr(index, "_object_map", None),
    )
    return retriever.retrieve(query)


def apply_topk_rerank(
    cfg: AppConfig,
    query: str,
    nodes: list,
    top_k: int,
    use_rerank: bool,
) -> tuple[list, str]:
    """在 vector_retrieve 结果上做 Top-K 截断或 Cross-Encoder 重排。"""
    if not nodes:
        return [], "vector"
    top_k = max(1, min(top_k, len(nodes)))
    if use_rerank:
        rr = cfg.rerank
        device = rr.get("device") or None
        if device in ("", "auto"):
            device = None
        out = rerank_nodes(
            query,
            nodes,
            model_name=rr["model_name"],
            device=device,
            top_k=top_k,
        )
        return out, "rerank"
    return nodes[:top_k], "vector"


def _text_from_blocks_text_only(msg: Any) -> str:
    """只拼接正文 TextBlock，避免把 ThinkingBlock 与正文重复累计。"""
    if msg is None:
        return ""
    parts: list[str] = []
    for b in getattr(msg, "blocks", []) or []:
        if type(b).__name__ == "TextBlock":
            parts.append(getattr(b, "text", "") or "")
    return "".join(parts)


def _extract_stream_delta(chunk: Any) -> str | None:
    """从 LlamaIndex ChatResponse 取增量文本；兼容不同版本 / pydantic 访问方式。"""
    d = getattr(chunk, "delta", None)
    if isinstance(d, str) and d:
        return d
    if d is not None:
        c = getattr(d, "content", None)
        if isinstance(c, str) and c:
            return c
    raw = getattr(chunk, "raw", None)
    if isinstance(raw, dict):
        mc = (raw.get("message") or {}).get("content")
        if isinstance(mc, str) and mc:
            return mc
    return None


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
    nodes = vector_retrieve(cfg, index, query, top_n)
    return apply_topk_rerank(cfg, query, nodes, top_k, use_rerank)


def stream_answer(
    cfg: AppConfig,
    query: str,
    system_prompt: str,
    nodes,
    llm_model: str | None = None,
    llm_num_ctx: int | None = None,
) -> Iterator[str]:
    if not nodes:
        return
    o = dict(cfg.ollama)
    if llm_model:
        o["llm_model"] = llm_model
    if llm_num_ctx is not None:
        o["num_ctx"] = int(llm_num_ctx)
    read_sec = float(o.get("request_timeout", 600.0))

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
    max_c = int(o.get("max_llm_context_chars", 100_000))
    if len(user_content) > max_c:
        user_content = (
            user_content[:max_c]
            + "\n\n[系统提示：已知上下文过长，已截断尾部；请缩小检索 Top-K 或缩短文档片段。]"
        )
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content=system_prompt.strip()),
        ChatMessage(role=MessageRole.USER, content=user_content),
    ]

    # 思考链模型：LlamaIndex 对 content 为 None 的流式片段会 continue，界面长时间停在「正在生成」。
    # 默认关闭 thinking；若需开启可在 config.yaml 设置 ollama.thinking
    _think = o.get("thinking", False)
    _ctx = _ollama_context_window_for_llm(o)
    _kw = dict(
        model=o["llm_model"],
        base_url=o["base_url"],
        request_timeout=read_sec,
        temperature=0.2,
        client=_ollama_llm_sdk_client(o),
    )
    if _ctx > 0:
        _kw["context_window"] = _ctx
    try:
        llm = Ollama(**_kw, thinking=_think if _think is not None else False)
    except TypeError:
        llm = Ollama(**_kw)
    Settings.llm = llm

    stream = llm.stream_chat(messages)
    # 必须统一累计「已从流里发出的正文」：若前半段走 delta、后半段只有 message.full，
    # 旧逻辑只累计 message 分支，会把 delta 已输出的前缀再整段 yield 一次，造成重复。
    emitted = ""
    for chunk in stream:
        piece = _extract_stream_delta(chunk)
        if piece:
            emitted += piece
            yield piece
            continue
        msg = getattr(chunk, "message", None)
        if msg is None:
            continue
        full = _text_from_blocks_text_only(msg)
        if len(full) > len(emitted):
            suffix = full[len(emitted) :]
            if suffix:
                yield suffix
            emitted = full


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
    llm_num_ctx: int | None = None,
) -> dict[str, Any]:
    o = cfg.ollama
    snap: dict[str, Any] = {
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
    if llm_num_ctx is not None:
        snap["llm_num_ctx"] = int(llm_num_ctx)
    return snap
