from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator
import time
import unicodedata

import httpx
from ollama import Client as OllamaSdkClient

from llama_index.core import Settings
from llama_index.core.indices.vector_store.retrievers import VectorIndexRetriever
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

from rag_lite.config import AppConfig
from rag_lite.ingest import chunk_diagnostics, load_index_from_disk, resolve_active_chroma_dir, uploaded_files_snapshot
from rag_lite.eval_platform import attach_run_fingerprint
from rag_lite.retrieval import (
    RetrievalPipelineResult,
    clear_keyword_cache,
    keyword_retrieve,
    merge_retrieval_candidates,
    retrieval_mode_label,
)
from rag_lite.rerank import rerank_nodes
from rag_lite.store import ExperimentStore

_index_cache: dict[tuple[str | None, str | None], Any] = {}


def _index_cache_key(cfg: AppConfig, embed_model: str | None = None) -> tuple[str | None, str | None]:
    em = embed_model or cfg.ollama.get("embed_model")
    active_dir = resolve_active_chroma_dir(cfg, allow_legacy=True)
    active_label = None
    if active_dir is not None:
        try:
            active_label = str(active_dir.resolve())
        except Exception:
            active_label = str(active_dir)
    return em, active_label


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


def _ollama_embed_client_kwargs(o: dict[str, Any]) -> dict[str, Any]:
    read_sec = float(o.get("request_timeout", 600.0))
    conn_sec = float(o.get("connect_timeout", 60.0))
    return {
        "timeout": httpx.Timeout(
            connect=conn_sec,
            read=read_sec,
            write=max(read_sec, 120.0),
            pool=conn_sec,
        ),
    }


def _embed_model_with_http_timeouts(cfg: AppConfig, model_name: str | None):
    o = dict(cfg.ollama)
    name = str(model_name or o.get("embed_model") or "").strip()
    return OllamaEmbedding(
        model_name=name,
        base_url=str(o.get("base_url") or "http://127.0.0.1:11434"),
        ollama_additional_kwargs={},
        client_kwargs=_ollama_embed_client_kwargs(o),
    )


def _current_index_embed_model_name(index: Any, cfg: AppConfig) -> str | None:
    model = getattr(index, "_embed_model", None)
    return getattr(model, "model_name", None) or cfg.ollama.get("embed_model")


def _is_transient_ollama_embed_error(exc: Exception) -> bool:
    cur: BaseException | None = exc
    seen: set[int] = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, (httpx.ReadError, httpx.ConnectError, httpx.RemoteProtocolError)):
            return True
        msg = str(cur).lower()
        if any(k in msg for k in (
            "winerror 10054",
            "connection reset",
            "forcibly closed",
            "remote protocol error",
            "failed to connect to ollama",
            "readerror",
            "connecterror",
        )):
            return True
        cur = getattr(cur, "__cause__", None) or getattr(cur, "__context__", None)
    return False


def _vector_retrieve_once(index, query: str, top_n: int):
    top_n = max(1, top_n)
    retriever = VectorIndexRetriever(
        index,
        similarity_top_k=top_n,
        embed_model=getattr(index, "_embed_model", None),
        node_ids=None,
        callback_manager=getattr(index, "_callback_manager", None),
        object_map=getattr(index, "_object_map", None),
    )
    return retriever.retrieve(_normalize_retrieval_query(query))


def clear_all_index_caches() -> None:
    global _index_cache
    _index_cache.clear()
    clear_keyword_cache()
    try:
        import gc

        gc.collect()
    except Exception:
        pass


def refresh_index_cache(cfg: AppConfig, embed_model: str | None = None) -> None:
    """Reload index from disk for the current embedding model and active Chroma path."""
    global _index_cache
    key = _index_cache_key(cfg, embed_model=embed_model)
    _index_cache[key] = load_index_from_disk(cfg, embed_model_override=key[0])


def get_index(cfg: AppConfig, embed_model: str | None = None):
    global _index_cache
    key = _index_cache_key(cfg, embed_model=embed_model)
    if _index_cache.get(key) is None:
        _index_cache[key] = load_index_from_disk(cfg, embed_model_override=key[0])
    return _index_cache.get(key)


def _node_file_name(nws) -> str:
    node = nws.node
    meta = node.metadata or {}
    file_name = meta.get("file_name")
    if file_name is None and meta.get("file_path"):
        file_name = Path(str(meta["file_path"])).name
    return str(file_name or "unknown")


def _file_name_sig(name: str) -> str:
    return unicodedata.normalize("NFKC", str(Path(str(name or "")).name)).casefold().strip()


def node_to_source_dict(nws, score_kind: str) -> dict[str, Any]:
    file_name = _node_file_name(nws)
    node = nws.node
    text = node.get_content(metadata_mode="none")
    preview = text if len(text) <= 2000 else text[:2000] + "\n..."
    score = nws.score
    md_now = node.metadata
    vec_raw = md_now.get("rag_vector_score") if isinstance(md_now, dict) else None
    keyword_raw = md_now.get("rag_keyword_score") if isinstance(md_now, dict) else None
    merged_raw = md_now.get("rag_merged_score") if isinstance(md_now, dict) else None
    retrieval_sources = md_now.get("rag_retrieval_sources") if isinstance(md_now, dict) else None
    return {
        "node_id": str(getattr(node, "node_id", "") or getattr(node, "id_", "") or ""),
        "file_name": file_name,
        "chunk": preview,
        "score": float(score) if score is not None else None,
        "score_kind": score_kind,
        "vector_score": float(vec_raw) if vec_raw is not None else None,
        "keyword_score": float(keyword_raw) if keyword_raw is not None else None,
        "merged_score": float(merged_raw) if merged_raw is not None else None,
        "retrieval_sources": list(retrieval_sources) if isinstance(retrieval_sources, list) else [],
    }

def nodes_to_source_dicts(nodes, score_kind: str) -> list[dict[str, Any]]:
    return [node_to_source_dict(n, score_kind) for n in nodes]


def _node_context_part(nws, idx: int) -> str:
    meta = nws.node.metadata or {}
    name = meta.get("file_name")
    if name is None and meta.get("file_path"):
        name = Path(str(meta["file_path"])).name
    name = name or "unknown"
    body = nws.node.get_content(metadata_mode="none")
    return f"[片段 {idx}] 来源文件: {name}\n{body}"


def select_nodes_for_answer(
    cfg: AppConfig,
    query: str,
    nodes,
    *,
    llm_model: str | None = None,
    llm_num_ctx: int | None = None,
) -> tuple[list, bool]:
    """
    选择实际喂给 LLM 的节点：先按 answer_top_k（生成侧上限）截断，
    再按 max_llm_context_chars 做字符预算，避免引用展示与模型所见不一致。
    返回 (selected_nodes, truncated)。
    """
    nodes_list = list(nodes or [])
    if not nodes_list:
        return [], False
    # 生成侧 Top-K：默认可低于检索 top_k，减轻小模型串文档
    try:
        answer_top_k = int((cfg.retrieval or {}).get("answer_top_k") or 0)
    except (TypeError, ValueError):
        answer_top_k = 0
    capped_by_k = False
    if answer_top_k > 0 and len(nodes_list) > answer_top_k:
        nodes_list = nodes_list[:answer_top_k]
        capped_by_k = True
    o = dict(cfg.ollama)
    if llm_model:
        o["llm_model"] = llm_model
    if llm_num_ctx is not None:
        o["num_ctx"] = int(llm_num_ctx)
    max_c = int(o.get("max_llm_context_chars", 100_000))
    q = str(query or "").strip()
    fixed_len = len("【已知上下文】\n") + len("\n\n【用户问题】\n") + len(q)
    budget = max_c - fixed_len
    if budget <= 0:
        return [], True
    selected: list = []
    used = 0
    sep_len = len("\n\n---\n\n")
    for nws in nodes_list:
        part = _node_context_part(nws, len(selected) + 1)
        add = len(part) + (sep_len if selected else 0)
        if used + add > budget:
            break
        selected.append(nws)
        used += add
    truncated = capped_by_k or (len(selected) < len(nodes or []))
    return selected, truncated


def filter_nodes_by_excluded_files(nodes: list, excluded_files: list[str] | tuple[str, ...] | None) -> tuple[list, int]:
    excluded = {
        _file_name_sig(x)
        for x in (excluded_files or [])
        if str(x or "").strip()
    }
    if not excluded:
        return list(nodes or []), 0
    kept: list = []
    excluded_count = 0
    for nws in nodes or []:
        if _file_name_sig(_node_file_name(nws)) in excluded:
            excluded_count += 1
            continue
        kept.append(nws)
    return kept, excluded_count


def _normalize_retrieval_query(query: str) -> str:
    text = unicodedata.normalize("NFKC", str(query or "").strip())
    if not text:
        return ""
    trans = str.maketrans({
        "“": "",
        "”": "",
        "‘": "",
        "’": "",
        '"': "",
        "'": "",
        "「": "",
        "」": "",
        "『": "",
        "』": "",
        "《": "",
        "》": "",
    })
    text = text.translate(trans)
    return " ".join(text.split())


def build_anchored_eval_query(question: str, expected_file_names: list[str] | tuple[str, ...] | None) -> tuple[str, list[str]]:
    base_question = str(question or "").strip()
    anchors: list[str] = []
    seen: set[str] = set()
    for raw in expected_file_names or []:
        text = str(raw or "").strip()
        if not text:
            continue
        stem = Path(text).stem.strip() or Path(text).name.strip() or text
        norm = _normalize_retrieval_query(stem)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        anchors.append(stem)
    if not anchors:
        return base_question, []
    anchored = f"参考文档：{'；'.join(anchors)}\n问题：{base_question}" if base_question else f"参考文档：{'；'.join(anchors)}"
    return anchored, anchors


def vector_retrieve(cfg: AppConfig, index, query: str, top_n: int):
    """仅向量初筛（Top-N），供 UI 分阶段展示「检索中」进度。"""
    top_n = max(1, top_n)
    normalized_query = _normalize_retrieval_query(query)
    attempts = max(1, int(cfg.ollama.get("embed_retry_attempts", 2) or 2))
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return _vector_retrieve_once(index, normalized_query, top_n)
        except Exception as exc:
            last_exc = exc
            if not _is_transient_ollama_embed_error(exc) or attempt >= attempts:
                raise
            setattr(index, "_embed_model", _embed_model_with_http_timeouts(cfg, _current_index_embed_model_name(index, cfg)))
            time.sleep(min(1.5 * attempt, 3.0))
    if last_exc is not None:
        raise last_exc
    return []


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
    excluded_files: list[str] | tuple[str, ...] | None = None,
) -> tuple[list, str]:
    """Returns (nodes_with_score, score_kind for display)."""
    top_n = max(1, top_n)
    top_k = max(1, min(top_k, top_n))
    nodes = vector_retrieve(cfg, index, query, top_n)
    nodes, _ = filter_nodes_by_excluded_files(nodes, excluded_files)
    return apply_topk_rerank(cfg, query, nodes, top_k, use_rerank)


def hybrid_retrieve(
    cfg: AppConfig,
    index,
    query: str,
    top_n: int,
    top_k: int,
    use_rerank: bool,
    excluded_files: list[str] | tuple[str, ...] | None = None,
    *,
    keyword_top_n: int | None = None,
    vector_enabled: bool = True,
    keyword_enabled: bool = True,
) -> RetrievalPipelineResult:
    """Vector + local BM25 retrieval, de-duplicated before optional rerank."""
    top_n = max(1, int(top_n))
    top_k = max(1, int(top_k))
    keyword_n = max(1, int(keyword_top_n or top_n))
    vector_error = ""
    if vector_enabled:
        try:
            vector_nodes = vector_retrieve(cfg, index, query, top_n)
        except Exception as exc:
            vector_error = f"{type(exc).__name__}: {exc}"
            vector_nodes = []
    else:
        vector_error = "vector retrieval skipped"
        vector_nodes = []
    vector_nodes_filtered, excluded_count = filter_nodes_by_excluded_files(vector_nodes, excluded_files)
    if keyword_enabled:
        keyword_nodes = keyword_retrieve(
            cfg,
            index,
            query,
            keyword_n,
            excluded_files=excluded_files,
        )
    else:
        keyword_nodes = []
    merged_nodes = merge_retrieval_candidates(vector_nodes_filtered, keyword_nodes)
    final_nodes, kind = apply_topk_rerank(cfg, query, merged_nodes, top_k, use_rerank)
    requested_mode = retrieval_mode_label(
        vector_enabled=bool(vector_enabled),
        keyword_enabled=bool(keyword_enabled),
    )
    vector_failed = bool(
        vector_enabled
        and str(vector_error or "").strip()
        and "skipped" not in str(vector_error or "").lower()
    )
    # 向量阶段失败时不得继续标成 hybrid/rerank，避免伪成功
    if vector_failed:
        if keyword_enabled and merged_nodes:
            kind = "keyword_fallback"
        else:
            kind = "vector_error"
    elif kind == "vector":
        kind = requested_mode
    return RetrievalPipelineResult(
        vector_nodes=vector_nodes,
        keyword_nodes=keyword_nodes,
        merged_nodes=merged_nodes,
        final_nodes=final_nodes,
        score_kind=kind,
        excluded_candidate_count=excluded_count,
        vector_error=vector_error,
        retrieval_degraded=bool(vector_failed),
        requested_retrieval_mode=requested_mode,
    )


def stream_answer(
    cfg: AppConfig,
    query: str,
    system_prompt: str,
    nodes,
    llm_model: str | None = None,
    llm_num_ctx: int | None = None,
) -> Iterator[str]:
    selected_nodes, context_truncated = select_nodes_for_answer(
        cfg,
        query,
        nodes,
        llm_model=llm_model,
        llm_num_ctx=llm_num_ctx,
    )
    if not selected_nodes:
        return
    o = dict(cfg.ollama)
    if llm_model:
        o["llm_model"] = llm_model
    if llm_num_ctx is not None:
        o["num_ctx"] = int(llm_num_ctx)
    read_sec = float(o.get("request_timeout", 600.0))

    context_parts: list[str] = []
    for i, nws in enumerate(selected_nodes, 1):
        context_parts.append(_node_context_part(nws, i))
    context_str = "\n\n---\n\n".join(context_parts)

    user_content = (
        f"【已知上下文】\n{context_str}\n\n"
        f"【用户问题】\n{query.strip()}"
    )
    if context_truncated:
        user_content += "\n\n[系统提示：候选上下文过长，已按片段边界截断尾部。]"
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


def _safe_manifest(cfg: AppConfig) -> dict[str, Any] | None:
    try:
        return ExperimentStore(cfg.sqlite_path).get_index_manifest()
    except Exception:
        return None


def _safe_active_chroma_label(cfg: AppConfig) -> str | None:
    active_dir = resolve_active_chroma_dir(cfg, allow_legacy=True)
    if active_dir is None:
        return None
    try:
        return str(active_dir.resolve())
    except Exception:
        return str(active_dir)


def _norm_path_for_compare(path_text: str | None) -> str:
    t = str(path_text or "").strip()
    if not t:
        return ""
    t = t.replace("\\", "/")
    while "//" in t:
        t = t.replace("//", "/")
    return t.casefold()


def _build_index_snapshot(cfg: AppConfig) -> dict[str, Any]:
    manifest = _safe_manifest(cfg) or {}
    uploads = uploaded_files_snapshot(cfg)
    active_label = _safe_active_chroma_label(cfg)
    manifest_active = str(manifest.get("active_chroma_subdir") or "").strip() or None
    manifest_files = {
        str(x.get("name") or "").strip(): x
        for x in (manifest.get("files") or [])
        if isinstance(x, dict) and str(x.get("name") or "").strip()
    }
    uploads_changed = False
    for item in uploads:
        name = str(item.get("name") or "").strip()
        prev = manifest_files.get(name)
        if prev is None:
            uploads_changed = True
            break
        if int(prev.get("size", -1)) != int(item.get("size", -2)) or int(prev.get("mtime_ns", -1)) != int(item.get("mtime_ns", -2)):
            uploads_changed = True
            break
    if not uploads_changed:
        for name in manifest_files:
            if not any(str(x.get("name") or "").strip() == name for x in uploads):
                uploads_changed = True
                break
    active_norm = _norm_path_for_compare(active_label)
    manifest_norm = _norm_path_for_compare(manifest_active)
    active_matches_manifest = bool(
        manifest_norm
        and active_norm
        and (active_norm == manifest_norm or active_norm.endswith(manifest_norm))
    )
    readiness = dict(manifest.get("readiness") or {})
    final_health = dict(readiness.get("final_health") or {}) if isinstance(readiness.get("final_health"), dict) else {}
    diag_summary: dict[str, Any] = {}
    try:
        diag = chunk_diagnostics(cfg, index=None, prefer_index=True)
        diag_summary = dict(diag.get("summary") or {})
    except Exception:
        diag_summary = {}
    missing_uploaded = list(readiness.get("missing_uploaded_files") or final_health.get("missing_uploaded_files") or [])
    readiness_healthy = bool((readiness.get("ok") or final_health.get("ok")) and readiness.get("diagnostics_ok", True))
    return {
        "active": {
            "chroma_dir": active_label,
        },
        "manifest": {
            "build_id": str(manifest.get("build_id") or "").strip() or None,
            "active_chroma_subdir": manifest_active,
            "activated_at": str(manifest.get("activated_at") or "").strip() or None,
            "built_at": str(manifest.get("built_at") or "").strip() or None,
            "embed_model": str(manifest.get("embed_model") or "").strip() or None,
            "chunk_mode": str(manifest.get("chunk_mode") or "").strip() or None,
            "chunk_size": manifest.get("chunk_size"),
            "chunk_overlap": manifest.get("chunk_overlap"),
        },
        "readiness": readiness,
        "consistency": {
            "uploads_changed_since_manifest": uploads_changed,
            "active_dir_matches_manifest": active_matches_manifest,
            "readiness_healthy": readiness_healthy,
            "missing_uploaded_files_count": len(missing_uploaded),
            "missing_uploaded_files_preview": missing_uploaded[:10],
            "diagnostics_total_files": int(
                diag_summary.get("total_files")
                or readiness.get("diagnostics_total_files")
                or final_health.get("diagnostics_total_files")
                or 0
            ),
            "diagnostics_total_chunks": int(
                diag_summary.get("total_chunks")
                or readiness.get("diagnostics_total_chunks")
                or final_health.get("diagnostics_total_chunks")
                or 0
            ),
            "uploads_file_count": len(uploads),
            "manifest_file_count": len(manifest_files),
        },
    }


def resolve_index_chunk_params(
    cfg: AppConfig,
    ui_chunk_size: int | None = None,
    ui_chunk_overlap: int | None = None,
    ui_chunk_mode: str | None = None,
) -> dict[str, Any]:
    """以活跃索引 manifest 的切片参数为准；UI 仅作对照。

    检索读的是已构建索引，UI 改 chunk 不会改变召回结果。
    """
    index_snapshot = _build_index_snapshot(cfg)
    manifest = dict(index_snapshot.get("manifest") or {})
    def _parse_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    ui_mode_raw = str(ui_chunk_mode or "").strip().lower() or None
    ui_size = _parse_int(ui_chunk_size)
    ui_overlap = _parse_int(ui_chunk_overlap)
    # None/<=0 的 size 视为未指定（参数网格占位）；overlap 仅 None 视为未指定
    ui_size_set = ui_size is not None and ui_size > 0
    ui_overlap_set = ui_overlap is not None
    ui_mode_set = bool(ui_mode_raw)

    index_mode = str(manifest.get("chunk_mode") or "").strip().lower() or None
    index_size_i = _parse_int(manifest.get("chunk_size"))
    if index_size_i is not None and index_size_i <= 0:
        index_size_i = None
    index_overlap_i = _parse_int(manifest.get("chunk_overlap"))

    effective_mode = index_mode or (ui_mode_raw if ui_mode_set else "sentence")
    if index_size_i is not None:
        effective_size = int(index_size_i)
    elif ui_size_set:
        effective_size = int(ui_size)
    else:
        effective_size = 512
    if index_overlap_i is not None:
        effective_overlap = int(index_overlap_i)
    elif ui_overlap_set:
        effective_overlap = int(ui_overlap)
    else:
        effective_overlap = 64

    mismatch = False
    if index_mode and ui_mode_set and index_mode != ui_mode_raw:
        mismatch = True
    if index_size_i is not None and ui_size_set and int(index_size_i) != int(ui_size):
        mismatch = True
    if index_overlap_i is not None and ui_overlap_set and int(index_overlap_i) != int(ui_overlap):
        mismatch = True

    warning = ""
    if mismatch:
        warning = (
            "UI 切分参数与当前索引不一致；本次 RUN 以索引 manifest 为准"
            f"（index={effective_mode}/{effective_size}/{effective_overlap}，"
            f"ui={ui_mode_raw}/{ui_size if ui_size_set else '—'}/"
            f"{ui_overlap if ui_overlap_set else '—'}）。改切片需重建索引。"
        )
    return {
        "chunk_mode": effective_mode,
        "chunk_size": int(effective_size),
        "chunk_overlap": int(effective_overlap),
        "ui_chunk_mode": ui_mode_raw,
        "ui_chunk_size": ui_size if ui_size_set else None,
        "ui_chunk_overlap": ui_overlap if ui_overlap_set else None,
        "index_chunk_mode": index_mode,
        "index_chunk_size": index_size_i,
        "index_chunk_overlap": index_overlap_i,
        "chunk_params_mismatch": bool(mismatch),
        "chunk_params_warning": warning,
        "index_snapshot": index_snapshot,
    }


def build_params_snapshot(
    cfg: AppConfig,
    chunk_size: int | None,
    chunk_overlap: int | None,
    top_n: int,
    top_k: int,
    use_rerank: bool,
    chunk_mode: str | None = "sentence",
    llm_model: str | None = None,
    embed_model: str | None = None,
    llm_num_ctx: int | None = None,
    excluded_files: list[str] | tuple[str, ...] | None = None,
    excluded_doc_classes: list[str] | tuple[str, ...] | None = None,
    include_zero_chunk: bool | None = None,
    query_anchoring_enabled: bool | None = None,
    query_anchoring_source: str | None = None,
    vector_enabled: bool = True,
    keyword_enabled: bool = True,
    generation_mode: str | None = None,
    prefer_index_chunk_params: bool = True,
    retrieval_degraded: bool | None = None,
    vector_error: str | None = None,
) -> dict[str, Any]:
    o = cfg.ollama
    chunk_info = resolve_index_chunk_params(cfg, chunk_size, chunk_overlap, chunk_mode)
    if prefer_index_chunk_params:
        eff_size = int(chunk_info["chunk_size"])
        eff_overlap = int(chunk_info["chunk_overlap"])
        eff_mode = str(chunk_info["chunk_mode"])
    else:
        try:
            eff_size = int(chunk_size) if chunk_size is not None else 512
        except (TypeError, ValueError):
            eff_size = 512
        try:
            eff_overlap = int(chunk_overlap) if chunk_overlap is not None else 64
        except (TypeError, ValueError):
            eff_overlap = 64
        eff_mode = str(chunk_mode or "sentence").strip().lower() or "sentence"
    snap: dict[str, Any] = {
        "chunk_size": eff_size,
        "chunk_overlap": eff_overlap,
        "chunk_mode": eff_mode,
        "ui_chunk_size": chunk_info.get("ui_chunk_size"),
        "ui_chunk_overlap": chunk_info.get("ui_chunk_overlap"),
        "ui_chunk_mode": chunk_info.get("ui_chunk_mode"),
        "chunk_params_mismatch": bool(chunk_info.get("chunk_params_mismatch")),
        "top_n": top_n,
        "top_k": top_k,
        "use_rerank": use_rerank,
        "generation_mode": generation_mode or "llm",
        "retrieval_mode": retrieval_mode_label(vector_enabled=bool(vector_enabled), keyword_enabled=bool(keyword_enabled)),
        "keyword_top_n": top_n,
        "vector_enabled": bool(vector_enabled),
        "keyword_enabled": bool(keyword_enabled),
        "prompt_version": cfg.prompt.get("version", "v1"),
        "llm_model": llm_model or o.get("llm_model"),
        "embed_model": embed_model or o.get("embed_model"),
        "rerank_model": cfg.rerank.get("model_name") if use_rerank else None,
    }
    if retrieval_degraded is not None:
        snap["retrieval_degraded"] = bool(retrieval_degraded)
    if vector_error is not None:
        snap["vector_error"] = str(vector_error or "")
    if llm_num_ctx is not None:
        snap["llm_num_ctx"] = int(llm_num_ctx)
    if query_anchoring_enabled is not None:
        snap["query_anchoring_enabled"] = bool(query_anchoring_enabled)
    if query_anchoring_source is not None:
        snap["query_anchoring_source"] = str(query_anchoring_source or "").strip() or None
    if excluded_doc_classes is not None:
        snap["excluded_doc_classes"] = [str(x) for x in excluded_doc_classes if str(x or "").strip()]
    if include_zero_chunk is not None:
        snap["include_zero_chunk"] = bool(include_zero_chunk)
    if excluded_files is not None:
        cleaned = [str(x) for x in excluded_files if str(x or "").strip()]
        snap["excluded_files"] = cleaned
        snap["excluded_file_count"] = len(cleaned)
        snap["excluded_file_preview"] = cleaned[:10]
    snap["index_snapshot"] = chunk_info.get("index_snapshot") or _build_index_snapshot(cfg)
    if chunk_info.get("chunk_params_warning"):
        snap["chunk_params_warning"] = chunk_info["chunk_params_warning"]
    return attach_run_fingerprint(snap)
