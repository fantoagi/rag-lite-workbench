from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llama_index.core.schema import NodeWithScore, TextNode

from rag_lite.config import AppConfig
from rag_lite.ingest import (
    _candidate_file_names_from_meta,
    _chroma_all_ids_paginated,
    _chroma_get_docs_metas_by_ids,
    _open_chroma_collection_info,
)


@dataclass
class RetrievalPipelineResult:
    vector_nodes: list[NodeWithScore]
    keyword_nodes: list[NodeWithScore]
    merged_nodes: list[NodeWithScore]
    final_nodes: list[NodeWithScore]
    score_kind: str
    excluded_candidate_count: int = 0
    vector_error: str = ""
    retrieval_degraded: bool = False
    requested_retrieval_mode: str = "hybrid"


_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_BM25_CACHE: dict[tuple[str, str, int], dict[str, Any]] = {}


def _normalize_text(text: Any) -> str:
    return unicodedata.normalize("NFKC", str(text or "")).casefold()


def tokenize_for_keyword_search(text: Any) -> list[str]:
    normalized = _normalize_text(text)
    raw = [m.group(0) for m in _TOKEN_RE.finditer(normalized)]
    out: list[str] = []
    cjk_run: list[str] = []

    def flush_cjk() -> None:
        if not cjk_run:
            return
        out.extend(cjk_run)
        out.extend("".join(cjk_run[i : i + 2]) for i in range(len(cjk_run) - 1))
        out.extend("".join(cjk_run[i : i + 3]) for i in range(len(cjk_run) - 2))
        cjk_run.clear()

    for tok in raw:
        if _CJK_RE.fullmatch(tok):
            cjk_run.append(tok)
            continue
        flush_cjk()
        if len(tok) >= 2:
            out.append(tok)
    flush_cjk()
    return out


def _node_id(nws: NodeWithScore) -> str:
    return str(getattr(nws.node, "node_id", "") or getattr(nws.node, "id_", "") or "")


def _file_name_from_meta(meta: dict[str, Any]) -> str:
    for cand in _candidate_file_names_from_meta(meta):
        if str(cand or "").strip():
            return Path(str(cand)).name
    return "unknown"


def _collection_cache_key(cfg: AppConfig, index: Any | None) -> tuple[str, str, int] | None:
    info = _open_chroma_collection_info(cfg, index=index, prefer_index=True)
    coll = info.get("collection")
    if coll is None:
        return None
    active = str(info.get("active_chroma_dir") or "")
    try:
        count = int(coll.count())
    except Exception:
        count = -1
    return active, str(info.get("source") or "none"), count


def _load_keyword_corpus(cfg: AppConfig, index: Any | None = None) -> dict[str, Any]:
    key = _collection_cache_key(cfg, index)
    if key is not None and key in _BM25_CACHE:
        return _BM25_CACHE[key]

    info = _open_chroma_collection_info(cfg, index=index, prefer_index=True)
    coll = info.get("collection")
    records: list[dict[str, Any]] = []
    doc_freq: dict[str, int] = {}
    total_len = 0
    if coll is not None:
        ids = _chroma_all_ids_paginated(coll)
        for node_id, doc, meta in _chroma_get_docs_metas_by_ids(coll, ids, batch_size=96):
            text = str(doc or "")
            tokens = tokenize_for_keyword_search(text)
            if not tokens:
                continue
            tf: dict[str, int] = {}
            for tok in tokens:
                tf[tok] = tf.get(tok, 0) + 1
            for tok in tf:
                doc_freq[tok] = doc_freq.get(tok, 0) + 1
            total_len += len(tokens)
            m = meta if isinstance(meta, dict) else {}
            records.append(
                {
                    "id": str(node_id),
                    "text": text,
                    "meta": m,
                    "tf": tf,
                    "length": len(tokens),
                    "file_name": _file_name_from_meta(m),
                }
            )

    corpus = {
        "records": records,
        "doc_freq": doc_freq,
        "avg_len": (total_len / len(records)) if records else 0.0,
        "collection_source": str(info.get("source") or "none"),
        "collection_active_dir": str(info.get("active_chroma_dir") or ""),
    }
    if key is not None:
        _BM25_CACHE[key] = corpus
    return corpus


def clear_keyword_cache() -> None:
    _BM25_CACHE.clear()


def keyword_retrieve(
    cfg: AppConfig,
    index: Any,
    query: str,
    top_n: int,
    *,
    excluded_files: list[str] | tuple[str, ...] | None = None,
) -> list[NodeWithScore]:
    q_tokens = tokenize_for_keyword_search(query)
    if not q_tokens:
        return []
    corpus = _load_keyword_corpus(cfg, index=index)
    records = list(corpus.get("records") or [])
    if not records:
        return []

    excluded = {
        unicodedata.normalize("NFKC", Path(str(x or "")).name).casefold().strip()
        for x in (excluded_files or [])
        if str(x or "").strip()
    }
    n_docs = len(records)
    avg_len = float(corpus.get("avg_len") or 0.0) or 1.0
    df = dict(corpus.get("doc_freq") or {})
    q_unique = list(dict.fromkeys(q_tokens))
    k1 = 1.5
    b = 0.75
    scored: list[tuple[float, dict[str, Any]]] = []
    for rec in records:
        if excluded and unicodedata.normalize("NFKC", str(rec.get("file_name") or "")).casefold().strip() in excluded:
            continue
        tf = rec.get("tf") or {}
        length = max(1, int(rec.get("length") or 1))
        score = 0.0
        for tok in q_unique:
            f = int(tf.get(tok) or 0)
            if f <= 0:
                continue
            idf = math.log(1.0 + (n_docs - int(df.get(tok, 0)) + 0.5) / (int(df.get(tok, 0)) + 0.5))
            denom = f + k1 * (1.0 - b + b * (length / avg_len))
            score += idf * (f * (k1 + 1.0)) / denom
        if score > 0:
            scored.append((score, rec))

    scored.sort(key=lambda x: x[0], reverse=True)
    out: list[NodeWithScore] = []
    for score, rec in scored[: max(1, int(top_n))]:
        md = dict(rec.get("meta") or {})
        md["rag_keyword_score"] = float(score)
        md["rag_retrieval_sources"] = ["keyword"]
        node = TextNode(id_=str(rec["id"]), text=str(rec.get("text") or ""), metadata=md)
        out.append(NodeWithScore(node=node, score=float(score)))
    return out


def merge_retrieval_candidates(
    vector_nodes: list[NodeWithScore],
    keyword_nodes: list[NodeWithScore],
) -> list[NodeWithScore]:
    by_id: dict[str, NodeWithScore] = {}
    rank_scores: dict[str, float] = {}

    def add(nws: NodeWithScore, source: str, rank: int) -> None:
        nid = _node_id(nws)
        if not nid:
            return
        md = dict(nws.node.metadata or {})
        sources = list(md.get("rag_retrieval_sources") or [])
        if source not in sources:
            sources.append(source)
        md["rag_retrieval_sources"] = sources
        if source == "vector" and nws.score is not None:
            md["rag_vector_score"] = float(nws.score)
        if source == "keyword" and nws.score is not None:
            md["rag_keyword_score"] = float(nws.score)
        nws.node.metadata = md
        rank_scores[nid] = rank_scores.get(nid, 0.0) + (1.0 / (60.0 + max(1, rank)))
        if nid not in by_id:
            by_id[nid] = nws
        else:
            old = by_id[nid]
            old_md = dict(old.node.metadata or {})
            old_sources = list(old_md.get("rag_retrieval_sources") or [])
            for s in sources:
                if s not in old_sources:
                    old_sources.append(s)
            old_md.update({k: v for k, v in md.items() if k.startswith("rag_")})
            old_md["rag_retrieval_sources"] = old_sources
            old.node.metadata = old_md

    for i, nws in enumerate(vector_nodes or [], start=1):
        add(nws, "vector", i)
    for i, nws in enumerate(keyword_nodes or [], start=1):
        add(nws, "keyword", i)

    merged = list(by_id.values())
    for nws in merged:
        nid = _node_id(nws)
        nws.score = float(rank_scores.get(nid, 0.0))
        md = dict(nws.node.metadata or {})
        md["rag_merged_score"] = float(nws.score)
        nws.node.metadata = md
    merged.sort(key=lambda n: float(n.score or 0.0), reverse=True)
    return merged


def retrieval_mode_label(vector_enabled: bool = True, keyword_enabled: bool = True) -> str:
    if vector_enabled and keyword_enabled:
        return "hybrid"
    if keyword_enabled:
        return "keyword"
    return "vector"
