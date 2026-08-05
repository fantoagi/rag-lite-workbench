from __future__ import annotations

from typing import Any


RETRIEVAL_ONLY_SENTINELS = {"[retrieval-only]", "retrieval-only", "retrieval_only"}


def normalize_generation_mode(value: Any = None, llm_model: Any = None) -> str:
    raw = str(value or "").strip().lower()
    model = str(llm_model or "").strip().lower()
    if raw in {"retrieval-only", "retrieval_only", "retrieval only"}:
        return "retrieval_only"
    if model in RETRIEVAL_ONLY_SENTINELS:
        return "retrieval_only"
    return "llm"


def normalize_retrieval_mode(value: Any = None) -> str:
    raw = str(value or "").strip().lower().replace("-", "_")
    if raw in {"vector", "keyword", "hybrid"}:
        return raw
    if raw in {"bm25", "keyword_only"}:
        return "keyword"
    if raw in {"vector_only", "dense"}:
        return "vector"
    return "hybrid"


def retrieval_flags(mode: Any = None) -> tuple[bool, bool]:
    normalized = normalize_retrieval_mode(mode)
    if normalized == "vector":
        return True, False
    if normalized == "keyword":
        return False, True
    return True, True


def effective_llm_model(llm_model: Any, generation_mode: Any = None) -> str | None:
    if normalize_generation_mode(generation_mode, llm_model) == "retrieval_only":
        return None
    text = str(llm_model or "").strip()
    return text or None

