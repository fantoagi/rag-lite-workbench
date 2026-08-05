from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any, Callable


def stable_fingerprint(payload: dict[str, Any], *, length: int = 16) -> str:
    """Create a stable short fingerprint for experiment comparison."""
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[: max(8, int(length))]


def attach_run_fingerprint(params: dict[str, Any]) -> dict[str, Any]:
    out = dict(params or {})
    index_snapshot = dict(out.get("index_snapshot") or {})
    manifest = dict(index_snapshot.get("manifest") or {})
    active = dict(index_snapshot.get("active") or {})
    consistency = dict(index_snapshot.get("consistency") or {})
    fingerprint_payload = {
        "chunk_size": out.get("chunk_size"),
        "chunk_overlap": out.get("chunk_overlap"),
        "chunk_mode": out.get("chunk_mode"),
        "top_n": out.get("top_n"),
        "top_k": out.get("top_k"),
        "use_rerank": out.get("use_rerank"),
        "generation_mode": out.get("generation_mode"),
        "retrieval_mode": out.get("retrieval_mode"),
        "vector_enabled": out.get("vector_enabled"),
        "keyword_top_n": out.get("keyword_top_n"),
        "prompt_version": out.get("prompt_version"),
        "llm_model": out.get("llm_model"),
        "embed_model": out.get("embed_model"),
        "rerank_model": out.get("rerank_model"),
        "llm_num_ctx": out.get("llm_num_ctx"),
        "query_anchoring_enabled": out.get("query_anchoring_enabled"),
        "query_anchoring_source": out.get("query_anchoring_source"),
        "excluded_doc_classes": out.get("excluded_doc_classes"),
        "include_zero_chunk": out.get("include_zero_chunk"),
        "excluded_file_count": out.get("excluded_file_count"),
        "retrieval_degraded": out.get("retrieval_degraded"),
        "index_build_id": manifest.get("build_id"),
        "index_active_dir": active.get("chroma_dir"),
        "index_active_subdir": manifest.get("active_chroma_subdir"),
        "index_embed_model": manifest.get("embed_model"),
        "index_chunk_mode": manifest.get("chunk_mode"),
        "index_chunk_size": manifest.get("chunk_size"),
        "index_chunk_overlap": manifest.get("chunk_overlap"),
        "index_file_count": consistency.get("manifest_file_count"),
        "index_chunk_count": consistency.get("diagnostics_total_chunks"),
    }
    out["run_fingerprint"] = stable_fingerprint(fingerprint_payload)
    out["run_fingerprint_payload"] = fingerprint_payload
    return out


def dataset_quality_summary(
    cases: list[dict[str, Any]],
    *,
    resolve_expected_file_details: Callable[[list[Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    total = len(cases or [])
    issue_counts: Counter[str] = Counter()
    tag_counts: Counter[str] = Counter()
    examples: dict[str, list[str]] = {}

    def add_issue(code: str, question: str) -> None:
        issue_counts[code] += 1
        examples.setdefault(code, [])
        if len(examples[code]) < 5:
            examples[code].append(str(question or "")[:120])

    for case in cases or []:
        question = str(case.get("question") or "")
        expected_files = [str(x).strip() for x in (case.get("expected_file_names") or []) if str(x).strip()]
        expected_answer = str(case.get("expected_answer") or "").strip()
        expected_chunk = str(case.get("expected_chunk_content") or "").strip()
        keywords = [str(x).strip() for x in (case.get("expected_answer_keywords") or []) if str(x).strip()]
        tags = [str(x).strip() for x in (case.get("tags") or []) if str(x).strip()]
        for tag in tags:
            tag_counts[tag] += 1
        if not question.strip():
            add_issue("EMPTY_QUESTION", question)
        if not expected_files:
            add_issue("NO_EXPECTED_FILE", question)
        elif resolve_expected_file_details is not None:
            info = resolve_expected_file_details(expected_files)
            if info.get("has_unresolved"):
                add_issue("EXPECTED_FILE_UNRESOLVED", question)
            if info.get("has_ambiguous"):
                add_issue("EXPECTED_FILE_AMBIGUOUS", question)
        if not expected_chunk:
            add_issue("NO_EXPECTED_CHUNK", question)
        if not expected_answer and not keywords and not bool(case.get("allow_abstain")):
            add_issue("NO_ANSWER_OR_KEYWORDS", question)
        if bool(case.get("allow_abstain")) and (expected_files or expected_chunk):
            add_issue("ABSTAIN_WITH_TARGET_CONTEXT", question)
        if not tags:
            add_issue("NO_TAG", question)

    return {
        "case_count": total,
        "file_expected_count": sum(1 for x in cases or [] if x.get("expected_file_names")),
        "chunk_expected_count": sum(1 for x in cases or [] if str(x.get("expected_chunk_content") or "").strip()),
        "keyword_expected_count": sum(1 for x in cases or [] if x.get("expected_answer_keywords")),
        "abstain_count": sum(1 for x in cases or [] if bool(x.get("allow_abstain"))),
        "issue_counts": dict(issue_counts),
        "tag_counts": dict(tag_counts),
        "issue_examples": examples,
    }


def dataset_quality_rows(summary: dict[str, Any] | None) -> list[list[Any]]:
    s = summary or {}
    total = int(s.get("case_count") or 0)
    rows = [
        ["CASE_COUNT", total, "100.0%" if total else "-"],
        ["FILE_EXPECTED", int(s.get("file_expected_count") or 0), _pct(s.get("file_expected_count"), total)],
        ["CHUNK_EXPECTED", int(s.get("chunk_expected_count") or 0), _pct(s.get("chunk_expected_count"), total)],
        ["KEYWORD_EXPECTED", int(s.get("keyword_expected_count") or 0), _pct(s.get("keyword_expected_count"), total)],
        ["ABSTAIN", int(s.get("abstain_count") or 0), _pct(s.get("abstain_count"), total)],
    ]
    for code, count in sorted((s.get("issue_counts") or {}).items(), key=lambda kv: (-int(kv[1]), str(kv[0]))):
        rows.append([str(code), int(count), _pct(count, total)])
    return rows or [["NO_DATA", 0, "-"]]


def attribution_code(
    *,
    diagnostics: dict[str, Any] | None,
    judge: dict[str, Any] | None = None,
    error_type: str | None = None,
) -> str:
    diag = diagnostics or {}
    j = judge or {}
    err = str(error_type or j.get("error_type") or "").strip()
    eval_diag = dict(diag.get("eval_case_diagnostics") or {})
    issue_codes = set(eval_diag.get("issue_codes") or j.get("issue_codes") or [])
    vector_candidates = list(diag.get("vector_candidates") or [])
    keyword_candidates = list(diag.get("keyword_candidates") or [])
    merged_candidates = list(diag.get("merged_candidates") or [])
    final_sources = list(diag.get("final_sources") or diag.get("final_contexts") or [])
    vector_error = str(diag.get("vector_error") or "").strip()

    if err == "OK":
        return "OK"
    if "EXPECTED_FILE_UNRESOLVED" in issue_codes:
        return "DATASET_EXPECTED_FILE_UNRESOLVED"
    if "EXPECTED_FILE_AMBIGUOUS" in issue_codes:
        return "DATASET_EXPECTED_FILE_AMBIGUOUS"
    if vector_error and "skipped" not in vector_error.lower():
        return "VECTOR_STAGE_ERROR"
    if err == "ANSWERED_WHEN_SHOULD_ABSTAIN":
        return "ABSTAIN_POLICY_MISS"
    if err == "ABSTAIN_WHEN_HAS_TARGET_CONTEXT":
        return "OVER_ABSTAIN_WITH_TARGET"
    if not merged_candidates:
        return "NO_RETRIEVAL_CANDIDATES"
    if err == "TARGET_FILE_MISS":
        if keyword_candidates and not vector_candidates:
            return "KEYWORD_ONLY_TARGET_MISS"
        if vector_candidates and not keyword_candidates:
            return "VECTOR_ONLY_TARGET_MISS"
        return "HYBRID_TARGET_MISS"
    if err == "FILTERED_OUT":
        return "TARGET_FILE_FILTERED_OUT"
    if err == "TOPK_DROP":
        return "TOPK_DROPPED_TARGET"
    if err == "RERANK_DROP":
        return "TOPK_OR_RERANK_DROPPED_TARGET"
    if err == "CONTEXT_BUDGET_DROP":
        return "CONTEXT_BUDGET_DROPPED_TARGET"
    if err == "CONTEXT_MISS":
        return "CONTEXT_SELECTION_MISS"
    if err == "TARGET_FILE_HIT_BUT_CHUNK_MISS":
        return "CHUNK_MISS_AFTER_FILE_HIT"
    if err in ("NON_FILE_CHUNK_MISS", "NON_FILE_ANSWER_MISS"):
        return err
    if final_sources and not diag.get("context_truncated") and err:
        return "ANSWER_OR_JUDGE_MISS"
    return err or "UNKNOWN"


def attribution_rows(results: list[dict[str, Any]]) -> list[list[Any]]:
    counts: Counter[str] = Counter()
    total = len(results or [])
    for item in results or []:
        counts[attribution_code(diagnostics=item.get("diagnostics"), error_type=item.get("error_type"))] += 1
    if not counts:
        return [["ATTRIBUTION:NO_DATA", 0, "-"]]
    return [[f"ATTRIBUTION:{code}", int(count), _pct(count, total)] for code, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]


def _pct(value: Any, total: int) -> str:
    if not total:
        return "-"
    return f"{(int(value or 0) / total) * 100:.1f}%"
