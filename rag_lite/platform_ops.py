from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import httpx


def eval_run_report_payload(store: Any, run_id: int) -> dict[str, Any]:
    run_rows = store.list_eval_runs(limit=500)
    run = next((r for r in run_rows if int(r.get("id") or 0) == int(run_id)), None)
    if not run:
        raise ValueError(f"eval run not found: {run_id}")
    results = store.fetch_eval_case_results(run_id)
    dataset_id = int(run.get("dataset_id") or 0)
    cases = store.fetch_eval_cases(dataset_id) if dataset_id > 0 else []
    return {
        "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run": run,
        "cases": cases,
        "results": results,
    }


def write_eval_run_report(store: Any, run_id: int, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = eval_run_report_payload(store, run_id)
    run = payload.get("run") or {}
    fp = str((run.get("params") or {}).get("run_fingerprint") or "no-fingerprint")
    path = output_dir / f"eval_run_{int(run_id)}_{fp}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    md_path = output_dir / f"eval_run_{int(run_id)}_{fp}.md"
    md_path.write_text(_eval_run_report_markdown(payload), encoding="utf-8")
    try:
        write_eval_case_compare(store, run_id, output_dir, payload=payload)
    except Exception:
        # 对比表失败不阻断 JSON/MD 报告
        pass
    return path


def _tri_label(value: Any) -> str:
    if value is None:
        return "—"
    return "是" if bool(value) else "否"


def _match_label(value: Any) -> str:
    """Human-facing match label: 对上 / 未对上 / —."""
    if value is None:
        return "—"
    return "对上" if bool(value) else "未对上"


def _join_list(values: Any, sep: str = " | ") -> str:
    if not values:
        return ""
    if isinstance(values, str):
        return values.strip()
    return sep.join(str(x).strip() for x in values if str(x).strip())


def _clip_text(value: Any, limit: int = 800) -> str:
    text = str(value or "").replace("\r\n", "\n").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "…"


_ERROR_REASON_ZH: dict[str, str] = {
    "OK": "",
    "NO_RESULT": "本 RUN 无结果（可能中断）",
    "TARGET_FILE_MISS": "未召回到应命中文档",
    "TARGET_FILE_HIT_BUT_CHUNK_MISS": "文档对了，但证据片段未对上",
    "CONTEXT_MISS": "应命中文档未进入最终上下文",
    "TOPK_DROP": "应命中文档在 Top-K 截断中被丢掉",
    "RERANK_DROP": "应命中文档在重排后被丢掉",
    "CONTEXT_BUDGET_DROP": "应命中文档因上下文长度预算被丢掉",
    "FILTERED_OUT": "应命中文档被排除规则过滤",
    "EXPECTED_FILE_UNRESOLVED": "应命中文档名无法解析到知识库文件",
    "EXPECTED_FILE_AMBIGUOUS": "应命中文档名匹配到多个文件，无法唯一确定",
    "ANSWERED_WHEN_SHOULD_ABSTAIN": "题目应拒答，但模型给了答案",
    "ABSTAIN_WHEN_HAS_TARGET_CONTEXT": "已有目标上下文，但模型拒答了",
    "NON_FILE_CHUNK_MISS": "无目标文档时，证据片段未对上",
    "NON_FILE_ANSWER_MISS": "无目标文档时，答案未对上",
}


def _verdict_label(error_type: Any) -> str:
    et = str(error_type or "").strip()
    if not et or et == "NO_RESULT":
        return "无结果"
    if et == "OK":
        return "正确"
    return "错误"


def _failure_reason(result: dict[str, Any], ecd: dict[str, Any]) -> str:
    error_type = str(result.get("error_type") or "").strip()
    if not error_type or error_type == "OK":
        return ""
    base = _ERROR_REASON_ZH.get(error_type, f"未分类失败（{error_type}）")
    extras: list[str] = []
    if result.get("candidate_hit") is False:
        extras.append("候选阶段未出现目标文档")
    if result.get("context_hit") is False and result.get("candidate_hit") is True:
        extras.append("最终上下文未保留目标文档")
    if result.get("chunk_hit") is False and error_type not in {
        "TARGET_FILE_HIT_BUT_CHUNK_MISS",
        "NON_FILE_CHUNK_MISS",
    }:
        extras.append("证据片段未对上")
    if result.get("answer_hit") is False and error_type not in {
        "NON_FILE_ANSWER_MISS",
        "ANSWERED_WHEN_SHOULD_ABSTAIN",
    }:
        extras.append("答案未对上")
    if result.get("abstain_correct") is False and error_type not in {
        "ANSWERED_WHEN_SHOULD_ABSTAIN",
        "ABSTAIN_WHEN_HAS_TARGET_CONTEXT",
    }:
        extras.append("拒答行为不符")
    if ecd.get("target_file_hit_but_chunk_miss") and error_type != "TARGET_FILE_HIT_BUT_CHUNK_MISS":
        extras.append("文件已中但证据未中")
    if extras:
        return f"{base}；" + "；".join(extras)
    return base


def build_eval_case_compare_rows(
    cases: list[dict[str, Any]],
    results: list[dict[str, Any]],
    *,
    run_id: int,
) -> list[dict[str, Any]]:
    """Merge original dataset cases with one RUN's per-case results for QA review.

    Column order prioritizes human review: verdict → expected → actual → match flags.
    """
    case_by_id = {int(c.get("id") or 0): c for c in cases if int(c.get("id") or 0) > 0}
    rows: list[dict[str, Any]] = []

    def _format_hit_chunks(sources: list[Any], *, per_chunk: int = 2000, max_chunks: int = 16) -> tuple[str, str]:
        files: list[str] = []
        blocks: list[str] = []
        for i, src in enumerate(sources[:max_chunks], start=1):
            if not isinstance(src, dict):
                continue
            fname = str(src.get("file_name") or "").strip() or "—"
            files.append(fname)
            body = _clip_text(src.get("chunk") or src.get("text") or "", per_chunk)
            score = src.get("score")
            score_s = f" score={score}" if score is not None else ""
            blocks.append(f"【片段{i} | {fname}{score_s}】\n{body}" if body else f"【片段{i} | {fname}{score_s}】")
        return _join_list(files), "\n\n".join(blocks)

    def _row_from_result(ord_idx: int, result: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
        diag = result.get("diagnostics") or {}
        ecd = diag.get("eval_case_diagnostics") or {}
        if not isinstance(ecd, dict):
            ecd = {}
        chunk_diag = ecd.get("chunk_diagnostics") or {}
        if not isinstance(chunk_diag, dict):
            chunk_diag = {}
        best = chunk_diag.get("best") or {}
        if not isinstance(best, dict):
            best = {}
        sources = result.get("sources") or []
        source_files, hit_chunks = _format_hit_chunks(sources)
        error_type = str(result.get("error_type") or "")
        allow_abstain = bool(case.get("allow_abstain"))
        return {
            # —— 一眼看对错 ——
            "序号": ord_idx,
            "判定结果": _verdict_label(error_type),
            "失败原因": _failure_reason(result, ecd),
            "问题": case.get("question") or result.get("question") or "",
            "是否允许拒答": "是" if allow_abstain else "否",
            # —— 期望 vs 实际：文档 ——
            "应命中文档": _join_list(case.get("expected_file_names") or []),
            "实际召回文档": source_files,
            "文档是否对上": _match_label(result.get("context_hit") if result.get("context_hit") is not None else result.get("candidate_hit")),
            # —— 期望 vs 实际：片段 ——
            "应命中片段": case.get("expected_chunk_content") or "",
            "实际命中片段": hit_chunks,
            "片段是否对上": _match_label(result.get("chunk_hit")),
            # —— 期望 vs 实际：答案 ——
            "标准答案": case.get("expected_answer") or "",
            "实际答案": str(result.get("answer") or ""),
            "答案是否对上": _match_label(result.get("answer_hit")),
            "模型是否拒答": _tri_label(result.get("abstain_actual")),
            # —— 辅助定位 ——
            "问题类型/标签": _join_list(case.get("tags") or []),
            "备注": case.get("note") or "",
            "与期望最接近的片段": _clip_text(best.get("preview") or "", 800),
            "最接近片段相似度": best.get("similarity") if best.get("similarity") is not None else "",
            "最接近片段覆盖率": best.get("coverage") if best.get("coverage") is not None else "",
            "最接近片段来源文件": best.get("file_name") or "",
            # —— 技术字段（靠后，需要时再看）——
            "case_id": int(result.get("case_id") or 0) or "",
            "sort_index": case.get("sort_index", ""),
            "候选阶段文档命中": _match_label(result.get("candidate_hit")),
            "最终上下文文档命中": _match_label(result.get("context_hit")),
            "技术错误码": error_type if error_type and error_type != "OK" else "",
            "归因码": str(diag.get("retrieval_attribution") or ""),
            "qa_id": result.get("qa_id") if result.get("qa_id") is not None else "",
            "run_id": int(run_id),
        }

    for ord_idx, result in enumerate(results, start=1):
        case_id = int(result.get("case_id") or 0)
        case = case_by_id.get(case_id) or {}
        rows.append(_row_from_result(ord_idx, result, case))

    # Include dataset cases that somehow have no result row (incomplete RUN)
    seen = {int(r.get("case_id") or 0) for r in results}
    for case in cases:
        cid = int(case.get("id") or 0)
        if cid <= 0 or cid in seen:
            continue
        rows.append(
            {
                "序号": len(rows) + 1,
                "判定结果": "无结果",
                "失败原因": "本 RUN 无结果（可能中断）",
                "问题": case.get("question") or "",
                "是否允许拒答": "是" if bool(case.get("allow_abstain")) else "否",
                "应命中文档": _join_list(case.get("expected_file_names") or []),
                "实际召回文档": "",
                "文档是否对上": "—",
                "应命中片段": case.get("expected_chunk_content") or "",
                "实际命中片段": "",
                "片段是否对上": "—",
                "标准答案": case.get("expected_answer") or "",
                "实际答案": "",
                "答案是否对上": "—",
                "模型是否拒答": "—",
                "问题类型/标签": _join_list(case.get("tags") or []),
                "备注": case.get("note") or "",
                "与期望最接近的片段": "",
                "最接近片段相似度": "",
                "最接近片段覆盖率": "",
                "最接近片段来源文件": "",
                "case_id": cid,
                "sort_index": case.get("sort_index", ""),
                "候选阶段文档命中": "—",
                "最终上下文文档命中": "—",
                "技术错误码": "NO_RESULT",
                "归因码": "",
                "qa_id": "",
                "run_id": int(run_id),
            }
        )
    return rows


def write_eval_case_compare(
    store: Any,
    run_id: int,
    output_dir: Path,
    *,
    payload: dict[str, Any] | None = None,
) -> Path:
    """Write original-case + eval-result comparison workbook (xlsx + csv).

    Returns the ``.xlsx`` path (primary). CSV is written alongside for quick filtering.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    data = payload or eval_run_report_payload(store, run_id)
    run = data.get("run") or {}
    cases = list(data.get("cases") or [])
    results = list(data.get("results") or [])
    fp = str((run.get("params") or {}).get("run_fingerprint") or "no-fingerprint")
    rows = build_eval_case_compare_rows(cases, results, run_id=int(run_id))
    stem = f"eval_run_{int(run_id)}_{fp}_case_compare"
    xlsx_path = output_dir / f"{stem}.xlsx"
    csv_path = output_dir / f"{stem}.csv"

    fieldnames = list(rows[0].keys()) if rows else [
        "序号",
        "判定结果",
        "失败原因",
        "问题",
        "应命中文档",
        "实际召回文档",
        "应命中片段",
        "实际命中片段",
        "标准答案",
        "实际答案",
        "run_id",
    ]

    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    try:
        import pandas as pd

        summary = run.get("summary") or {}
        params = run.get("params") or {}
        verdict_counts: dict[str, int] = {}
        for row in rows:
            v = str(row.get("判定结果") or "—")
            verdict_counts[v] = verdict_counts.get(v, 0) + 1
        meta_rows = [
            {"项目": "exported_at", "值": time.strftime("%Y-%m-%d %H:%M:%S")},
            {"项目": "run_id", "值": int(run_id)},
            {"项目": "dataset", "值": run.get("dataset_name") or ""},
            {"项目": "run_name", "值": run.get("name") or ""},
            {"项目": "fingerprint", "值": fp},
            {"项目": "正确题数", "值": verdict_counts.get("正确", 0)},
            {"项目": "错误题数", "值": verdict_counts.get("错误", 0)},
            {"项目": "无结果题数", "值": verdict_counts.get("无结果", 0)},
            {"项目": "generation_mode", "值": params.get("generation_mode") or ""},
            {"项目": "retrieval_mode", "值": params.get("retrieval_mode") or ""},
            {"项目": "top_n", "值": params.get("top_n")},
            {"项目": "top_k", "值": params.get("top_k")},
            {"项目": "answer_top_k", "值": params.get("answer_top_k")},
            {"项目": "llm_model", "值": params.get("llm_model") or ""},
            {"项目": "embed_model", "值": params.get("embed_model") or ""},
            {"项目": "chunk", "值": f"{params.get('chunk_mode')}/{params.get('chunk_size')}/{params.get('chunk_overlap')}"},
            {"项目": "case_count", "值": summary.get("case_count")},
            {"项目": "candidate_hit_rate", "值": summary.get("candidate_hit_rate")},
            {"项目": "context_hit_rate", "值": summary.get("context_hit_rate")},
            {"项目": "chunk_hit_rate", "值": summary.get("chunk_hit_rate")},
            {"项目": "answer_hit_rate", "值": summary.get("answer_hit_rate")},
            {"项目": "abstain_accuracy", "值": summary.get("abstain_accuracy")},
            {"项目": "ok_count", "值": summary.get("ok_count")},
        ]
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            pd.DataFrame(rows).to_excel(writer, sheet_name="逐题对比", index=False)
            pd.DataFrame(meta_rows).to_excel(writer, sheet_name="RUN摘要", index=False)
        return xlsx_path
    except Exception:
        return csv_path


def _eval_run_report_markdown(payload: dict[str, Any]) -> str:
    run = payload.get("run") or {}
    params = run.get("params") or {}
    summary = run.get("summary") or {}
    attrs = summary.get("attribution_counts") or {}
    attr_text = ", ".join(f"{k}:{v}" for k, v in sorted(attrs.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))[:8]) or "-"
    lines = [
        f"# Eval RUN {run.get('id')}",
        "",
        f"- Dataset: {run.get('dataset_name')}",
        f"- Name: {run.get('name')}",
        f"- Created at: {run.get('created_at')}",
        f"- Fingerprint: {params.get('run_fingerprint')}",
        f"- Retrieval mode: {params.get('retrieval_mode')}",
        f"- LLM: {params.get('llm_model')}",
        f"- Embedding: {params.get('embed_model')}",
        "",
        "## Summary",
        "",
        f"- Cases: {summary.get('case_count')}",
        f"- Candidate hit rate: {summary.get('candidate_hit_rate')}",
        f"- Context hit rate: {summary.get('context_hit_rate')}",
        f"- Chunk hit rate: {summary.get('chunk_hit_rate')}",
        f"- Answer hit rate: {summary.get('answer_hit_rate')}",
        f"- Abstain accuracy: {summary.get('abstain_accuracy')}",
        f"- OK count: {summary.get('ok_count')}",
        f"- Top attribution: {attr_text}",
        "",
        "## Failed Cases",
        "",
    ]
    for item in payload.get("results") or []:
        if str(item.get("error_type") or "") == "OK":
            continue
        diag = item.get("diagnostics") or {}
        lines.append(
            f"- Case {item.get('case_id')}: {item.get('error_type')} / "
            f"{diag.get('retrieval_attribution') or '-'} / QA#{item.get('qa_id')} / {item.get('question')}"
        )
    if lines[-1] == "":
        lines.append("- No failed cases.")
    return "\n".join(str(x) for x in lines) + "\n"


def ollama_health_snapshot(base_url: str, *, timeout_s: float = 3.0) -> dict[str, Any]:
    base = str(base_url or "http://127.0.0.1:11434").rstrip("/")
    started = time.perf_counter()
    out: dict[str, Any] = {"base_url": base, "ok": False, "latency_ms": None, "models": [], "error": ""}
    try:
        with httpx.Client(timeout=httpx.Timeout(timeout_s)) as client:
            resp = client.get(f"{base}/api/tags")
            out["status_code"] = resp.status_code
            resp.raise_for_status()
            payload = resp.json()
        out["models"] = [str(x.get("name") or "") for x in payload.get("models", []) if x.get("name")]
        out["ok"] = True
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        out["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
    return out


def ollama_health_rows(snapshot: dict[str, Any] | None) -> list[list[Any]]:
    s = snapshot or {}
    return [
        ["Base URL", s.get("base_url") or ""],
        ["OK", bool(s.get("ok"))],
        ["Latency ms", s.get("latency_ms")],
        ["Status code", s.get("status_code") or "-"],
        ["Models", " / ".join(s.get("models") or []) or "-"],
        ["Error", s.get("error") or "-"],
    ]
