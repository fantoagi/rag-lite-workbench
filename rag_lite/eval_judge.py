# -*- coding: utf-8 -*-
"""评估判定逻辑（抽出与参数化）。

历史背景：原判定函数 (`_text_match_loose` / `_chunk_match_metrics` /
`_chunk_match_diagnostics` / `_evaluate_case_result`) 散落在 ``main.py``，
启发式阈值（ratio>=0.6、common>=24、ratio>=0.35、substring_step=8 等）以字面量形式
硬编码；CJK 场景下尤其脆弱。本模块承担两件事：

1. 把现有判定流程 1:1 抽到独立函数，保持默认行为完全一致（向后兼容）；
2. 暴露 ``EvalJudgeConfig`` 以便调用方按场景调阈值 / 开关 CJK bigram 二次校验；
   同时新增 ``ABSTAIN_WHEN_HAS_TARGET_CONTEXT`` 错误类型（与
   ``ANSWERED_WHEN_SHOULD_ABSTAIN`` 对偶），用于在 ``allow_abstain=false``
   但模型仍给出拒答文案时给出可定位的归因。

``main.py`` 通过薄壳函数复用本模块，所有现有调用方无须修改。
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

# 拒答文案嗅探子串（与历史 ``_is_abstain_answer`` 一致；保留以避免静默改变行为）
_ABSTAIN_PHRASES: tuple[str, ...] = (
    "\u65e0\u6cd5\u56de\u7b54",  # 无法回答
    "\u672a\u68c0\u7d22\u5230\u4efb\u4f55\u6587\u6863\u7247\u6bb5",  # 未检索到任何文档片段
)

# 系统哨兵：评测路径写入，避免空检索/跳过生成被误判为模型拒答
ANSWER_NO_CONTEXT = "[NO_CONTEXT]"
ANSWER_RETRIEVAL_ONLY = "[retrieval-only]"
_SYSTEM_ANSWER_SENTINELS: frozenset[str] = frozenset({ANSWER_NO_CONTEXT, ANSWER_RETRIEVAL_ONLY})

# 新增反向拒答错误类型（与 ANSWERED_WHEN_SHOULD_ABSTAIN 对偶）
ERROR_ABSTAIN_WHEN_HAS_TARGET_CONTEXT = "ABSTAIN_WHEN_HAS_TARGET_CONTEXT"
ERROR_ANSWERED_WHEN_SHOULD_ABSTAIN = "ANSWERED_WHEN_SHOULD_ABSTAIN"

_ISSUE_ABSTAIN_WHEN_HAS_TARGET_CONTEXT = "ABSTAIN_WHEN_HAS_TARGET_CONTEXT"
_ISSUE_ANSWERED_WHEN_SHOULD_ABSTAIN = "ANSWERED_WHEN_SHOULD_ABSTAIN"


def is_abstain_answer(answer: Any) -> bool:
    """判定 LLM 输出是否属于拒答文案。

    历史口径：含 ``无法回答`` 或 ``未检索到任何文档片段`` 任一子串。
    保留为公共函数以便测试与复用。
    """
    s = str(answer or "")
    return any(phrase in s for phrase in _ABSTAIN_PHRASES)


@dataclass
class EvalJudgeConfig:
    """评估判定的可调阈值集合。

    默认面向金融中文语料收紧：提高 ratio 门槛，并对短 CJK 启用 bigram 门禁，
    避免「招商证券」≈「浙商证券」一类假阳性污染 chunk_hit。
    如需旧版宽松口径，使用 ``EvalJudgeConfig.legacy()``。
    """

    # difflib SequenceMatcher 相似度阈值（>= 即视为命中）
    chunk_match_ratio: float = 0.85
    # 当 difflib ratio 较低时，启用滑动窗口公共子串作为兜底；要求
    # 公共子串长度 >= chunk_common_substring_min 且 difflib ratio >= chunk_common_substring_ratio
    chunk_common_substring_min: int = 24
    chunk_common_substring_ratio: float = 0.45
    chunk_substring_step: int = 8
    chunk_substring_len: int = 24
    # 比较时的最大字符数（防止 50MB 文档卡住 difflib）
    max_compare_chars: int = 4000
    # 公共子串覆盖率（LCS / len(a)）的统计上限
    max_lcs_chars: int = 5000
    # CJK 场景：两侧都达到该长度且含 CJK 时，ratio/公共子串命中还需通过 bigram 门禁
    cjk_bigram_min_chars: int = 4
    cjk_bigram_min_overlap: float = 0.6
    # True：CJK bigram 作为门禁（不通过则拒绝 ratio/公共子串假阳性）
    # False：仅作为额外命中通道（历史行为）
    cjk_bigram_as_gate: bool = True
    # 允许通过的最长 difflib ratio 上限（仅用于分析；不影响命中判定）
    ratio_for_diagnostics: bool = True

    @classmethod
    def legacy(cls) -> "EvalJudgeConfig":
        """历史宽松阈值（ratio>=0.6，bigram 仅作加分项）。"""
        return cls(
            chunk_match_ratio=0.6,
            chunk_common_substring_ratio=0.35,
            cjk_bigram_min_chars=12,
            cjk_bigram_min_overlap=0.5,
            cjk_bigram_as_gate=False,
        )


def _is_cjk_char(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff"


def _cjk_bigrams(text: str) -> set[str]:
    """提取 CJK bigram（连续两字），保留 ASCII 段以兼容中英混排。"""
    if not text:
        return set()
    out: set[str] = set()
    for i in range(len(text) - 1):
        a, b = text[i], text[i + 1]
        if _is_cjk_char(a) and _is_cjk_char(b):
            out.add(a + b)
    return out


def _normalize_for_judge(text: Any) -> str:
    """与 ``_normalize_eval_text`` 等价：去空白、转小写。

    提供独立的实现以避免 ``main.py`` 依赖被引入纯算法模块。
    """
    s = str(text or "").strip().lower()
    return "".join(ch for ch in s if not ch.isspace())


def _longest_common_substring_len(a: str, b: str) -> int:
    """朴素 O(len(a)*len(b)) DP；输入均已由调用方限制为 <= 5000 字符。"""
    if not a or not b:
        return 0
    if len(a) > len(b):
        a, b = b, a
    prev = [0] * (len(a) + 1)
    best = 0
    for ch_b in b:
        cur = [0] * (len(a) + 1)
        for i, ch_a in enumerate(a, start=1):
            if ch_a == ch_b:
                cur[i] = prev[i - 1] + 1
                if cur[i] > best:
                    best = cur[i]
        prev = cur
    return best


def _common_substring_sampled(a: str, b: str, config: EvalJudgeConfig) -> int:
    """在 a 上按 ``chunk_substring_step`` 抽样 ``chunk_substring_len`` 长窗口，
    返回 b 中出现的最长公共子串长度（用于 ratio 较低时的兜底判定）。
    """
    if not a or not b:
        return 0
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    step = max(1, int(config.chunk_substring_step))
    seg_len = max(1, int(config.chunk_substring_len))
    common = 0
    for i in range(0, max(1, len(short) - seg_len + 1), step):
        seg = short[i : i + seg_len]
        if seg and seg in long:
            common = max(common, len(seg))
    return common


def _preview_chunk_text(raw: Any, max_chars: int = 160) -> str:
    t = " ".join(str(raw or "").split())
    if len(t) > max_chars:
        return t[: max_chars - 1] + "\u2026"
    return t


def _cjk_bigram_jaccard(a: str, b: str) -> float | None:
    ba, bb = _cjk_bigrams(a), _cjk_bigrams(b)
    if not ba or not bb:
        return None
    return len(ba & bb) / max(1, len(ba | bb))


def _cjk_bigram_gate_ok(a: str, b: str, cfg: EvalJudgeConfig) -> bool | None:
    """CJK bigram 门禁：True 通过 / False 拒绝 / None 不适用。

    对「短期望 vs 长检索片段」用滑动窗口取最大 Jaccard，避免长文本稀释分母
    导致「几乎整段命中」仍被拒；等长近音词（招商/浙商）仍按全串 Jaccard 拒绝。
    """
    if len(a) < cfg.cjk_bigram_min_chars or len(b) < cfg.cjk_bigram_min_chars:
        return None
    threshold = float(cfg.cjk_bigram_min_overlap)
    jaccard = _cjk_bigram_jaccard(a, b)
    if jaccard is not None and jaccard >= threshold:
        return True
    # 期望明显短于实际时，在实际文本上滑窗再比一次
    if len(b) >= max(len(a) + 8, int(len(a) * 1.5)):
        win = max(len(a), min(len(b), max(24, int(len(a) * 1.25))))
        step = max(4, win // 4)
        best = jaccard if jaccard is not None else 0.0
        for i in range(0, max(1, len(b) - win + 1), step):
            jj = _cjk_bigram_jaccard(a, b[i : i + win])
            if jj is not None and jj > best:
                best = jj
                if best >= threshold:
                    return True
        return False if jaccard is not None or best > 0 else None
    if jaccard is None:
        return None
    return False


def _split_expected_chunk_parts(text: str) -> list[str]:
    """把带省略号拼接的金标拆成可独立核验的子片段。"""
    raw = str(text or "").strip()
    if not raw:
        return []
    parts = [p.strip() for p in re.split(r"(?:\.{2,}|…+)", raw) if p and p.strip()]
    return parts if parts else [raw]


def text_match_loose(
    expect_text: Any,
    actual_text: Any,
    config: EvalJudgeConfig | None = None,
) -> bool:
    """对 ``expect_text`` 与 ``actual_text`` 做宽松命中判定。

    判定规则（按 ``config`` 调整）：

    1. 任一为空 -> False
    2. 互为子串 -> True（真包含，不经 bigram 门禁）
    3. 期望在实际中的最长公共子串覆盖率很高 -> True（近包含，抗少量 OCR/标点差）
    4. difflib.SequenceMatcher.ratio >= ``chunk_match_ratio``
       （CJK 且 ``cjk_bigram_as_gate`` 时还需通过 bigram 门禁）
    5. 滑动窗口公共子串兜底（同样可受 bigram 门禁约束）
    6. 若未开启门禁：CJK bigram Jaccard 仍可作为额外命中通道
    """
    cfg = config or EvalJudgeConfig()
    a = _normalize_for_judge(expect_text)
    b = _normalize_for_judge(actual_text)
    if not a or not b:
        return False
    if a in b or b in a:
        return True
    lcs_cap = max(1, int(cfg.max_lcs_chars))
    common_len = _longest_common_substring_len(a[:lcs_cap], b[:lcs_cap])
    # 近包含：金标绝大部分已作为连续子串出现在检索文本中
    if (
        len(a) >= cfg.chunk_common_substring_min
        and common_len >= cfg.chunk_common_substring_min
        and (common_len / max(1, len(a))) >= 0.9
    ):
        return True
    cap = max(1, int(cfg.max_compare_chars))
    ratio = difflib.SequenceMatcher(None, a[:cap], b[:cap]).ratio()
    gate = _cjk_bigram_gate_ok(a, b, cfg) if cfg.cjk_bigram_as_gate else None

    if ratio >= cfg.chunk_match_ratio:
        if gate is False:
            pass
        else:
            return True
    # 保留 ``len>=20`` 公共子串兜底（CJK 与英文同享此分支）
    if len(a) >= 20 and len(b) >= 20:
        common = _common_substring_sampled(a, b, cfg)
        if common >= cfg.chunk_common_substring_min and ratio >= cfg.chunk_common_substring_ratio:
            if gate is False:
                pass
            else:
                return True
    # 非门禁模式：bigram 仍可作为额外命中通道（历史行为）
    if not cfg.cjk_bigram_as_gate and len(a) >= cfg.cjk_bigram_min_chars and len(b) >= cfg.cjk_bigram_min_chars:
        jaccard = _cjk_bigram_jaccard(a, b)
        if jaccard is not None and jaccard >= float(cfg.cjk_bigram_min_overlap):
            return True
    return False


def texts_match_expected_chunk(
    expect_text: Any,
    actual_texts: Iterable[Any],
    config: EvalJudgeConfig | None = None,
) -> bool:
    """对多段检索文本判定金标是否命中。

    - 逐段匹配；
    - 拼接全部 context 再匹配（覆盖跨 chunk 金标）；
    - 金标含 ``...`` / ``…`` 时，要求各子片段均能命中（允许落在不同段）。
    """
    cfg = config or EvalJudgeConfig()
    texts = [str(x or "") for x in (actual_texts or []) if str(x or "").strip()]
    if not texts:
        return False
    joined = "\n".join(texts)
    parts = _split_expected_chunk_parts(str(expect_text or ""))
    if len(parts) <= 1:
        target = parts[0] if parts else expect_text
        if text_match_loose(target, joined, cfg):
            return True
        return any(text_match_loose(target, t, cfg) for t in texts)
    # 多段金标：每个子片段至少命中某一段或拼接文本
    for part in parts:
        norm = _normalize_for_judge(part)
        if len(norm) < 12:
            continue
        if text_match_loose(part, joined, cfg):
            continue
        if any(text_match_loose(part, t, cfg) for t in texts):
            continue
        return False
    return True


def chunk_match_metrics(
    expect_text: Any,
    actual_text: Any,
    config: EvalJudgeConfig | None = None,
) -> dict[str, Any]:
    """返回 ``{similarity, coverage, common_chars, hit}``。

    ``hit`` 的判定与 :func:`text_match_loose` 完全等价；``coverage`` 是
    LCS 长度 / ``len(expect)``（粗略覆盖度，仅供诊断列展示，不参与命中）。
    """
    cfg = config or EvalJudgeConfig()
    a = _normalize_for_judge(expect_text)
    b = _normalize_for_judge(actual_text)
    if not a or not b:
        return {"similarity": 0.0, "coverage": 0.0, "common_chars": 0, "hit": False}
    cap = max(1, int(cfg.max_compare_chars))
    lcs_cap = max(1, int(cfg.max_lcs_chars))
    common_len = _longest_common_substring_len(a[:lcs_cap], b[:lcs_cap])
    ratio = difflib.SequenceMatcher(None, a[:cap], b[:cap]).ratio()
    coverage = common_len / max(1, len(a))
    hit = text_match_loose(a, b, cfg)
    return {
        "similarity": round(float(ratio), 4),
        "coverage": round(float(coverage), 4),
        "common_chars": int(common_len),
        "hit": bool(hit),
    }


def chunk_match_diagnostics(
    expected_chunk: Any,
    candidates: Iterable[dict[str, Any]] | None,
    sources: Iterable[dict[str, Any]] | None,
    config: EvalJudgeConfig | None = None,
) -> dict[str, Any]:
    """对所有 candidate / context 候选做 ``chunk_match_metrics``，返回排序结果。

    字段与历史 ``_chunk_match_diagnostics`` 保持一致：``has_expected_chunk``、
    ``expected_preview``、``best``、``top_matches``。
    """
    cfg = config or EvalJudgeConfig()
    expected = str(expected_chunk or "").strip()
    if not expected:
        return {"has_expected_chunk": False, "best": {}, "top_matches": []}
    rows: list[dict[str, Any]] = []
    for stage, items in (
        ("candidate", list(candidates or [])),
        ("context", list(sources or [])),
    ):
        for idx, item in enumerate(items, start=1):
            metrics = chunk_match_metrics(expected, str(item.get("chunk") or ""), cfg)
            rows.append(
                {
                    "stage": stage,
                    "rank": idx,
                    "file_name": str(item.get("file_name") or ""),
                    "score": item.get("score"),
                    "score_kind": str(item.get("score_kind") or ""),
                    "similarity": metrics["similarity"],
                    "coverage": metrics["coverage"],
                    "common_chars": metrics["common_chars"],
                    "hit": bool(metrics["hit"]),
                    "preview": _preview_chunk_text(str(item.get("chunk") or ""), 220),
                }
            )
    rows.sort(
        key=lambda x: (
            1 if x.get("stage") == "context" else 0,
            float(x.get("similarity") or 0),
            float(x.get("coverage") or 0),
            int(x.get("common_chars") or 0),
        ),
        reverse=True,
    )
    return {
        "has_expected_chunk": True,
        "expected_preview": _preview_chunk_text(expected, 220),
        "best": rows[0] if rows else {},
        "top_matches": rows[:5],
    }


def _file_name(item: dict[str, Any]) -> str:
    return str(item.get("file_name") or "").strip()


def _file_list_hits_expected(
    file_names: list[str],
    expected_files: set[str],
    expected_raw_values: list[Any] | None = None,
) -> bool | None:
    """与历史 ``_file_list_hits_expected`` 等价的纯函数实现。

    期望 raw 在 ``expected_files`` 内、或与 ``expected_raw_values`` 字面相同
    视为命中；非空但无命中为 False；空期望为 None。
    """
    if not file_names:
        return None
    raw_set = {str(x).strip() for x in (expected_raw_values or []) if str(x).strip()}
    for name in file_names:
        if not name:
            continue
        if name in expected_files or name in raw_set:
            return True
    return False


def _eval_filename_matches_expected(
    actual_name: str,
    expected_names: set[str],
    expected_raw_values: list[Any] | None = None,
) -> bool:
    raw_set = {str(x).strip() for x in (expected_raw_values or []) if str(x).strip()}
    if not actual_name:
        return False
    return actual_name in expected_names or actual_name in raw_set


def _is_system_answer_sentinel(answer: Any) -> bool:
    return str(answer or "").strip() in _SYSTEM_ANSWER_SENTINELS


def judge_case(
    case: dict[str, Any],
    vector_diag: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    answer: str,
    *,
    final_ranked_sources: list[dict[str, Any]] | None = None,
    use_rerank: bool = False,
    context_truncated: bool = False,
    config: EvalJudgeConfig | None = None,
    expected_file_info: dict[str, Any] | None = None,
    generation_mode: str | None = None,
    no_context: bool | None = None,
) -> dict[str, Any]:
    """主判定函数。

    ``expected_file_info`` 由调用方（``main.py`` 中的
    ``_resolve_expected_eval_file_details``）预先计算，包含 ``resolved_files``、
    ``has_unresolved``、``has_ambiguous`` 等字段——本函数不再做 IO。

    ``generation_mode=retrieval_only`` 时不评判 answer/abstain。
    系统哨兵答案（``[NO_CONTEXT]`` / ``[retrieval-only]``）不计为模型拒答，
    避免空检索被误标为 OVER_ABSTAIN。
    """
    cfg = config or EvalJudgeConfig()
    retrieval_only = str(generation_mode or "").strip().lower() in {"retrieval_only", "retrieval-only"}
    expected_raw_files = [str(x).strip() for x in (case.get("expected_file_names") or []) if str(x).strip()]
    expected_info = expected_file_info or {"resolved_files": [], "has_unresolved": False, "has_ambiguous": False}
    expected_files = set(expected_info.get("resolved_files") or [])

    candidate_file_list = [_file_name(x) for x in (vector_diag or []) if _file_name(x)]
    final_ranked_list = [_file_name(x) for x in (final_ranked_sources or []) if _file_name(x)]
    context_file_list = [_file_name(x) for x in (sources or []) if _file_name(x)]

    candidate_hit = _file_list_hits_expected(candidate_file_list, expected_files, expected_raw_files)
    final_ranked_hit = _file_list_hits_expected(final_ranked_list, expected_files, expected_raw_files)
    context_hit = _file_list_hits_expected(context_file_list, expected_files, expected_raw_files)

    expected_chunk = str(case.get("expected_chunk_content") or "").strip()
    chunk_hit: bool | None = None
    chunk_diag = chunk_match_diagnostics(expected_chunk, vector_diag, sources, cfg)
    if expected_chunk:
        chunk_hit = texts_match_expected_chunk(
            expected_chunk,
            [str(x.get("chunk") or "") for x in (sources or [])],
            cfg,
        )

    expected_answer = _normalize_for_judge(case.get("expected_answer"))
    answer_norm = _normalize_for_judge(answer)
    keywords = [
        _normalize_for_judge(x)
        for x in (case.get("expected_answer_keywords") or [])
        if _normalize_for_judge(x)
    ]
    answer_hit: bool | None = None
    if not retrieval_only:
        if keywords:
            answer_hit = all(k in answer_norm for k in keywords)
        elif expected_answer:
            answer_hit = expected_answer in answer_norm

    abstain_expected = bool(case.get("allow_abstain"))
    system_sentinel = _is_system_answer_sentinel(answer)
    empty_context = bool(no_context) if no_context is not None else (
        system_sentinel and str(answer or "").strip() == ANSWER_NO_CONTEXT
    )
    # retrieval-only：不评判 abstain
    # [NO_CONTEXT]：视为系统级拒答（对 allow_abstain=true 算正确；不算模型 OVER_ABSTAIN）
    # [retrieval-only]：不计 abstain
    if retrieval_only:
        abstain_actual = False
        abstain_correct = None
    elif empty_context or str(answer or "").strip() == ANSWER_NO_CONTEXT:
        abstain_actual = True
        abstain_correct = abstain_actual == abstain_expected
    elif system_sentinel:
        abstain_actual = False
        abstain_correct = abstain_actual == abstain_expected
    else:
        abstain_actual = is_abstain_answer(answer)
        abstain_correct = abstain_actual == abstain_expected

    matched_candidates = [
        name for name in candidate_file_list
        if _eval_filename_matches_expected(name, expected_files, expected_raw_files)
    ]
    non_file_eval = not bool(expected_raw_files)
    filtered_out = bool(candidate_file_list) and not bool(context_file_list) and candidate_hit is True and context_hit is False
    topk_or_rerank_drop = bool(candidate_hit is True and final_ranked_hit is False)
    context_budget_drop = bool(final_ranked_hit is True and context_hit is False and context_truncated)
    target_file_hit_but_chunk_miss = bool(context_hit is True and expected_chunk and chunk_hit is False)
    has_target_context = bool(expected_files) or bool(expected_chunk) or bool(case.get("expected_answer"))
    # 仅模型输出拒答文案才算 OVER_ABSTAIN；系统 [NO_CONTEXT] 走检索失败口径
    model_over_abstain = (
        (not retrieval_only)
        and (not empty_context)
        and (not system_sentinel)
        and (not abstain_expected)
        and bool(abstain_actual)
        and has_target_context
    )

    issue_codes: list[str] = []
    if bool(expected_info.get("has_unresolved")):
        issue_codes.append("EXPECTED_FILE_UNRESOLVED")
    if bool(expected_info.get("has_ambiguous")):
        issue_codes.append("EXPECTED_FILE_AMBIGUOUS")
    if filtered_out:
        issue_codes.append("FILTERED_OUT")
    if topk_or_rerank_drop:
        issue_codes.append("RERANK_DROP" if use_rerank else "TOPK_DROP")
    if context_budget_drop:
        issue_codes.append("CONTEXT_BUDGET_DROP")
    # 空候选时 candidate_hit 为 None；对有可解析期望文件的题目仍视为未命中
    target_file_miss = bool(expected_files) and candidate_hit is not True
    if target_file_miss:
        issue_codes.append("TARGET_FILE_MISS")
    if target_file_hit_but_chunk_miss:
        issue_codes.append("TARGET_FILE_HIT_BUT_CHUNK_MISS")
    if empty_context and expected_raw_files:
        issue_codes.append("NO_CONTEXT")
    if (not retrieval_only) and abstain_expected and abstain_correct is False:
        issue_codes.append(_ISSUE_ANSWERED_WHEN_SHOULD_ABSTAIN)
    if model_over_abstain:
        issue_codes.append(_ISSUE_ABSTAIN_WHEN_HAS_TARGET_CONTEXT)
    if non_file_eval and expected_chunk and chunk_hit is False:
        issue_codes.append("NON_FILE_CHUNK_MISS")
    if non_file_eval and answer_hit is False:
        issue_codes.append("NON_FILE_ANSWER_MISS")

    # 错误优先级：数据质量 / 检索失败 > 模型拒答策略（空检索哨兵不再抢占 TARGET_FILE_MISS）
    if (not retrieval_only) and abstain_expected:
        error_type = "OK" if abstain_correct else ERROR_ANSWERED_WHEN_SHOULD_ABSTAIN
    elif non_file_eval:
        if expected_chunk and chunk_hit is False:
            error_type = "NON_FILE_CHUNK_MISS"
        elif answer_hit is False:
            error_type = "NON_FILE_ANSWER_MISS"
        else:
            error_type = "OK"
    elif bool(expected_info.get("has_unresolved")):
        error_type = "EXPECTED_FILE_UNRESOLVED"
    elif bool(expected_info.get("has_ambiguous")):
        error_type = "EXPECTED_FILE_AMBIGUOUS"
    elif target_file_miss:
        error_type = "TARGET_FILE_MISS"
    elif filtered_out:
        error_type = "FILTERED_OUT"
    elif target_file_hit_but_chunk_miss:
        error_type = "TARGET_FILE_HIT_BUT_CHUNK_MISS"
    elif context_hit is not True and bool(expected_files):
        if context_budget_drop:
            error_type = "CONTEXT_BUDGET_DROP"
        elif topk_or_rerank_drop:
            error_type = "RERANK_DROP" if use_rerank else "TOPK_DROP"
        else:
            error_type = "CONTEXT_MISS"
    elif model_over_abstain:
        # 仅在检索已命中目标上下文时，才把模型拒答当作主错误
        error_type = ERROR_ABSTAIN_WHEN_HAS_TARGET_CONTEXT
    else:
        error_type = "OK"

    return {
        "candidate_hit": candidate_hit,
        "context_hit": context_hit,
        "chunk_hit": chunk_hit,
        "answer_hit": answer_hit,
        "abstain_expected": abstain_expected,
        "abstain_actual": abstain_actual,
        "abstain_correct": abstain_correct,
        "abstain_with_target_context": bool(model_over_abstain),
        "error_type": error_type,
        "issue_codes": issue_codes,
        "expected_file_resolution": expected_info,
        "matched_candidate_files": matched_candidates,
        "filtered_out": filtered_out,
        "final_ranked_hit": final_ranked_hit,
        "topk_or_rerank_drop": topk_or_rerank_drop,
        "context_budget_drop": context_budget_drop,
        "target_file_hit_but_chunk_miss": target_file_hit_but_chunk_miss,
        "chunk_diagnostics": chunk_diag,
        "retrieval_only": retrieval_only,
        "no_context": bool(empty_context),
    }


# 简化的 ``Path`` 重导出，便于调用方写 ``eval_judge.Path(...)`` 时不依赖 main
__all__ = [
    "EvalJudgeConfig",
    "ANSWER_NO_CONTEXT",
    "ANSWER_RETRIEVAL_ONLY",
    "ERROR_ABSTAIN_WHEN_HAS_TARGET_CONTEXT",
    "ERROR_ANSWERED_WHEN_SHOULD_ABSTAIN",
    "is_abstain_answer",
    "text_match_loose",
    "chunk_match_metrics",
    "chunk_match_diagnostics",
    "judge_case",
]
