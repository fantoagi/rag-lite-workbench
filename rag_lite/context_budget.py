"""从已排序片段里选出真正送给 LLM 的那一段。

两道裁剪要分开记：
- answer_top_k：生成侧条数上限（配置 retrieval.answer_top_k）
- 字符预算：max_llm_context_chars，按完整片段边界截断

评测里前者是 ANSWER_TOPK_DROP，后者才是 CONTEXT_BUDGET_DROP。
"""
from __future__ import annotations

from typing import Any, Callable


_PREFIX = "【已知上下文】\n"
_MID = "\n\n【用户问题】\n"
_SEP = "\n\n---\n\n"


def select_for_llm_context(
    items: list[Any],
    render_part: Callable[[Any, int], str],
    *,
    query: str,
    max_context_chars: int,
    answer_top_k: int,
) -> tuple[list[Any], bool, bool]:
    """返回 (selected, char_truncated, capped_by_answer_top_k)。"""
    pool = list(items or [])
    if not pool:
        return [], False, False
    try:
        k = int(answer_top_k)
    except (TypeError, ValueError):
        k = 0
    capped = False
    if k > 0 and len(pool) > k:
        pool = pool[:k]
        capped = True
    q = str(query or "").strip()
    budget = int(max_context_chars) - (len(_PREFIX) + len(_MID) + len(q))
    if budget <= 0:
        return [], True, capped
    selected: list[Any] = []
    used = 0
    sep_len = len(_SEP)
    for item in pool:
        part = render_part(item, len(selected) + 1)
        add = len(part) + (sep_len if selected else 0)
        if used + add > budget:
            break
        selected.append(item)
        used += add
    char_truncated = len(selected) < len(pool)
    return selected, char_truncated, capped
