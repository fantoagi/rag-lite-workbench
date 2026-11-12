"""生成侧条数上限和字符预算是两件不同的事。"""
from rag_lite.context_budget import select_for_llm_context


def _render(item, idx: int) -> str:
    return f"[片段 {idx}] {item}"


def test_answer_top_k_cap_is_not_char_truncation():
    selected, char_truncated, capped = select_for_llm_context(
        ["甲" * 20, "乙" * 20, "丙" * 20],
        _render,
        query="总部在哪",
        max_context_chars=100_000,
        answer_top_k=1,
    )
    assert selected == ["甲" * 20]
    assert capped is True
    assert char_truncated is False


def test_char_budget_drops_later_parts_without_answer_cap():
    query = "q"
    first = "A"
    part = _render(first, 1)
    fixed = len("【已知上下文】\n") + len("\n\n【用户问题】\n") + len(query)
    selected, char_truncated, capped = select_for_llm_context(
        [first, "B" * 80],
        _render,
        query=query,
        max_context_chars=fixed + len(part),
        answer_top_k=0,
    )
    assert selected == [first]
    assert char_truncated is True
    assert capped is False


def test_empty_input():
    selected, char_truncated, capped = select_for_llm_context(
        [], _render, query="q", max_context_chars=1000, answer_top_k=4
    )
    assert selected == []
    assert char_truncated is False
    assert capped is False
