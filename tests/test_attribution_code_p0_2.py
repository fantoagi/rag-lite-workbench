"""Test that attribution_code routes the new ABSTAIN_WHEN_HAS_TARGET_CONTEXT
error to a dedicated OVER_ABSTAIN_WITH_TARGET bucket instead of falling
through to the ANSWER_OR_JUDGE_MISS catch-all.

Background: see P0-2 in the second-round review. The new judge error type
covers cases where allow_abstain=false but the model still emits an abstain
phrase AND the case has a target context (file/chunk/expected answer).
"""
from __future__ import annotations

from rag_lite import eval_platform


def test_abstain_when_has_target_context_is_routed_specifically():
    # Simulate a fully-served context: final_sources is non-empty and
    # context_truncated is False. Without the new branch, the catch-all at the
    # bottom of attribution_code would return ANSWER_OR_JUDGE_MISS, which is
    # semantically wrong (the failure is the model refusing to answer, not a
    # judgement miss).
    diagnostics = {
        "final_sources": [
            {"file_name": "招商证券2025年报.pdf", "chunk": "示例片段内容"}
        ],
        "context_truncated": False,
    }
    out = eval_platform.attribution_code(
        diagnostics=diagnostics,
        error_type="ABSTAIN_WHEN_HAS_TARGET_CONTEXT",
    )
    assert out == "OVER_ABSTAIN_WITH_TARGET", out


def test_abstain_when_has_target_context_via_judge_dict():
    # attribution_code may also be driven by the judge dict instead of an
    # explicit error_type. Cover that path too.
    judge = {"error_type": "ABSTAIN_WHEN_HAS_TARGET_CONTEXT"}
    out = eval_platform.attribution_code(
        diagnostics={"final_sources": [{"file_name": "x.pdf", "chunk": "y"}], "context_truncated": False},
        judge=judge,
    )
    assert out == "OVER_ABSTAIN_WITH_TARGET", out


def test_answered_when_should_abstain_still_uses_policy_bucket():
    # The pre-existing branch must keep working; we only added a new branch
    # and must not regress the symmetric ANSWERED_WHEN_SHOULD_ABSTAIN case.
    out = eval_platform.attribution_code(
        diagnostics={},
        error_type="ANSWERED_WHEN_SHOULD_ABSTAIN",
    )
    assert out == "ABSTAIN_POLICY_MISS", out
