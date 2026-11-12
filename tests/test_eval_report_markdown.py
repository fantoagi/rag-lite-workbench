"""评测 Markdown 要能直接读出中文失败原因，而不是只剩错误码。"""
from rag_lite.platform_ops import _eval_run_report_markdown


def test_markdown_counts_chinese_failure_reason():
    payload = {
        "run": {
            "id": 3,
            "dataset_name": "finance",
            "name": "retrieval",
            "created_at": "2026-11-12T00:00:00+00:00",
            "params": {"run_fingerprint": "abc", "retrieval_mode": "hybrid"},
            "summary": {"case_count": 1, "ok_count": 0, "attribution_counts": {"HYBRID_TARGET_MISS": 1}},
        },
        "cases": [
            {
                "id": 1,
                "question": "总部在哪里？",
                "expected_file_names": ["招商证券年报.pdf"],
                "expected_chunk_content": "",
                "expected_answer": "深圳",
                "allow_abstain": False,
                "tags": ["事实抽取"],
                "note": "",
            }
        ],
        "results": [
            {
                "case_id": 1,
                "question": "总部在哪里？",
                "error_type": "TARGET_FILE_MISS",
                "answer": "[NO_CONTEXT]",
                "candidate_hit": False,
                "context_hit": False,
                "chunk_hit": None,
                "answer_hit": False,
                "abstain_actual": False,
                "abstain_correct": None,
                "sources": [],
                "qa_id": 9,
                "diagnostics": {
                    "retrieval_attribution": "HYBRID_TARGET_MISS",
                    "eval_case_diagnostics": {},
                },
            }
        ],
    }
    md = _eval_run_report_markdown(payload)
    assert "## 失败原因" in md
    assert "未召回到应命中文档" in md
    assert "TARGET_FILE_MISS" in md
