"""知识库文件表：状态行与真实文件行的对应关系。

表格顶部有若干「索引状态」行，Gradio 的行号不是上传目录快照下标。
用行号直接删文件会删错；这里只按单元格文件名（或扣掉状态行后的下标）解析。
"""
from __future__ import annotations

from typing import Any

KB_STATUS_PREFIX = "[索引状态]"
KB_EMPTY_PREFIX = "（暂无"


def kb_status_rows(index_state: dict[str, Any] | None) -> list[list[str]]:
    state = index_state or {}
    files = int(state.get("diagnostics_total_files") or 0)
    chunks = int(state.get("diagnostics_total_chunks") or 0)
    return [
        [f"{KB_STATUS_PREFIX} build_id", str(state.get("manifest_build_id") or "—"), "—", "—", "—"],
        [f"{KB_STATUS_PREFIX} active一致", "是" if state.get("active_dir_matches_manifest") else "否", "—", "—", "—"],
        [f"{KB_STATUS_PREFIX} readiness", "通过" if state.get("readiness_ok") else "未通过", "—", "—", "—"],
        [f"{KB_STATUS_PREFIX} uploads变化", "是" if state.get("uploads_changed_since_manifest") else "否", "—", "—", "—"],
        [f"{KB_STATUS_PREFIX} diagnostics", f"{files} 文件 / {chunks} 切片", "—", "—", "—"],
        [f"{KB_STATUS_PREFIX} 缺失上传文件", str(int(state.get("missing_uploaded_files_count") or 0)), "—", "—", "—"],
    ]


def kb_status_row_count() -> int:
    return len(kb_status_rows({}))


def disk_index_for_kb_table_row(
    row_index: int | None,
    row_cells: list | tuple | None,
    disk_names: list[str],
    *,
    status_row_count: int | None = None,
) -> int | None:
    """把表格点击解析成上传快照下标。状态行和空占位返回 None。"""
    names = [str(n) for n in disk_names]
    prefix = kb_status_row_count() if status_row_count is None else int(status_row_count)
    cells = list(row_cells) if isinstance(row_cells, (list, tuple)) else None
    if cells:
        head = str(cells[0]).strip() if cells else ""
        if not head or head.startswith(KB_STATUS_PREFIX) or head.startswith(KB_EMPTY_PREFIX):
            return None
        for i, name in enumerate(names):
            if name == head:
                return i
        base = head.rstrip("…").rstrip(".").strip()
        if base:
            hits = [
                i
                for i, name in enumerate(names)
                if name.startswith(base) or (len(base) >= 6 and base in name)
            ]
            if len(hits) == 1:
                return hits[0]
        return None
    if row_index is None:
        return None
    try:
        disk_i = int(row_index) - prefix
    except (TypeError, ValueError):
        return None
    if 0 <= disk_i < len(names):
        return disk_i
    return None
