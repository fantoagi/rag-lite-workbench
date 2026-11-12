"""知识库表格行号不能直接当成上传文件下标。"""
from rag_lite.kb_files import disk_index_for_kb_table_row, kb_status_row_count, kb_status_rows


def test_status_row_count_matches_builder():
    assert kb_status_row_count() == len(kb_status_rows({}))
    assert kb_status_row_count() == 6
    assert str(kb_status_rows({"manifest_build_id": "abc"})[0][0]).startswith("[索引状态]")


def test_clicking_status_row_does_not_select_first_file():
    # 旧逻辑：行号 0 < len(disk) 就返回 0，点「索引状态」会删掉第一个文件。
    names = ["甲.pdf", "乙.pdf", "丙.pdf"]
    cells = ["[索引状态] build_id", "abc", "—", "—", "—"]
    assert disk_index_for_kb_table_row(0, cells, names) is None
    assert disk_index_for_kb_table_row(0, ["（暂无文件）", "—", "—", "—", "—"], names) is None


def test_file_row_uses_filename_not_table_index():
    names = ["甲.pdf", "乙.pdf", "丙.pdf"]
    # 表格第 7 行（下标 6）才是第一个文件。旧逻辑会把 6 当成 disk[6]。
    assert disk_index_for_kb_table_row(6, ["甲.pdf", "1 KB", "—", "—", "已入库"], names) == 0
    assert disk_index_for_kb_table_row(8, ["丙.pdf", "1 KB", "—", "—", "已入库"], names) == 2


def test_index_fallback_subtracts_status_rows_when_cells_missing():
    names = ["甲.pdf", "乙.pdf"]
    assert disk_index_for_kb_table_row(6, None, names) == 0
    assert disk_index_for_kb_table_row(0, None, names) is None
    assert disk_index_for_kb_table_row(7, None, names) == 1
