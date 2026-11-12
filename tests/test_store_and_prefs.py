"""SQLite 上传时间精确匹配、WAL，以及偏好文件的原子写入。"""
import sqlite3

from rag_lite.prefs import load_prefs, save_prefs
from rag_lite.store import ExperimentStore


def test_upload_time_does_not_confuse_filename_suffix(tmp_path):
    store = ExperimentStore(tmp_path / "rag.db")
    store.log_upload(str(tmp_path / "报告.pdf"), "报告.pdf", 10)
    store.log_upload(str(tmp_path / "年度报告.pdf"), "年度报告.pdf", 20)
    with store._connect() as conn:
        conn.execute(
            "UPDATE upload_log SET created_at = ? WHERE original_name = ?",
            ("2020-01-01T00:00:00+00:00", "报告.pdf"),
        )
        conn.execute(
            "UPDATE upload_log SET created_at = ? WHERE original_name = ?",
            ("2024-06-01T00:00:00+00:00", "年度报告.pdf"),
        )
        conn.commit()
    # 旧 SQL 是 LIKE %报告.pdf，较新的「年度报告.pdf」会盖住「报告.pdf」。
    assert store.get_latest_upload_time_for_filename("报告.pdf") == "2020-01-01T00:00:00+00:00"
    assert store.get_latest_upload_time_for_filename("年度报告.pdf") == "2024-06-01T00:00:00+00:00"


def test_sqlite_uses_wal_and_longer_busy_timeout(tmp_path):
    store = ExperimentStore(tmp_path / "rag.db")
    conn = store._connect()
    try:
        mode = str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower()
        busy = int(conn.execute("PRAGMA busy_timeout").fetchone()[0])
    finally:
        conn.close()
    assert mode == "wal"
    assert busy >= 30000
    # 持久化的 journal mode 也能被另一个连接读到
    with sqlite3.connect(tmp_path / "rag.db") as other:
        mode2 = str(other.execute("PRAGMA journal_mode").fetchone()[0]).lower()
    assert mode2 == "wal"


def test_save_prefs_replaces_atomically(tmp_path):
    save_prefs(tmp_path, {"embed_model": "bge-m3"})
    assert load_prefs(tmp_path)["embed_model"] == "bge-m3"
    assert not (tmp_path / "ui_preferences.json.tmp").exists()
    save_prefs(tmp_path, {"embed_model": "bge-m3", "llm_model": "qwen2.5:0.5b"})
    assert load_prefs(tmp_path)["llm_model"] == "qwen2.5:0.5b"
