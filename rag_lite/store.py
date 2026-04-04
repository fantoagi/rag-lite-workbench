from __future__ import annotations

import csv
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class QALogRow:
    id: int | None
    created_at: str
    question: str
    answer: str
    sources: list[dict[str, Any]]
    params: dict[str, Any]
    rating: int | None
    note: str | None
    session_id: int | None = None


class ExperimentStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS qa_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    sources_json TEXT NOT NULL,
                    params_json TEXT NOT NULL,
                    rating INTEGER,
                    note TEXT,
                    session_id INTEGER
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS upload_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    saved_path TEXT NOT NULL,
                    original_name TEXT,
                    size_bytes INTEGER,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS index_manifest (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    built_at TEXT NOT NULL,
                    embed_model TEXT NOT NULL,
                    chunk_mode TEXT NOT NULL,
                    chunk_size INTEGER NOT NULL,
                    chunk_overlap INTEGER NOT NULL,
                    files_json TEXT NOT NULL
                )
                """
            )
            try:
                conn.execute("ALTER TABLE qa_log ADD COLUMN session_id INTEGER")
            except sqlite3.OperationalError:
                pass
            conn.commit()

    def log_upload(self, saved_path: str, original_name: str, size_bytes: int) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO upload_log (saved_path, original_name, size_bytes, created_at) VALUES (?,?,?,?)",
                (saved_path, original_name, size_bytes, _utc_now()),
            )
            conn.commit()

    def insert_qa(
        self,
        question: str,
        answer: str,
        sources: list[dict[str, Any]],
        params: dict[str, Any],
        rating: int | None = None,
        note: str | None = None,
        session_id: int | None = None,
    ) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO qa_log (created_at, question, answer, sources_json, params_json, rating, note, session_id)
                VALUES (?,?,?,?,?,?,?,?)
                """,
                (
                    _utc_now(),
                    question,
                    answer,
                    json.dumps(sources, ensure_ascii=False),
                    json.dumps(params, ensure_ascii=False),
                    rating,
                    note,
                    session_id,
                ),
            )
            conn.commit()
            return int(cur.lastrowid)

    def update_qa_rating(self, row_id: int, rating: int | None, note: str | None) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE qa_log SET rating = ?, note = ? WHERE id = ?",
                (rating, note, row_id),
            )
            conn.commit()

    def fetch_all_qa(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT id, created_at, question, answer, sources_json, params_json, rating, note, session_id FROM qa_log ORDER BY id ASC"
            ).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            out.append(
                {
                    "id": r["id"],
                    "created_at": r["created_at"],
                    "question": r["question"],
                    "answer": r["answer"],
                    "sources": json.loads(r["sources_json"]),
                    "params": json.loads(r["params_json"]),
                    "rating": r["rating"],
                    "note": r["note"],
                    "session_id": r["session_id"],
                }
            )
        return out

    def export_json(self, dest: Path) -> None:
        data = self.fetch_all_qa()
        dest.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def export_csv(self, dest: Path) -> None:
        rows = self.fetch_all_qa()
        if not rows:
            dest.write_text("", encoding="utf-8")
            return
        fields = [
            "id",
            "created_at",
            "question",
            "answer",
            "sources_json",
            "params_json",
            "rating",
            "note",
            "session_id",
        ]
        with dest.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow(
                    {
                        "id": r["id"],
                        "created_at": r["created_at"],
                        "question": r["question"],
                        "answer": r["answer"],
                        "sources_json": json.dumps(r["sources"], ensure_ascii=False),
                        "params_json": json.dumps(r["params"], ensure_ascii=False),
                        "rating": r["rating"],
                        "note": r["note"],
                        "session_id": r["session_id"],
                    }
                )

    # --- Sessions ---

    def create_session(self, title: str | None = None) -> int:
        now = _utc_now()
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO chat_sessions (title, created_at, updated_at) VALUES (?,?,?)",
                (title, now, now),
            )
            conn.commit()
            return int(cur.lastrowid)

    def update_session_meta(self, session_id: int, title: str | None = None) -> None:
        now = _utc_now()
        with self._connect() as conn:
            if title is not None:
                conn.execute(
                    "UPDATE chat_sessions SET title = ?, updated_at = ? WHERE id = ?",
                    (title, now, session_id),
                )
            else:
                conn.execute(
                    "UPDATE chat_sessions SET updated_at = ? WHERE id = ?",
                    (now, session_id),
                )
            conn.commit()

    def list_sessions(self, limit: int = 200) -> list[dict[str, Any]]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT id, title, created_at, updated_at FROM chat_sessions
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def fetch_qa_rows_for_session(self, session_id: int) -> list[dict[str, Any]]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT id, created_at, question, answer, sources_json, params_json, rating, note
                FROM qa_log WHERE session_id = ? ORDER BY id ASC
                """,
                (session_id,),
            ).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            out.append(
                {
                    "id": r["id"],
                    "created_at": r["created_at"],
                    "question": r["question"],
                    "answer": r["answer"],
                    "sources": json.loads(r["sources_json"]),
                    "params": json.loads(r["params_json"]),
                    "rating": r["rating"],
                    "note": r["note"],
                }
            )
        return out

    def ensure_default_session(self) -> int:
        rows = self.list_sessions(limit=1)
        if rows:
            return int(rows[0]["id"])
        return self.create_session("默认会话")

    def delete_session(self, session_id: int) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM qa_log WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))
            conn.commit()

    # --- Index manifest (last successful vector build) ---

    def save_index_manifest(
        self,
        embed_model: str,
        chunk_mode: str,
        chunk_size: int,
        chunk_overlap: int,
        files: list[dict[str, Any]],
    ) -> None:
        payload = json.dumps(files, ensure_ascii=False)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO index_manifest
                (id, built_at, embed_model, chunk_mode, chunk_size, chunk_overlap, files_json)
                VALUES (1, ?, ?, ?, ?, ?, ?)
                """,
                (_utc_now(), embed_model, chunk_mode, int(chunk_size), int(chunk_overlap), payload),
            )
            conn.commit()

    def get_index_manifest(self) -> dict[str, Any] | None:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM index_manifest WHERE id = 1").fetchone()
        if row is None:
            return None
        d = dict(row)
        try:
            d["files"] = json.loads(d.pop("files_json"))
        except Exception:
            d["files"] = []
        return d

    def clear_index_manifest(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM index_manifest WHERE id = 1")
            conn.commit()
