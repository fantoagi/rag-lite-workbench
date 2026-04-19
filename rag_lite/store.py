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
                    diagnostics_json TEXT,
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
                    files_json TEXT NOT NULL,
                    build_id TEXT,
                    active_chroma_subdir TEXT,
                    activated_at TEXT,
                    readiness_json TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS eval_datasets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    description TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    case_count INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS eval_cases (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_id INTEGER NOT NULL,
                    question TEXT NOT NULL,
                    expected_answer TEXT,
                    expected_chunk_content TEXT,
                    expected_file_names_json TEXT,
                    expected_answer_keywords_json TEXT,
                    allow_abstain INTEGER NOT NULL DEFAULT 0,
                    tags_json TEXT,
                    note TEXT,
                    sort_index INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS eval_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    params_json TEXT NOT NULL,
                    summary_json TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS eval_case_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    case_id INTEGER NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    sources_json TEXT NOT NULL,
                    diagnostics_json TEXT,
                    candidate_hit INTEGER,
                    context_hit INTEGER,
                    chunk_hit INTEGER,
                    answer_hit INTEGER,
                    abstain_expected INTEGER NOT NULL DEFAULT 0,
                    abstain_actual INTEGER NOT NULL DEFAULT 0,
                    abstain_correct INTEGER,
                    error_type TEXT,
                    qa_id INTEGER
                )
                """
            )
            try:
                conn.execute("ALTER TABLE qa_log ADD COLUMN session_id INTEGER")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE qa_log ADD COLUMN diagnostics_json TEXT")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE eval_cases ADD COLUMN expected_chunk_content TEXT")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE eval_case_results ADD COLUMN chunk_hit INTEGER")
            except sqlite3.OperationalError:
                pass
            for sql in (
                "ALTER TABLE index_manifest ADD COLUMN build_id TEXT",
                "ALTER TABLE index_manifest ADD COLUMN active_chroma_subdir TEXT",
                "ALTER TABLE index_manifest ADD COLUMN activated_at TEXT",
                "ALTER TABLE index_manifest ADD COLUMN readiness_json TEXT",
            ):
                try:
                    conn.execute(sql)
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

    def get_latest_upload_time_for_filename(self, filename: str) -> str | None:
        """upload_log 中与该 basename 匹配的最新一条上传时间（ISO）。"""
        fn = Path(filename).name
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """
                SELECT created_at FROM upload_log
                WHERE original_name = ? OR saved_path LIKE ?
                ORDER BY created_at DESC LIMIT 1
                """,
                (fn, f"%{fn}"),
            ).fetchone()
        return str(row["created_at"]) if row else None

    def insert_qa(
        self,
        question: str,
        answer: str,
        sources: list[dict[str, Any]],
        params: dict[str, Any],
        diagnostics: dict[str, Any] | None = None,
        rating: int | None = None,
        note: str | None = None,
        session_id: int | None = None,
    ) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO qa_log (
                    created_at, question, answer, sources_json, params_json, diagnostics_json, rating, note, session_id
                )
                VALUES (?,?,?,?,?,?,?,?,?)
                """,
                (
                    _utc_now(),
                    question,
                    answer,
                    json.dumps(sources, ensure_ascii=False),
                    json.dumps(params, ensure_ascii=False),
                    json.dumps(diagnostics or {}, ensure_ascii=False),
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
                "SELECT id, created_at, question, answer, sources_json, params_json, diagnostics_json, rating, note, session_id FROM qa_log ORDER BY id ASC"
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
                    "diagnostics": json.loads(r["diagnostics_json"] or "{}"),
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
            "diagnostics_json",
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
                        "diagnostics_json": json.dumps(r.get("diagnostics") or {}, ensure_ascii=False),
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
                SELECT id, created_at, question, answer, sources_json, params_json, diagnostics_json, rating, note
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
                    "diagnostics": json.loads(r["diagnostics_json"] or "{}"),
                    "rating": r["rating"],
                    "note": r["note"],
                }
            )
        return out

    def get_latest_qa_id_for_session(self, session_id: int) -> int | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id FROM qa_log WHERE session_id = ? ORDER BY id DESC LIMIT 1",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return int(row[0])

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

    # --- Eval datasets / runs ---

    def replace_eval_dataset(
        self,
        name: str,
        cases: list[dict[str, Any]],
        description: str | None = None,
    ) -> int:
        now = _utc_now()
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            old = conn.execute(
                "SELECT id FROM eval_datasets WHERE name = ?",
                (name,),
            ).fetchone()
            if old is not None:
                dataset_id = int(old["id"])
                run_rows = conn.execute(
                    "SELECT id FROM eval_runs WHERE dataset_id = ?",
                    (dataset_id,),
                ).fetchall()
                run_ids = [int(r[0]) for r in run_rows]
                if run_ids:
                    qmarks = ",".join("?" for _ in run_ids)
                    conn.execute(f"DELETE FROM eval_case_results WHERE run_id IN ({qmarks})", run_ids)
                conn.execute("DELETE FROM eval_runs WHERE dataset_id = ?", (dataset_id,))
                conn.execute("DELETE FROM eval_cases WHERE dataset_id = ?", (dataset_id,))
                conn.execute(
                    """
                    UPDATE eval_datasets
                    SET description = ?, updated_at = ?, case_count = ?
                    WHERE id = ?
                    """,
                    (description, now, len(cases), dataset_id),
                )
            else:
                cur = conn.execute(
                    """
                    INSERT INTO eval_datasets (name, description, created_at, updated_at, case_count)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (name, description, now, now, len(cases)),
                )
                dataset_id = int(cur.lastrowid)

            for idx, case in enumerate(cases, start=1):
                conn.execute(
                    """
                    INSERT INTO eval_cases (
                        dataset_id, question, expected_answer, expected_chunk_content, expected_file_names_json,
                        expected_answer_keywords_json, allow_abstain, tags_json, note, sort_index
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        dataset_id,
                        str(case.get("question") or "").strip(),
                        str(case.get("expected_answer") or "").strip() or None,
                        str(case.get("expected_chunk_content") or "").strip() or None,
                        json.dumps(case.get("expected_file_names") or [], ensure_ascii=False),
                        json.dumps(case.get("expected_answer_keywords") or [], ensure_ascii=False),
                        1 if bool(case.get("allow_abstain")) else 0,
                        json.dumps(case.get("tags") or [], ensure_ascii=False),
                        str(case.get("note") or "").strip() or None,
                        idx,
                    ),
                )
            conn.commit()
        return dataset_id

    def list_eval_datasets(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT id, name, description, created_at, updated_at, case_count
                FROM eval_datasets
                ORDER BY updated_at DESC, id DESC
                """
            ).fetchall()
        return [dict(r) for r in rows]

    def get_eval_dataset(self, dataset_id: int) -> dict[str, Any] | None:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """
                SELECT id, name, description, created_at, updated_at, case_count
                FROM eval_datasets
                WHERE id = ?
                """,
                (dataset_id,),
            ).fetchone()
        return dict(row) if row is not None else None

    def fetch_eval_cases(self, dataset_id: int) -> list[dict[str, Any]]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT id, question, expected_answer, expected_chunk_content, expected_file_names_json,
                       expected_answer_keywords_json, allow_abstain, tags_json, note, sort_index
                FROM eval_cases
                WHERE dataset_id = ?
                ORDER BY sort_index ASC, id ASC
                """,
                (dataset_id,),
            ).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            out.append(
                {
                    "id": int(r["id"]),
                    "question": r["question"],
                    "expected_answer": r["expected_answer"] or "",
                    "expected_chunk_content": r["expected_chunk_content"] or "",
                    "expected_file_names": json.loads(r["expected_file_names_json"] or "[]"),
                    "expected_answer_keywords": json.loads(r["expected_answer_keywords_json"] or "[]"),
                    "allow_abstain": bool(r["allow_abstain"]),
                    "tags": json.loads(r["tags_json"] or "[]"),
                    "note": r["note"] or "",
                    "sort_index": int(r["sort_index"] or 0),
                }
            )
        return out

    def create_eval_run(
        self,
        dataset_id: int,
        name: str,
        params: dict[str, Any],
    ) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO eval_runs (dataset_id, name, created_at, params_json, summary_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (dataset_id, name, _utc_now(), json.dumps(params, ensure_ascii=False), json.dumps({}, ensure_ascii=False)),
            )
            conn.commit()
            return int(cur.lastrowid)

    def save_eval_case_result(
        self,
        run_id: int,
        case_id: int,
        question: str,
        answer: str,
        sources: list[dict[str, Any]],
        diagnostics: dict[str, Any] | None,
        candidate_hit: bool | None,
        context_hit: bool | None,
        chunk_hit: bool | None,
        answer_hit: bool | None,
        abstain_expected: bool,
        abstain_actual: bool,
        abstain_correct: bool | None,
        error_type: str,
        qa_id: int | None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO eval_case_results (
                    run_id, case_id, question, answer, sources_json, diagnostics_json,
                    candidate_hit, context_hit, chunk_hit, answer_hit, abstain_expected,
                    abstain_actual, abstain_correct, error_type, qa_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    case_id,
                    question,
                    answer,
                    json.dumps(sources, ensure_ascii=False),
                    json.dumps(diagnostics or {}, ensure_ascii=False),
                    None if candidate_hit is None else int(candidate_hit),
                    None if context_hit is None else int(context_hit),
                    None if chunk_hit is None else int(chunk_hit),
                    None if answer_hit is None else int(answer_hit),
                    int(bool(abstain_expected)),
                    int(bool(abstain_actual)),
                    None if abstain_correct is None else int(abstain_correct),
                    error_type,
                    qa_id,
                ),
            )
            conn.commit()

    def finish_eval_run(self, run_id: int, summary: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE eval_runs SET summary_json = ? WHERE id = ?",
                (json.dumps(summary, ensure_ascii=False), run_id),
            )
            conn.commit()

    def list_eval_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT r.id, r.dataset_id, d.name AS dataset_name, r.name, r.created_at,
                       r.params_json, r.summary_json
                FROM eval_runs r
                JOIN eval_datasets d ON d.id = r.dataset_id
                ORDER BY r.id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            out.append(
                {
                    "id": int(r["id"]),
                    "dataset_id": int(r["dataset_id"]),
                    "dataset_name": r["dataset_name"],
                    "name": r["name"],
                    "created_at": r["created_at"],
                    "params": json.loads(r["params_json"] or "{}"),
                    "summary": json.loads(r["summary_json"] or "{}"),
                }
            )
        return out

    def fetch_eval_case_results(self, run_id: int) -> list[dict[str, Any]]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT id, run_id, case_id, question, answer, sources_json, diagnostics_json,
                       candidate_hit, context_hit, chunk_hit, answer_hit, abstain_expected, abstain_actual,
                       abstain_correct, error_type, qa_id
                FROM eval_case_results
                WHERE run_id = ?
                ORDER BY id ASC
                """,
                (run_id,),
            ).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            out.append(
                {
                    "id": int(r["id"]),
                    "run_id": int(r["run_id"]),
                    "case_id": int(r["case_id"]),
                    "question": r["question"],
                    "answer": r["answer"],
                    "sources": json.loads(r["sources_json"] or "[]"),
                    "diagnostics": json.loads(r["diagnostics_json"] or "{}"),
                    "candidate_hit": None if r["candidate_hit"] is None else bool(r["candidate_hit"]),
                    "context_hit": None if r["context_hit"] is None else bool(r["context_hit"]),
                    "chunk_hit": None if r["chunk_hit"] is None else bool(r["chunk_hit"]),
                    "answer_hit": None if r["answer_hit"] is None else bool(r["answer_hit"]),
                    "abstain_expected": bool(r["abstain_expected"]),
                    "abstain_actual": bool(r["abstain_actual"]),
                    "abstain_correct": None if r["abstain_correct"] is None else bool(r["abstain_correct"]),
                    "error_type": r["error_type"] or "",
                    "qa_id": r["qa_id"],
                }
            )
        return out

    # --- Index manifest (last successful vector build) ---

    def save_index_manifest(
        self,
        embed_model: str,
        chunk_mode: str,
        chunk_size: int,
        chunk_overlap: int,
        files: list[dict[str, Any]],
        *,
        build_id: str | None = None,
        active_chroma_subdir: str | None = None,
        activated_at: str | None = None,
        readiness: dict[str, Any] | None = None,
    ) -> None:
        payload = json.dumps(files, ensure_ascii=False)
        readiness_json = json.dumps(readiness or {}, ensure_ascii=False)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO index_manifest
                (id, built_at, embed_model, chunk_mode, chunk_size, chunk_overlap, files_json,
                 build_id, active_chroma_subdir, activated_at, readiness_json)
                VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _utc_now(),
                    embed_model,
                    chunk_mode,
                    int(chunk_size),
                    int(chunk_overlap),
                    payload,
                    build_id,
                    active_chroma_subdir,
                    activated_at,
                    readiness_json,
                ),
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
        try:
            d["readiness"] = json.loads(d.pop("readiness_json") or "{}")
        except Exception:
            d["readiness"] = {}
        return d

    def clear_index_manifest(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM index_manifest WHERE id = 1")
            conn.commit()
