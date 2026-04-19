from __future__ import annotations

import concurrent.futures
import gc
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
import uuid
import unicodedata
from collections.abc import Iterator
from itertools import zip_longest
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.indices.vector_store.retrievers import VectorIndexRetriever
from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter
from llama_index.core.schema import MetadataMode
from llama_index.core.utils import iter_batch
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from rag_lite.config import AppConfig
from rag_lite.readers import default_local_file_extractors

# 与 VectorStoreIndex.insert_batch_size 对齐；避免默认 2048 导致「一整根 2048 步」tqdm，看起来像反复在向量化
_EMBED_BATCH_SIZE = 48


@contextmanager
def _suppress_embedding_tqdm():
    """构建索引时关闭 LlamaIndex/tqdm 写终端；进度以 Gradio 状态框为准。"""
    key = "TQDM_DISABLE"
    prev = os.environ.get(key)
    os.environ[key] = "1"
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = prev


def _allowed_suffix(name: str) -> bool:
    return Path(name).suffix.lower() in {".pdf", ".txt", ".md", ".markdown", ".docx"}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _chroma_versions_root(cfg: AppConfig) -> Path:
    root = cfg.chroma_dir.parent / f"{cfg.chroma_dir.name}.__versions__"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _ensure_chroma_versions_root(cfg: AppConfig) -> Path:
    root = _chroma_versions_root(cfg)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _versioned_chroma_dir(cfg: AppConfig, build_id: str) -> Path:
    return _ensure_chroma_versions_root(cfg) / build_id


def _load_index_manifest(cfg: AppConfig, store: Any = None) -> dict[str, Any] | None:
    if store is not None:
        try:
            manifest = store.get_index_manifest()
            if isinstance(manifest, dict):
                return manifest
        except Exception:
            pass
    try:
        from rag_lite.store import ExperimentStore

        manifest = ExperimentStore(cfg.sqlite_path).get_index_manifest()
        if isinstance(manifest, dict):
            return manifest
    except Exception:
        pass
    return None


def _manifest_active_chroma_dir(cfg: AppConfig, manifest: dict[str, Any] | None) -> Path | None:
    if not isinstance(manifest, dict):
        return None
    subdir = str(manifest.get("active_chroma_subdir") or "").strip()
    if not subdir:
        return None
    cand = Path(subdir)
    if not cand.is_absolute():
        cand = cfg.data_dir / cand
    return cand if cand.is_dir() else None


def resolve_active_chroma_dir(
    cfg: AppConfig,
    store: Any = None,
    *,
    allow_legacy: bool = True,
) -> Path | None:
    manifest = _load_index_manifest(cfg, store=store)
    active = _manifest_active_chroma_dir(cfg, manifest)
    if active is not None:
        return active
    if allow_legacy and cfg.chroma_dir.is_dir():
        return cfg.chroma_dir
    return None


def active_chroma_label(cfg: AppConfig, store: Any = None) -> str:
    active = resolve_active_chroma_dir(cfg, store=store)
    if active is None:
        return "（未发现可用索引目录）"
    try:
        return str(active.resolve())
    except Exception:
        return str(active)


def _temp_chroma_dir(cfg: AppConfig) -> Path:
    parent = cfg.chroma_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    return parent / f"{cfg.chroma_dir.name}.__building__.{os.getpid()}.{time.time_ns()}"


def _new_build_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]


def _manifest_subdir_for_chroma_dir(cfg: AppConfig, chroma_dir: Path) -> str | None:
    try:
        resolved = chroma_dir.resolve()
        data_root = cfg.data_dir.resolve()
        try:
            return str(resolved.relative_to(data_root)).replace("\\", "/")
        except Exception:
            return str(resolved)
    except Exception:
        return None


def _build_compatible_collection_config_json(
    chroma_dir: Path,
    collection_name: str,
) -> str | None:
    """为 Chroma 1.5.x 补齐 collections.config_json_str，兼容旧版/空配置。"""
    try:
        from chromadb.api.configuration import CollectionConfigurationInternal
    except Exception:
        return None

    db = chroma_dir / "chroma.sqlite3"
    if not db.is_file():
        return None

    try:
        cfg_json = CollectionConfigurationInternal().to_json()
    except Exception:
        return None

    try:
        con = sqlite3.connect(str(db))
        cur = con.cursor()
        row = cur.execute(
            "select id, schema_str from collections where name = ? limit 1",
            (collection_name,),
        ).fetchone()
        if not row:
            con.close()
            return json.dumps(cfg_json, ensure_ascii=False)
        coll_id, schema_str = row
        meta_rows = cur.execute(
            "select key, str_value, int_value, float_value, bool_value from collection_metadata where collection_id = ?",
            (coll_id,),
        ).fetchall()
        con.close()
    except Exception:
        return json.dumps(cfg_json, ensure_ascii=False)

    try:
        hnsw_cfg = cfg_json.get("hnsw_configuration") or {}
        if isinstance(schema_str, str) and schema_str.strip():
            schema = json.loads(schema_str)
            hnsw = (
                (((schema.get("defaults") or {}).get("float_list") or {}).get("vector_index") or {}).get("config") or {}
            ).get("hnsw") or {}
            if isinstance(hnsw, dict):
                if "space" in hnsw:
                    hnsw_cfg["space"] = hnsw.get("space") or hnsw_cfg.get("space")
                if "ef_construction" in hnsw:
                    hnsw_cfg["ef_construction"] = int(hnsw.get("ef_construction") or hnsw_cfg.get("ef_construction") or 100)
                if "ef_search" in hnsw:
                    hnsw_cfg["ef_search"] = int(hnsw.get("ef_search") or hnsw_cfg.get("ef_search") or 100)
                if "num_threads" in hnsw:
                    hnsw_cfg["num_threads"] = int(hnsw.get("num_threads") or hnsw_cfg.get("num_threads") or 12)
                if "resize_factor" in hnsw:
                    hnsw_cfg["resize_factor"] = float(hnsw.get("resize_factor") or hnsw_cfg.get("resize_factor") or 1.2)
                if "max_neighbors" in hnsw:
                    hnsw_cfg["M"] = int(hnsw.get("max_neighbors") or hnsw_cfg.get("M") or 16)
                if "batch_size" in hnsw:
                    hnsw_cfg["batch_size"] = int(hnsw.get("batch_size") or hnsw_cfg.get("batch_size") or 100)
                if "sync_threshold" in hnsw:
                    hnsw_cfg["sync_threshold"] = int(hnsw.get("sync_threshold") or hnsw_cfg.get("sync_threshold") or 1000)
        for key, str_value, int_value, float_value, bool_value in meta_rows:
            if key == "hnsw:batch_size":
                hnsw_cfg["batch_size"] = int(int_value or float_value or str_value or hnsw_cfg.get("batch_size") or 100)
            elif key == "hnsw:sync_threshold":
                hnsw_cfg["sync_threshold"] = int(int_value or float_value or str_value or hnsw_cfg.get("sync_threshold") or 1000)
        cfg_json["hnsw_configuration"] = hnsw_cfg
    except Exception:
        pass
    return json.dumps(cfg_json, ensure_ascii=False)


def _repair_chroma_collection_config_if_needed(chroma_dir: Path, collection_name: str) -> bool:
    db = chroma_dir / "chroma.sqlite3"
    if not db.is_file():
        return False
    try:
        con = sqlite3.connect(str(db))
        cur = con.cursor()
        row = cur.execute(
            "select config_json_str from collections where name = ? limit 1",
            (collection_name,),
        ).fetchone()
        if not row:
            con.close()
            return False
        raw = row[0]
        needs_repair = False
        if raw is None:
            needs_repair = True
        else:
            try:
                obj = json.loads(str(raw).strip() or "{}")
                needs_repair = not (isinstance(obj, dict) and obj.get("_type"))
            except Exception:
                needs_repair = True
        if not needs_repair:
            con.close()
            return False
        new_raw = _build_compatible_collection_config_json(chroma_dir, collection_name)
        if not new_raw:
            con.close()
            return False
        cur.execute(
            "update collections set config_json_str = ? where name = ?",
            (new_raw, collection_name),
        )
        con.commit()
        con.close()
        return True
    except Exception:
        return False


def _open_chroma_collection_for_dir(chroma_dir: Path, collection_name: str) -> Any | None:
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    if not chroma_dir.is_dir():
        return None
    last_exc: Exception | None = None
    repaired = False
    for attempt in range(4):
        try:
            client = chromadb.PersistentClient(
                path=str(chroma_dir),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            return client.get_collection(collection_name)
        except Exception as e:
            last_exc = e
            msg = str(e or "").strip().lower()
            if (not repaired) and ("keyerror: '_type'" in msg or '"_type"' in msg or "'_type'" in msg):
                repaired = _repair_chroma_collection_config_if_needed(chroma_dir, collection_name)
            _clear_chroma_process_cache()
            time.sleep(0.25 * (attempt + 1))
    if last_exc is not None:
        raise last_exc
    return None


def _activate_built_chroma_dir(cfg: AppConfig, built_dir: Path, build_id: str) -> Path:
    final_dir = _versioned_chroma_dir(cfg, build_id)
    last_exc: Exception | None = None
    _clear_chroma_process_cache()
    for attempt in range(6):
        try:
            if final_dir.exists():
                shutil.rmtree(final_dir, ignore_errors=True)
            final_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(built_dir), str(final_dir))
            last_exc = None
            break
        except Exception as e:
            last_exc = e
            time.sleep(0.2 * (attempt + 1))
    if last_exc is not None:
        raise last_exc
    _clear_chroma_process_cache()
    return final_dir


def _close_chroma_client(client: Any) -> None:
    close = getattr(client, "close", None)
    if callable(close):
        close()


def _sqlite_embedding_count(chroma_dir: Path) -> int:
    db = chroma_dir / "chroma.sqlite3"
    if not db.is_file():
        return 0
    con = sqlite3.connect(str(db))
    try:
        cur = con.cursor()
        row = cur.execute("select count(*) from embeddings").fetchone()
        return int((row or [0])[0] or 0)
    finally:
        con.close()


def _wait_for_chroma_sqlite_embeddings(
    chroma_dir: Path,
    expected_count: int,
    *,
    attempts: int = 40,
    sleep_s: float = 0.5,
) -> tuple[bool, int]:
    last = 0
    for _ in range(max(1, attempts)):
        try:
            last = _sqlite_embedding_count(chroma_dir)
        except Exception:
            last = 0
        if last >= expected_count:
            return True, last
        time.sleep(max(0.0, sleep_s))
    return False, last


def _wait_for_chroma_indexing_complete(
    chroma_collection: Any,
    *,
    attempts: int = 80,
    sleep_s: float = 0.25,
) -> tuple[bool, dict[str, int]]:
    last = {
        "num_indexed_ops": 0,
        "num_unindexed_ops": 0,
        "total_ops": 0,
    }
    for _ in range(max(1, attempts)):
        try:
            status = chroma_collection.get_indexing_status()
            last = {
                "num_indexed_ops": int(getattr(status, "num_indexed_ops", 0) or 0),
                "num_unindexed_ops": int(getattr(status, "num_unindexed_ops", 0) or 0),
                "total_ops": int(getattr(status, "total_ops", 0) or 0),
            }
            if last["total_ops"] > 0 and last["num_unindexed_ops"] == 0:
                return True, last
        except Exception:
            pass
        time.sleep(max(0.0, sleep_s))
    return False, last


def _chroma_hnsw_artifact_files(chroma_dir: Path) -> list[Path]:
    if not chroma_dir.is_dir():
        return []
    names = {
        "index_metadata.pickle",
        "header.bin",
        "data_level0.bin",
        "length.bin",
        "link_lists.bin",
    }
    try:
        return sorted(p for p in chroma_dir.rglob("*") if p.is_file() and p.name in names)
    except Exception:
        return []


def _has_complete_hnsw_artifacts(paths: list[Path]) -> bool:
    names = {p.name for p in (paths or [])}
    required = {
        "index_metadata.pickle",
        "header.bin",
        "data_level0.bin",
        "length.bin",
        "link_lists.bin",
    }
    return required.issubset(names)


def _wait_for_chroma_hnsw_artifacts(
    chroma_dir: Path,
    *,
    attempts: int = 40,
    sleep_s: float = 0.5,
) -> tuple[bool, list[Path]]:
    last: list[Path] = []
    for _ in range(max(1, attempts)):
        try:
            last = _chroma_hnsw_artifact_files(chroma_dir)
        except Exception:
            last = []
        if _has_complete_hnsw_artifacts(last):
            return True, last
        time.sleep(max(0.0, sleep_s))
    return False, last


def _run_self_check_until_ready(
    cfg: AppConfig,
    *,
    embed_model_override: str | None = None,
    chroma_dir_override: Path | None = None,
    require_query: bool,
    attempts: int,
    sleep_s: float,
) -> dict[str, Any]:
    last: dict[str, Any] = {
        "ok": False,
        "count_ok": False,
        "get_ok": False,
        "query_ok": False,
        "count": 0,
        "sample_id": "",
        "sample_query": "",
        "diagnostics_total_chunks": 0,
        "diagnostics_total_files": 0,
        "missing_uploaded_files": [],
        "top_files": [],
        "error": "self-check not started",
    }
    for i in range(max(1, attempts)):
        _clear_chroma_process_cache()
        last = _run_self_check_in_subprocess(
            cfg,
            embed_model_override=embed_model_override,
            chroma_dir_override=chroma_dir_override,
            require_query=require_query,
        )
        if bool(last.get("ok")):
            return last
        if i + 1 < attempts:
            time.sleep(max(0.0, sleep_s))
    return last


def _is_transient_chroma_reopen_error(err: Any) -> bool:
    s = str(err or "").strip().lower()
    if not s:
        return False
    # 仅对 metadata/config schema 兼容性问题做兜底；
    # HNSW 打不开（如 Cannot open header file）必须视为不可查询，不能再激活为正式索引。
    needles = (
        "keyerror: '_type'",
    )
    return any(x in s for x in needles)


def _release_chroma_runtime(*objs: Any, clear_process_cache: bool = True) -> None:
    for obj in objs:
        try:
            close = getattr(obj, "close", None)
            if callable(close):
                close()
        except Exception:
            pass
    gc.collect()
    if clear_process_cache:
        _clear_chroma_process_cache()


def _finalize_temp_chroma_build(
    chroma_dir: Path,
    chroma_collection: Any,
    chroma_client: Any,
    *,
    expected_count: int,
) -> tuple[bool, dict[str, Any]]:
    indexing_ok, indexing_status = _wait_for_chroma_indexing_complete(
        chroma_collection,
        attempts=120,
        sleep_s=0.25,
    )
    _release_chroma_runtime(chroma_collection, chroma_client, clear_process_cache=False)
    _clear_chroma_process_cache()
    persisted_ok, persisted_count = _wait_for_chroma_sqlite_embeddings(
        chroma_dir,
        expected_count,
        attempts=60,
        sleep_s=0.5,
    )
    hnsw_ok, hnsw_files = _wait_for_chroma_hnsw_artifacts(
        chroma_dir,
        attempts=60,
        sleep_s=0.5,
    )
    # NOTE:
    # Chroma 1.x 在不同版本/平台下，HNSW 持久化文件形态可能不再固定为
    # {index_metadata.pickle, header.bin, data_level0.bin, length.bin, link_lists.bin} 全套。
    # 这里仅把 hnsw 文件集合作为诊断信息；是否可用交给后续「重开自检（count/get/query）」兜底。
    return (
        persisted_ok,
        {
            "indexing_ok": indexing_ok,
            "indexing_status": indexing_status,
            "persisted_ok": persisted_ok,
            "persisted_count": persisted_count,
            "hnsw_ok": hnsw_ok,
            "hnsw_files": hnsw_files,
        },
    )


def _project_python_executable(cfg: AppConfig) -> str:
    """Use the project .venv interpreter for all ragZone subprocesses."""
    root = Path(cfg.root)
    candidates = [
        root / ".venv" / "Scripts" / "python.exe",
        root / ".venv" / "bin" / "python",
    ]
    for cand in candidates:
        try:
            if cand.is_file():
                return str(cand)
        except Exception:
            pass
    return sys.executable



def _run_self_check_in_subprocess(
    cfg: AppConfig,
    *,
    embed_model_override: str | None = None,
    chroma_dir_override: Path | None = None,
    require_query: bool = True,
) -> dict[str, Any]:
    payload = {
        "project_root": str(cfg.root),
        "embed_model_override": embed_model_override,
        "chroma_dir_override": str(chroma_dir_override) if chroma_dir_override is not None else None,
        "require_query": bool(require_query),
    }
    code = """
import json, sys
from pathlib import Path
payload = json.loads(sys.argv[1])
proj = Path(payload['project_root'])
sys.path.insert(0, str(proj))
from rag_lite.config import load_config
from rag_lite.ingest import self_check_index
cfg = load_config(proj)
chroma_dir = Path(payload['chroma_dir_override']) if payload.get('chroma_dir_override') else None
res = self_check_index(
    cfg,
    embed_model_override=payload.get('embed_model_override'),
    chroma_dir_override=chroma_dir,
    require_query=bool(payload.get('require_query', True)),
)
print(json.dumps(res, ensure_ascii=False))
"""
    try:
        r = subprocess.run(
            [_project_python_executable(cfg), "-c", code, json.dumps(payload, ensure_ascii=False)],
            capture_output=True,
            text=False,
            timeout=180,
            check=False,
        )
    except Exception as e:
        return {
            "ok": False,
            "count_ok": False,
            "get_ok": False,
            "query_ok": False,
            "count": 0,
            "sample_id": "",
            "sample_query": "",
            "diagnostics_total_chunks": 0,
            "diagnostics_total_files": 0,
            "missing_uploaded_files": [],
            "top_files": [],
            "error": f"subprocess self-check failed: {type(e).__name__}: {e}",
        }
    stdout = (r.stdout or b"").decode("utf-8", errors="replace").strip()
    stderr = (r.stderr or b"").decode("utf-8", errors="replace").strip()
    if r.returncode != 0:
        return {
            "ok": False,
            "count_ok": False,
            "get_ok": False,
            "query_ok": False,
            "count": 0,
            "sample_id": "",
            "sample_query": "",
            "diagnostics_total_chunks": 0,
            "diagnostics_total_files": 0,
            "missing_uploaded_files": [],
            "top_files": [],
            "error": f"subprocess returncode={r.returncode}: {stderr or stdout or 'unknown error'}",
        }
    if not stdout:
        return {
            "ok": False,
            "count_ok": False,
            "get_ok": False,
            "query_ok": False,
            "count": 0,
            "sample_id": "",
            "sample_query": "",
            "diagnostics_total_chunks": 0,
            "diagnostics_total_files": 0,
            "missing_uploaded_files": [],
            "top_files": [],
            "error": "subprocess self-check returned empty stdout",
        }
    try:
        return json.loads(stdout.splitlines()[-1])
    except Exception as e:
        return {
            "ok": False,
            "count_ok": False,
            "get_ok": False,
            "query_ok": False,
            "count": 0,
            "sample_id": "",
            "sample_query": "",
            "diagnostics_total_chunks": 0,
            "diagnostics_total_files": 0,
            "missing_uploaded_files": [],
            "top_files": [],
            "error": f"subprocess self-check parse failed: {type(e).__name__}: {e}; raw={stdout[-500:]}",
        }


def _clear_chroma_process_cache() -> None:
    try:
        from chromadb.api.shared_system_client import SharedSystemClient
        systems = list(getattr(SharedSystemClient, "_identifier_to_system", {}).items())
        for identifier, system in systems:
            try:
                system.stop()
            except Exception:
                pass
            try:
                SharedSystemClient._identifier_to_system.pop(identifier, None)
            except Exception:
                pass
            try:
                SharedSystemClient._identifier_to_refcount.pop(identifier, None)
            except Exception:
                pass
        SharedSystemClient.clear_system_cache()
    except Exception:
        pass
    gc.collect()


def _chroma_count_index_only(coll: Any) -> int:
    """Prefer compacted-index count so readiness isn't overstated by WAL-visible rows."""
    try:
        from chromadb.api.types import ReadLevel

        return int(coll.count(read_level=ReadLevel.INDEX_ONLY))
    except Exception:
        return int(coll.count())


def _node_file_name(node: Any) -> str:
    meta = getattr(node, "metadata", None) or {}
    n = meta.get("file_name")
    if n is None and meta.get("file_path"):
        n = Path(str(meta["file_path"])).name
    return str(n or "unknown")


def _doc_file_name(doc: Any) -> str:
    meta = getattr(doc, "metadata", None) or {}
    n = meta.get("file_name")
    if n is None and meta.get("file_path"):
        n = Path(str(meta["file_path"])).name
    return str(n or "unknown")


def uploaded_files_snapshot(cfg: AppConfig) -> list[dict[str, Any]]:
    """扫描上传目录，用于索引清单与 UI 状态。"""
    if not cfg.uploads_dir.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for p in sorted(cfg.uploads_dir.iterdir()):
        if not p.is_file() or not _allowed_suffix(p.name):
            continue
        st = p.stat()
        out.append({"name": p.name, "mtime_ns": st.st_mtime_ns, "size": st.st_size})
    return out


def _make_node_splitter(chunk_mode: str, chunk_size: int, chunk_overlap: int):
    mode = (chunk_mode or "sentence").strip().lower()
    if mode == "token":
        return TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if mode == "paragraph":
        return SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            paragraph_separator="\n\n",
        )
    return SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def _splitter_label(chunk_mode: str) -> str:
    mode = (chunk_mode or "sentence").strip().lower()
    if mode == "token":
        return "Token 切分"
    if mode == "paragraph":
        return "段落优先（双换行）"
    return "按句切分（默认）"


def save_uploads(
    cfg: AppConfig,
    files: list[str] | None,
    max_size_mb: float,
) -> tuple[int, str, list[str]]:
    """Copy Gradio file paths into uploads_dir. Returns (count, message, saved_paths)."""
    if not files:
        return 0, "未选择文件。", []
    cfg.uploads_dir.mkdir(parents=True, exist_ok=True)
    max_bytes = int(max_size_mb * 1024 * 1024)
    n = 0
    saved: list[str] = []
    for p in files:
        path = Path(p)
        if not path.is_file():
            continue
        if not _allowed_suffix(path.name):
            continue
        if path.stat().st_size > max_bytes:
            return n, f"文件过大（>{max_size_mb}MB）: {path.name}", saved
        dest = cfg.uploads_dir / path.name
        shutil.copy2(path, dest)
        saved.append(str(dest.resolve()))
        n += 1
    if n == 0:
        return 0, "没有可导入的文件（支持 PDF/TXT/Markdown/DOCX）。", []
    return n, f"已保存 {n} 个文件到知识库目录。", saved


def iter_build_index(
    cfg: AppConfig,
    chunk_size: int,
    chunk_overlap: int,
    chunk_mode: str = "sentence",
    embed_model: str | None = None,
    batch_size: int = _EMBED_BATCH_SIZE,
    store: Any = None,
    image_enrichment_override: bool | None = None,
    image_pipeline_overrides: dict[str, Any] | None = None,
) -> Iterator[str]:
    """
    流式汇报构建进度（供 Gradio 等界面逐行刷新）。
    成功时最后一行以「完成：」开头；失败为「失败：」。
    """
    o = dict(cfg.ollama)
    em_name = embed_model or o.get("embed_model", "")
    o["embed_model"] = em_name
    embed = OllamaEmbedding(
        model_name=o["embed_model"],
        base_url=o["base_url"],
        ollama_additional_kwargs={},
    )
    Settings.embed_model = embed

    yield "[1/5] 检查上传目录 …"
    if not cfg.uploads_dir.is_dir() or not any(cfg.uploads_dir.iterdir()):
        yield "失败：上传目录为空，请先上传文档。"
        return

    listed = sorted(
        p.name
        for p in cfg.uploads_dir.rglob("*")
        if p.is_file() and _allowed_suffix(p.name)
    )
    if listed:
        head = ", ".join(listed[:14])
        if len(listed) > 14:
            head += f" …（共 {len(listed)} 个）"
        yield f"    └ 待处理文件：{head}"

    img_on = (
        bool(image_enrichment_override)
        if image_enrichment_override is not None
        else bool(cfg.ingest.get("image_enrichment"))
    )
    yield (
        "[2/5] 加载文档（本地解析 PDF/TXT/Markdown/DOCX"
        + ("；PDF/DOCX 已启用图片 OCR+视觉补充" if img_on else "")
        + "）…"
    )
    file_extractor = default_local_file_extractors(
        cfg,
        image_enrichment=image_enrichment_override,
        pipeline_overrides=image_pipeline_overrides,
    )
    paths = sorted(
        p
        for p in cfg.uploads_dir.rglob("*")
        if p.is_file() and _allowed_suffix(p.name)
    )
    if not paths:
        yield "失败：没有可处理的文档（扩展名需为 .pdf / .txt / .md / .markdown / .docx）。"
        return
    # 按文件逐个 load；单文件内仍可能很慢（多图 OCR/视觉），用后台线程 + 定时 yield 心跳避免界面假死
    docs: list[Any] = []
    n_paths = len(paths)
    _heartbeat_s = 1.0
    for i, path in enumerate(paths, start=1):
        yield f"    └ 正在解析 [{i}/{n_paths}] {path.name} …"

        def _load_one(p: Path = path) -> list[Any]:
            r = SimpleDirectoryReader(
                input_files=[str(p.resolve())],
                file_extractor=file_extractor,
            )
            return r.load_data()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(_load_one)
            # 立刻再 yield 一行，避免首条心跳要等满一个间隔才出现（用户误以为卡死）
            yield (
                f"    └ 后台解析已启动 [{i}/{n_paths}] {path.name} "
                f"（大 PDF / 多图 OCR+视觉可能需数分钟；约每 {_heartbeat_s:.0f}s 刷新进度）…"
            )
            t0 = time.perf_counter()
            while not fut.done():
                concurrent.futures.wait([fut], timeout=_heartbeat_s)
                if fut.done():
                    break
                elapsed = int(time.perf_counter() - t0)
                yield (
                    f"    └ 仍解析中 [{i}/{n_paths}] {path.name} "
                    f"（已 {elapsed}s；图片多或 Paddle/视觉慢时会较久，非报错）…"
                )
            try:
                part = fut.result()
            except Exception as e:
                yield f"失败：解析 {path.name} 出错：{e}"
                return
        docs.extend(part)
    if not docs:
        yield "失败：未能解析出任何文档内容。"
        return
    src_names = sorted({_doc_file_name(d) for d in docs})
    if src_names:
        yield f"    └ 已解析文档来源：{', '.join(src_names)}"
    if img_on:
        lines = []
        for d in docs:
            s = (getattr(d, "metadata", None) or {}).get("image_enrichment_summary")
            if s:
                lines.append(f"{_doc_file_name(d)}({s})")
        if lines:
            yield f"    └ 图片增强统计：{'; '.join(lines)}"

    slabel = _splitter_label(chunk_mode)
    yield f"[3/5] 切分文本（{slabel}；size={chunk_size}, overlap={chunk_overlap}）…"
    splitter = _make_node_splitter(chunk_mode, chunk_size, chunk_overlap)
    nodes = splitter.get_nodes_from_documents(docs)
    content_nodes = [
        n for n in nodes if str(n.get_content(metadata_mode=MetadataMode.NONE) or "").strip()
    ]
    skipped = len(nodes) - len(content_nodes)
    if skipped:
        yield f"    └ 提示：跳过 {skipped} 个无正文的空块。"
    n_total = len(content_nodes)
    if n_total == 0:
        yield "失败：没有可写入向量的文本块。"
        return

    import chromadb
    from chromadb.config import Settings as ChromaSettings

    build_id = _new_build_id()
    temp_chroma_dir = _temp_chroma_dir(cfg)
    try:
        temp_chroma_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        shutil.rmtree(temp_chroma_dir, ignore_errors=True)
        temp_chroma_dir.mkdir(parents=True, exist_ok=False)

    yield "[4/5] 准备临时 Chroma 索引目录 …"
    chroma_client = chromadb.PersistentClient(
        path=str(temp_chroma_dir),
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    # Windows + Chroma 1.5.x 下，过大的 HNSW batch/sync 阈值会触发巨量内存分配，
    # 表面上 SQLite embeddings 已写入，实际 HNSW 文件却落不全，随后 query 会报
    # Cannot open header file。这里默认使用保守值，并保证 sync_threshold >= batch_size。
    hnsw_batch_size = max(1, int(cfg.ingest.get("chroma_hnsw_batch_size") or 100))
    hnsw_sync_threshold = max(
        hnsw_batch_size,
        int(cfg.ingest.get("chroma_hnsw_sync_threshold") or 1000),
    )
    chroma_collection = chroma_client.get_or_create_collection(
        cfg.collection_name,
        metadata={
            "hnsw:batch_size": max(1, hnsw_batch_size),
            "hnsw:sync_threshold": max(1, hnsw_sync_threshold),
        },
    )
    yield (
        "    └ Chroma 可靠模式："
        f"hnsw_batch_size={max(1, hnsw_batch_size)}，"
        f"hnsw_sync_threshold={max(1, hnsw_sync_threshold)}"
    )
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    yield (
        f"[5/5] 向量化并写入 Chroma（共 {n_total} 块，每批 ≤{batch_size}，"
        f"嵌入模型「{em_name}」；请看下方进度）…"
    )
    try:
        with _suppress_embedding_tqdm():
            index = VectorStoreIndex(
                [],
                storage_context=storage_context,
                embed_model=embed,
                insert_batch_size=batch_size,
                show_progress=False,
            )
            done = 0
            for batch in iter_batch(content_nodes, batch_size):
                batch_list = list(batch)
                index.insert_nodes(batch_list, show_progress=False)
                done += len(batch_list)
                names = sorted({_node_file_name(n) for n in batch_list})
                shown = ", ".join(names[:10])
                if len(names) > 10:
                    shown += f" 等 {len(names)} 个来源"
                yield f"    └ 已向量化 {done}/{n_total} 块 · 当前批次片段来自：{shown}"
    except Exception as e:
        shutil.rmtree(temp_chroma_dir, ignore_errors=True)
        yield f"失败：向量化或写入 Chroma 时出错：{e}"
        return

    _release_chroma_runtime(index, vector_store, storage_context, clear_process_cache=False)

    yield "    └ 正在等待 Chroma 完成索引落盘 …"
    finalized_ok, finalize_state = _finalize_temp_chroma_build(
        temp_chroma_dir,
        chroma_collection,
        chroma_client,
        expected_count=n_total,
    )
    indexing_ok = bool(finalize_state.get("indexing_ok"))
    indexing_status = finalize_state.get("indexing_status") or {}
    persisted_ok = bool(finalize_state.get("persisted_ok"))
    persisted_count = int(finalize_state.get("persisted_count") or 0)
    hnsw_ok = bool(finalize_state.get("hnsw_ok"))
    hnsw_files = list(finalize_state.get("hnsw_files") or [])
    indexing_total = int(indexing_status.get("total_ops") or 0)
    indexing_unindexed = int(indexing_status.get("num_unindexed_ops") or 0)
    if not finalized_ok:
        shutil.rmtree(temp_chroma_dir, ignore_errors=True)
        yield (
            "失败：临时索引最终持久化检查未通过（SQLite embeddings 未达到期望值），已终止替换。"
            f" indexing_total={indexing_total}；"
            f"unindexed={indexing_unindexed}；"
            f"SQLite embeddings={persisted_count}/{n_total}；"
            f"hnsw_artifacts={len(hnsw_files)}"
        )
        return

    yield "    └ 正在执行临时索引重开自检（count / get / diagnostics）…"
    temp_health = _run_self_check_until_ready(
        cfg,
        embed_model_override=em_name,
        chroma_dir_override=temp_chroma_dir,
        require_query=False,
        attempts=18,
        sleep_s=1.0,
    )
    temp_reopen_health = _run_self_check_until_ready(
        cfg,
        embed_model_override=em_name,
        chroma_dir_override=temp_chroma_dir,
        require_query=True,
        attempts=8,
        sleep_s=1.0,
    )
    if not bool(temp_reopen_health.get("ok")):
        shutil.rmtree(temp_chroma_dir, ignore_errors=True)
        yield (
            "失败：临时索引重开自检未通过，已终止替换。"
            f" indexing_total={indexing_total}；"
            f"unindexed={indexing_unindexed}；"
            f"SQLite embeddings={persisted_count}/{n_total}；"
            f"hnsw_artifacts={len(hnsw_files)}；"
            f"count={int((temp_reopen_health or {}).get('count') or 0)}/{n_total}；"
            f"diag_chunks={int((temp_health or {}).get('diagnostics_total_chunks') or 0)}；"
            f"diag_files={int((temp_health or {}).get('diagnostics_total_files') or 0)}；"
            f"missing={len((temp_health or {}).get('missing_uploaded_files') or [])}；"
            f"错误={(temp_reopen_health or {}).get('error') or '无'}"
        )
        return

    yield (
        "    └ 临时索引重开自检通过："
        f"indexing_total={indexing_total}；"
        f"unindexed={indexing_unindexed}；"
        f"SQLite embeddings={persisted_count}/{n_total}；"
        f"hnsw_artifacts={len(hnsw_files)}；"
        f"count={int(temp_reopen_health.get('count') or 0)}；"
        f"diag_chunks={int(temp_reopen_health.get('diagnostics_total_chunks') or 0)}；"
        f"diag_files={int(temp_reopen_health.get('diagnostics_total_files') or 0)}；"
        f"sample_id={temp_reopen_health.get('sample_id') or '—'}"
    )

    yield "    └ 新索引构建完成，正在激活新版本索引 …"
    try:
        active_dir = _activate_built_chroma_dir(cfg, temp_chroma_dir, build_id)
    except Exception as e:
        shutil.rmtree(temp_chroma_dir, ignore_errors=True)
        yield f"失败：新索引已构建，但激活新版本时出错：{e}"
        return

    yield "    └ 正在执行索引健康自检（count / get / query）…"
    health = _run_self_check_until_ready(
        cfg,
        embed_model_override=em_name,
        chroma_dir_override=active_dir,
        require_query=True,
        attempts=18,
        sleep_s=1.0,
    )
    if not bool((health or {}).get("ok")):
        _clear_chroma_process_cache()
        live_sqlite_count = 0
        try:
            live_sqlite_count = _sqlite_embedding_count(active_dir)
        except Exception:
            live_sqlite_count = 0
        shutil.rmtree(active_dir, ignore_errors=True)
        yield (
            "失败：新索引版本激活后健康自检未通过，已丢弃该版本。"
            f" 自检错误：{(health or {}).get('error') or '未知错误'}"
            f"；live SQLite embeddings={live_sqlite_count}"
        )
        return
    yield (
        f"    └ 自检通过：count={int(health.get('count') or 0)}；"
        f"sample_id={health.get('sample_id') or '—'}"
    )

    snap = uploaded_files_snapshot(cfg)
    doc_manifest_meta = _collect_doc_manifest_rows(docs)
    built_iso = _utc_now_iso()
    for item in snap:
        item["indexed_at"] = built_iso
        extra = doc_manifest_meta.get(_filename_sig(str(item.get("name") or ""))) or {}
        if extra:
            item.update(extra)
    readiness = {
        "ok": bool((health or {}).get("ok")),
        "count_ok": bool((health or {}).get("count_ok")),
        "get_ok": bool((health or {}).get("get_ok")),
        "query_ok": bool((health or {}).get("query_ok")),
        "error": str((health or {}).get("error") or ""),
        "missing_uploaded_files": list((health or {}).get("missing_uploaded_files") or []),
        "diagnostics_total_chunks": int((health or {}).get("diagnostics_total_chunks") or 0),
        "diagnostics_total_files": int((health or {}).get("diagnostics_total_files") or 0),
        "temp_reopen_health": dict(temp_reopen_health or {}),
        "final_health": dict(health or {}),
        "indexing_total": indexing_total,
        "indexing_unindexed": indexing_unindexed,
        "persisted_count": persisted_count,
        "total_chunks": n_total,
        "hnsw_artifacts": [str(p) for p in hnsw_files],
    }
    active_subdir = _manifest_subdir_for_chroma_dir(cfg, active_dir)
    manifest_saved = False
    manifest_save_error = ""
    if store is not None:
        try:
            store.save_index_manifest(
                embed_model=em_name,
                chunk_mode=(chunk_mode or "sentence").strip().lower(),
                chunk_size=int(chunk_size),
                chunk_overlap=int(chunk_overlap),
                files=snap,
                build_id=build_id,
                active_chroma_subdir=active_subdir,
                activated_at=built_iso,
                readiness=readiness,
            )
            manifest_saved = True
        except Exception as e:
            manifest_save_error = f"{type(e).__name__}: {e}"
            manifest_saved = False
    if not manifest_saved:
        try:
            from rag_lite.store import ExperimentStore

            ExperimentStore(cfg.sqlite_path).save_index_manifest(
                embed_model=em_name,
                chunk_mode=(chunk_mode or "sentence").strip().lower(),
                chunk_size=int(chunk_size),
                chunk_overlap=int(chunk_overlap),
                files=snap,
                build_id=build_id,
                active_chroma_subdir=active_subdir,
                activated_at=built_iso,
                readiness=readiness,
            )
            manifest_saved = True
        except Exception as e:
            if not manifest_save_error:
                manifest_save_error = f"{type(e).__name__}: {e}"
            manifest_saved = False

    if not manifest_saved:
        _clear_chroma_process_cache()
        try:
            shutil.rmtree(active_dir, ignore_errors=True)
        except Exception:
            pass
        yield (
            "失败：新索引已构建并通过自检，但写入 manifest 失败，已回滚该新版本。"
            f" 错误：{manifest_save_error or '未知错误'}"
        )
        return

    yield f"完成：索引已就绪，共 {n_total} 个块。可到「对话」页提问。"


def _nfc(s: str) -> str:
    """统一 Unicode 规范化（避免 NFC/NFD 导致与磁盘/Chroma 字符串不一致）。"""
    return unicodedata.normalize("NFC", str(s or "").strip())


def _norm_name(s: str) -> str:
    # NFKC：兼容全角/半角、兼容形近字符，减少「磁盘名与 Chroma 字符串」不一致
    return unicodedata.normalize("NFKC", _nfc(s)).casefold()


def _filename_sig(name: str) -> str:
    """
    文件名比对用签名：小写 + 空白压成单空格 + 去掉下划线两侧空格。
    解决「V1.0 _20240620.docx」与「V1.0_20240620.docx」在磁盘/Chroma 中不一致的问题。
    """
    n = _norm_name(Path(name).name)
    n = re.sub(r"\s+", " ", n).strip()
    n = re.sub(r"\s*_\s*", "_", n)
    return n


def _filename_equiv(a: str, b: str) -> bool:
    if not a or not b:
        return False
    return _filename_sig(a) == _filename_sig(b)


def _filename_query_variants(fn: str) -> list[str]:
    """Chroma where 查询可能命中多种字面量，尽量覆盖。"""
    fn = _nfc(Path(str(fn).strip()).name)
    out: list[str] = []
    seen: set[str] = set()

    def add(x: str) -> None:
        if x and x not in seen:
            seen.add(x)
            out.append(x)

    add(fn)
    add(re.sub(r"\s*_\s*", "_", fn))
    add(re.sub(r"\s+", " ", fn).strip())
    return out


def _node_json_might_match_file(obj: Any, want_base: str, depth: int = 0) -> bool:
    """在 LlamaIndex 序列化节点的 JSON 对象中递归查找 file_name / file_path。"""
    if depth > 8:
        return False
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("file_name", "file_path") and isinstance(v, str) and v.strip():
                vv = v.strip()
                if _filename_equiv(want_base, vv):
                    return True
            if _node_json_might_match_file(v, want_base, depth + 1):
                return True
    elif isinstance(obj, list):
        for it in obj:
            if _node_json_might_match_file(it, want_base, depth + 1):
                return True
    return False


def _regex_extract_file_names_from_node_content(raw: str) -> list[str]:
    """
    json.loads 失败或结构变化时，从 _node_content 原始字符串中提取 file_name 字段值。
    """
    if not raw or len(raw) < 12:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for m in re.finditer(r'"file_name"\s*:\s*"((?:[^"\\]|\\.)*)"', raw):
        s = (m.group(1) or "").strip()
        s = s.replace(r"\"", '"').replace(r"\\", "\\")
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _raw_node_content_has_file_fields(raw: str, want_name: str) -> bool:
    """JSON 解析失败（超长截断等）时，在原始字符串中宽松匹配文件名。"""
    want = _nfc(Path(want_name).name)
    if len(want) < 6:
        return False
    alt = re.sub(r"\s*_\s*", "_", want)
    if want not in raw and alt not in raw:
        return False
    for key in ('"file_name"', '"file_path"'):
        pos = 0
        while True:
            i = raw.find(key, pos)
            if i < 0:
                break
            win = raw[i : i + 520]
            if want in win or alt in win:
                return True
            pos = i + 1
    return False


def _ref_doc_id_from_meta(meta: dict[str, Any]) -> str:
    r = str(meta.get("ref_doc_id") or meta.get("document_id") or "").strip()
    if r in ("", "None"):
        return ""
    return r


def _candidate_file_names_from_meta(meta: dict[str, Any]) -> list[str]:
    """
    从单条 Chroma metadata 中收集所有可能的源文件名（用于匹配与 ref_doc_id 归并）。
    """
    cands: list[str] = []
    seen: set[str] = set()

    def add(x: str) -> None:
        x = str(x or "").strip()
        if not x or x in seen:
            return
        seen.add(x)
        cands.append(x)

    if not isinstance(meta, dict):
        return []
    add(str(meta.get("file_name") or "").strip())
    fp0 = str(meta.get("file_path") or "").strip()
    if fp0:
        add(Path(fp0).name)
    raw = meta.get("_node_content")
    if isinstance(raw, str) and raw:
        try:
            node_d = json.loads(raw)
            if isinstance(node_d, dict):
                inner = node_d.get("metadata") or {}
                if isinstance(inner, dict):
                    add(str(inner.get("file_name") or "").strip())
                    fp2 = str(inner.get("file_path") or "").strip()
                    if fp2:
                        add(Path(fp2).name)
        except Exception:
            for x in _regex_extract_file_names_from_node_content(raw):
                add(x)
        else:
            # JSON 成功时也补一遍正则，避免结构里另有重复字段
            for x in _regex_extract_file_names_from_node_content(raw):
                add(x)
    return cands


def _deep_scan_meta_for_filename(meta: dict[str, Any], want_name: str) -> bool:
    """
    递归扫描 metadata 与 _node_content 解析后的 JSON 中所有字符串，
    与 want 做 _filename_equiv / 签名比对。用于 file_name 嵌在非预期字段时的兜底。
    """
    if not isinstance(meta, dict):
        return False
    want = _nfc(Path(want_name).name)
    if not want:
        return False
    want_sig = _filename_sig(want)

    def check_str(s: str) -> bool:
        t = str(s).strip()
        if len(t) < 2:
            return False
        if _filename_equiv(want, t) or _filename_equiv(want, Path(t).name):
            return True
        return bool(len(want_sig) >= 4 and _filename_sig(Path(t).name) == want_sig)

    def walk(obj: Any, depth: int) -> bool:
        if depth > 16:
            return False
        if isinstance(obj, dict):
            for v in obj.values():
                if walk(v, depth + 1):
                    return True
        elif isinstance(obj, list):
            for v in obj:
                if walk(v, depth + 1):
                    return True
        elif isinstance(obj, str):
            if check_str(obj):
                return True
        return False

    if walk(meta, 0):
        return True
    raw = meta.get("_node_content")
    if isinstance(raw, str) and raw:
        try:
            j = json.loads(raw)
            return walk(j, 0)
        except Exception:
            return _raw_node_content_has_file_fields(raw, want_name)
    return False


def _metadata_matches_filename(meta: dict[str, Any], want_name: str) -> bool:
    """
    判断 Chroma 一条记录的 metadata 是否属于指定文件名。
    LlamaIndex 会将完整节点写入 _node_content；顶层 file_name 可能被清空，需在 _node_content JSON 内匹配。
    """
    if not isinstance(meta, dict):
        return False
    want = _nfc(Path(want_name).name)
    if not want:
        return False
    n = str(meta.get("file_name") or "").strip()
    fp = str(meta.get("file_path") or "").strip()
    if _filename_equiv(want, n) or (fp and _filename_equiv(want, fp)):
        return True
    if n and _filename_equiv(want, Path(n).name):
        return True
    # 表格截断：want 为短前缀
    if n and len(want) >= 6 and (n.startswith(want.rstrip("…")) or want.rstrip("…") in n):
        return True
    for c in _candidate_file_names_from_meta(meta):
        if _filename_equiv(want, c) or _filename_equiv(want, Path(c).name):
            return True
    raw = meta.get("_node_content")
    if isinstance(raw, str) and raw:
        try:
            node_d = json.loads(raw)
            if _node_json_might_match_file(node_d, want):
                return True
            inner = node_d.get("metadata") or {}
            if isinstance(inner, dict):
                n2 = str(inner.get("file_name") or "").strip()
                fp2 = str(inner.get("file_path") or "").strip()
                if _filename_equiv(want, n2) or (fp2 and _filename_equiv(want, fp2)):
                    return True
        except Exception:
            for x in _regex_extract_file_names_from_node_content(raw):
                if _filename_equiv(want, x):
                    return True
            if _raw_node_content_has_file_fields(raw, want):
                return True
    # 元数据里扩展名或括号全半角不一致时，用语干再比一次（仅 stem≥4，降低误匹配）
    ws = _filename_sig(Path(want_name).stem)
    if len(ws) >= 4:
        n0 = str(meta.get("file_name") or "").strip()
        fp0 = str(meta.get("file_path") or "").strip()
        for cand in (n0, Path(fp0).name if fp0 else ""):
            if cand and _filename_sig(Path(cand).stem) == ws:
                return True
        for c in _candidate_file_names_from_meta(meta):
            if _filename_sig(Path(c).stem) == ws:
                return True
    return _deep_scan_meta_for_filename(meta, want_name)


def _align_chroma_parallel_list(ids: list[Any], key: str, batch: dict[str, Any]) -> list[Any]:
    """保证与 ids 等长的并行列表，避免 metadatas/documents 缺省或长度不一致导致 zip 错位。"""
    n = len(ids)
    if n == 0:
        return []
    raw = batch.get(key)
    if raw is None:
        return [None] * n
    if not isinstance(raw, list):
        return [None] * n
    if len(raw) == n:
        return raw
    if len(raw) < n:
        return list(raw) + [None] * (n - len(raw))
    return raw[:n]


def _chroma_get_all_records(coll: Any, include: list[str]) -> tuple[list[Any], list[Any], list[Any]]:
    """
    分页拉取集合中的全部向量记录。
    若调用 get 时不带 limit，部分 Chroma 版本只会返回默认一页（常见为 100 条），
    会导致「共 116 条但预览某文件始终 0 条」——后排切片从未参与匹配。
    若某批 documents/metadatas 缺失或与 ids 长度不一致，必须用 None 补齐，否则与 id 错位，
    会出现「示例里能解析到文件名、按文件预览却 0 条」的现象。
    """
    n_total = int(coll.count())
    if n_total <= 0:
        return [], [], []
    page = 100
    all_ids: list[Any] = []
    all_docs: list[Any] = []
    all_metas: list[Any] = []
    offset = 0
    want_docs = "documents" in include
    want_meta = "metadatas" in include
    while offset < n_total:
        batch = coll.get(include=include, limit=page, offset=offset)
        ids = batch.get("ids") or []
        if not ids:
            break
        all_ids.extend(ids)
        if want_docs:
            all_docs.extend(_align_chroma_parallel_list(ids, "documents", batch))
        if want_meta:
            all_metas.extend(_align_chroma_parallel_list(ids, "metadatas", batch))
        offset += len(ids)
        if len(ids) < page:
            break
    return all_ids, all_docs, all_metas


def _chroma_all_ids_paginated(coll: Any, page: int = 200) -> list[str]:
    """仅分页收集全部 id，避免依赖 count() 触发 HNSW 读取。"""
    out: list[str] = []
    offset = 0
    while True:
        try:
            batch = coll.get(include=["metadatas"], limit=page, offset=offset)
        except Exception:
            break
        ids = batch.get("ids") or []
        if not ids:
            break
        out.extend(str(x) for x in ids)
        offset += len(ids)
        if len(ids) < page:
            break
    return out


def _chroma_get_docs_metas_by_ids(
    coll: Any,
    ids: list[str],
    *,
    batch_size: int = 48,
) -> list[tuple[str, str, dict[str, Any]]]:
    """
    按 id 列表批量拉取 documents + metadatas。Chroma 保证返回顺序与请求的 ids 一致，
    比 offset 全表扫描更可靠。
    """
    triples: list[tuple[str, str, dict[str, Any]]] = []
    if not ids:
        return triples
    bs = max(8, int(batch_size))
    for i in range(0, len(ids), bs):
        chunk = [str(x) for x in ids[i : i + bs]]
        try:
            r = coll.get(ids=chunk, include=["documents", "metadatas"])
        except Exception:
            for one_id in chunk:
                try:
                    r1 = coll.get(ids=[one_id], include=["documents", "metadatas"])
                    rids = r1.get("ids") or []
                    if not rids:
                        continue
                    docs = _align_chroma_parallel_list(rids, "documents", r1)
                    metas = _align_chroma_parallel_list(rids, "metadatas", r1)
                    for nid, doc, meta in zip_longest(rids, docs, metas, fillvalue=None):
                        if nid is None:
                            continue
                        m = meta if isinstance(meta, dict) else {}
                        triples.append((str(nid), doc or "", m))
                except Exception:
                    continue
            continue
        rids = r.get("ids") or []
        docs = _align_chroma_parallel_list(rids, "documents", r)
        metas = _align_chroma_parallel_list(rids, "metadatas", r)
        for nid, doc, meta in zip_longest(rids, docs, metas, fillvalue=None):
            if nid is None:
                continue
            m = meta if isinstance(meta, dict) else {}
            triples.append((str(nid), doc or "", m))
    return triples


_MAX_DOC_ORDER = 10**18


def _int_from_meta_val(val: Any) -> int | None:
    if val is None:
        return None
    try:
        s = str(val).strip()
        if s == "":
            return None
        return int(float(s))
    except (TypeError, ValueError):
        return None


def _order_keys_from_meta(meta: dict[str, Any]) -> tuple[int, int]:
    """
    从 Chroma 顶层或 _node_content.metadata 读取 start_char_idx / end_char_idx，
    用于预览切片按「文档原文顺序」排列（与 LlamaIndex 切分节点一致）。
    """
    s = _int_from_meta_val(meta.get("start_char_idx"))
    e = _int_from_meta_val(meta.get("end_char_idx"))
    if s is not None and e is not None:
        return s, e
    raw = meta.get("_node_content")
    if isinstance(raw, str) and raw:
        try:
            d = json.loads(raw)
            inner = d.get("metadata") or {}
            if isinstance(inner, dict):
                if s is None:
                    s = _int_from_meta_val(inner.get("start_char_idx"))
                if e is None:
                    e = _int_from_meta_val(inner.get("end_char_idx"))
        except Exception:
            pass
    s = s if s is not None else _MAX_DOC_ORDER
    e = e if e is not None else _MAX_DOC_ORDER
    return s, e


def _filename_appears_in_meta_blob(meta: dict[str, Any], want_name: str) -> bool:
    """
    最后兜底：完整文件名以子串形式出现在序列化 metadata 或 _node_content 中则视为命中。
    （用于 file_name 不在顶层、解析路径遗漏但 JSON 中仍含原文的情况。）
    """
    want = _nfc(Path(want_name).name)
    if not want or len(want) < 4:
        return False
    try:
        blob = json.dumps(meta, ensure_ascii=False)
    except Exception:
        blob = str(meta)
    if want in blob:
        return True
    raw = meta.get("_node_content")
    if isinstance(raw, str) and want in raw:
        return True
    return False


def _collection_from_index(index: Any | None) -> Any | None:
    if index is None:
        return None
    vs = getattr(index, "_vector_store", None) or getattr(index, "vector_store", None)
    if vs is None:
        return None
    return getattr(vs, "_collection", None) or getattr(vs, "client", None)


def _collection_supports_metadata_reads(coll: Any) -> bool:
    try:
        coll.get(include=["metadatas"], limit=1, offset=0)
        return True
    except Exception:
        return False


def _safe_resolve_path_label(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return str(path.resolve())
    except Exception:
        return str(path)


def _open_chroma_collection_info(
    cfg: AppConfig,
    index: Any | None = None,
    *,
    prefer_index: bool = True,
    chroma_dir_override: Path | None = None,
) -> dict[str, Any]:
    info: dict[str, Any] = {
        "collection": None,
        "source": "none",
        "requested_prefer_index": bool(prefer_index),
        "fallback_used": False,
        "active_chroma_dir": "",
        "fallback_reason": "",
    }
    active_dir = chroma_dir_override or resolve_active_chroma_dir(cfg, allow_legacy=True)
    info["active_chroma_dir"] = _safe_resolve_path_label(active_dir)
    if prefer_index:
        coll = _collection_from_index(index)
        if coll is not None:
            if _collection_supports_metadata_reads(coll):
                info["collection"] = coll
                info["source"] = "index"
                return info
            info["fallback_used"] = True
            info["fallback_reason"] = "index_collection_metadata_read_failed"
        elif index is not None:
            info["fallback_used"] = True
            info["fallback_reason"] = "index_collection_missing"
    if active_dir is None:
        return info
    info["collection"] = _open_chroma_collection_for_dir(active_dir, cfg.collection_name)
    if info["collection"] is not None:
        info["source"] = "disk"
    return info


def _open_chroma_collection(
    cfg: AppConfig,
    index: Any | None = None,
    *,
    prefer_index: bool = True,
    chroma_dir_override: Path | None = None,
) -> Any | None:
    return _open_chroma_collection_info(
        cfg,
        index=index,
        prefer_index=prefer_index,
        chroma_dir_override=chroma_dir_override,
    ).get("collection")


def chroma_collection_count(
    cfg: AppConfig,
    index: Any | None = None,
    *,
    prefer_index: bool = True,
) -> int:
    """当前 Chroma 集合中的向量条数；-1 表示无法读取。"""
    try:
        info = _open_chroma_collection_info(cfg, index=index, prefer_index=prefer_index)
        coll = info.get("collection")
        if coll is None:
            return -1
        return int(coll.count())
    except Exception:
        return -1


def chroma_sample_distinct_filenames(
    cfg: AppConfig,
    limit: int = 30,
    index: Any | None = None,
    *,
    prefer_index: bool = True,
) -> list[str]:
    """
    扫描当前集合，从元数据中提取可见的文件名（用于预览失败时的排查提示）。
    """
    try:
        info = _open_chroma_collection_info(cfg, index=index, prefer_index=prefer_index)
        coll = info.get("collection")
        if coll is None:
            return []
        all_ids = _chroma_all_ids_paginated(coll)
        triples = _chroma_get_docs_metas_by_ids(coll, all_ids, batch_size=48)
    except Exception:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for _, _, meta in triples:
        m = meta if isinstance(meta, dict) else {}
        for c in _candidate_file_names_from_meta(m):
            c = str(c).strip()
            if c and c not in seen:
                seen.add(c)
                out.append(c)
                if len(out) >= max(1, int(limit)):
                    return sorted(out, key=_norm_name)
    return sorted(out, key=_norm_name)


def _scan_doc_class_label(value: Any) -> str:
    mapping = {
        "text_normal": "正常文本",
        "low_text_normal": "低文本但正常",
        "scan_suspected": "疑似扫描件",
        "scan_recoverable": "扫描可恢复",
        "extract_failed": "抽取失败",
        "ocr_polluted": "OCR 污染",
    }
    key = str(value or "").strip().lower()
    return mapping.get(key, str(value or "").strip())


def _scan_doc_reason_label(value: Any) -> str:
    mapping = {
        "native_text_sufficient": "原生文本充足",
        "few_pages_with_some_native_text": "页数少且存在原生文本",
        "fallback_pages_mostly_suspicious": "fallback 页大多被判为可疑 OCR",
        "native_text_sparse_but_fallback_recovered": "原生文本稀疏，但 fallback 恢复出正文",
        "native_text_sparse_and_page_images_dominate": "原生文本极少，页面图像特征占主导",
        "fallback_attempted_without_usable_text": "已尝试 fallback，但没有得到可用正文",
        "low_text_without_scan_pattern": "文本少，但不符合扫描件特征",
    }
    key = str(value or "").strip().lower()
    return mapping.get(key, str(value or "").strip())


_ZERO_CHUNK_STATUS_LABELS = {
    "indexed": "已入库",
    "zero_chunk": "未入库（0 块）",
}


def _empty_diag_row(file_name: str) -> dict[str, Any]:
    return {
        "file_name": file_name,
        "chunk_count": 0,
        "avg_chars": 0,
        "max_chars": 0,
        "empty_chunks": 0,
        "image_hint_chunks": 0,
        "ocr_hint_chunks": 0,
        "vision_hint_chunks": 0,
        "pdf_doc_class": "",
        "pdf_doc_class_label": "",
        "pdf_doc_class_reason": "",
        "pdf_doc_class_reason_label": "",
        "pdf_text_pages": 0,
        "pdf_page_count": 0,
        "pdf_text_chars": 0,
        "pdf_text_lines": 0,
        "pdf_text_page_ratio_pct": 0,
        "pdf_fallback_text_ratio_pct": 0,
        "pdf_suspicious_page_ratio_pct": 0,
        "index_status": "indexed",
        "index_status_label": _ZERO_CHUNK_STATUS_LABELS["indexed"],
        "_char_sum": 0,
    }


def _apply_scan_meta_to_diag_row(row: dict[str, Any], meta: dict[str, Any] | None) -> None:
    m = meta if isinstance(meta, dict) else {}
    doc_class = str(m.get("pdf_doc_class") or "").strip()
    if doc_class:
        row["pdf_doc_class"] = doc_class
        row["pdf_doc_class_label"] = _scan_doc_class_label(doc_class)
    reason = str(m.get("pdf_doc_class_reason") or "").strip()
    if reason:
        row["pdf_doc_class_reason"] = reason
        row["pdf_doc_class_reason_label"] = _scan_doc_reason_label(reason)
    for key in (
        "pdf_text_pages",
        "pdf_page_count",
        "pdf_text_chars",
        "pdf_text_lines",
        "pdf_text_page_ratio_pct",
        "pdf_fallback_text_ratio_pct",
        "pdf_suspicious_page_ratio_pct",
    ):
        try:
            row[key] = int(m.get(key) or 0)
        except (TypeError, ValueError):
            row[key] = 0


def _manifest_file_scan_meta(cfg: AppConfig) -> dict[str, dict[str, Any]]:
    manifest = _load_index_manifest(cfg)
    if not isinstance(manifest, dict):
        return {}
    files = manifest.get("files") or []
    out: dict[str, dict[str, Any]] = {}
    for item in files:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        out[_filename_sig(name)] = dict(item)
    return out


def _collect_doc_manifest_rows(docs: list[Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for doc in docs:
        meta = dict(getattr(doc, "metadata", None) or {})
        file_name = str(meta.get("file_name") or meta.get("file_path") or "").strip()
        if not file_name:
            continue
        base = str(Path(file_name).name)
        sig = _filename_sig(base)
        row = out.setdefault(sig, {"name": base})
        row["name"] = base
        for key in (
            "pdf_doc_class",
            "pdf_doc_class_reason",
            "pdf_text_pages",
            "pdf_page_count",
            "pdf_text_chars",
            "pdf_text_lines",
            "pdf_text_page_ratio_pct",
            "pdf_fallback_text_ratio_pct",
            "pdf_suspicious_page_ratio_pct",
            "image_enrichment_summary",
        ):
            val = meta.get(key)
            if val is None:
                continue
            if isinstance(val, str) and not val.strip():
                continue
            row[key] = val
    return out


def chunk_diagnostics(
    cfg: AppConfig,
    index: Any | None = None,
    *,
    prefer_index: bool = True,
) -> dict[str, Any]:
    """
    汇总当前 Chroma 集合的切片质量统计，供验证环境做效果排查。
    返回：
      {
        "summary": {...},
        "files": [{...}, ...],
      }
    """
    out: dict[str, Any] = {
        "summary": {
            "total_chunks": 0,
            "total_files": 0,
            "indexed_files": 0,
            "zero_chunk_files": 0,
            "avg_chars": 0,
            "empty_chunks": 0,
            "image_hint_chunks": 0,
            "ocr_hint_chunks": 0,
            "vision_hint_chunks": 0,
            "scan_doc_files": 0,
            "ocr_polluted_files": 0,
            "extract_failed_files": 0,
        },
        "files": [],
        "collection_source": "none",
        "collection_active_dir": "",
        "collection_fallback_used": False,
        "collection_fallback_reason": "",
    }
    try:
        coll_info = _open_chroma_collection_info(cfg, index=index, prefer_index=prefer_index)
        out["collection_source"] = str(coll_info.get("source") or "none")
        out["collection_active_dir"] = str(coll_info.get("active_chroma_dir") or "")
        out["collection_fallback_used"] = bool(coll_info.get("fallback_used"))
        out["collection_fallback_reason"] = str(coll_info.get("fallback_reason") or "")
        coll = coll_info.get("collection")
        if coll is None:
            return out
        all_ids = _chroma_all_ids_paginated(coll)
        triples = _chroma_get_docs_metas_by_ids(coll, all_ids, batch_size=64)
    except Exception:
        return out

    per_file: dict[str, dict[str, Any]] = {}
    total_chars = 0
    for _, doc, meta in triples:
        m = meta if isinstance(meta, dict) else {}
        file_name = "unknown"
        for cand in _candidate_file_names_from_meta(m):
            if cand:
                file_name = str(Path(cand).name)
                break
        text = str(doc or "")
        chars = len(text)
        total_chars += chars
        low = text.casefold()
        has_page_image = ("[第" in text and "图片" in text) or "[docx 内嵌图" in low
        has_ocr = "ocr" in low
        has_vision = "vision" in low or "ocr+vision" in low
        row = per_file.setdefault(file_name, _empty_diag_row(file_name))
        row["chunk_count"] += 1
        row["_char_sum"] += chars
        row["max_chars"] = max(int(row["max_chars"]), chars)
        if not row.get("pdf_doc_class") and str(m.get("pdf_doc_class") or "").strip():
            _apply_scan_meta_to_diag_row(row, m)
        if chars == 0:
            row["empty_chunks"] += 1
        if has_page_image:
            row["image_hint_chunks"] += 1
        if has_ocr:
            row["ocr_hint_chunks"] += 1
        if has_vision:
            row["vision_hint_chunks"] += 1

    manifest_meta = _manifest_file_scan_meta(cfg)
    expected_names = _expected_uploaded_filenames(cfg)
    for name in expected_names:
        sig = _filename_sig(name)
        row = per_file.get(name)
        if row is None:
            row = _empty_diag_row(name)
            row["index_status"] = "zero_chunk"
            row["index_status_label"] = _ZERO_CHUNK_STATUS_LABELS["zero_chunk"]
            per_file[name] = row
        if not row.get("pdf_doc_class"):
            _apply_scan_meta_to_diag_row(row, manifest_meta.get(sig) or {})

    total_chunks = len(triples)
    summary = out["summary"]
    summary["total_chunks"] = total_chunks
    summary["avg_chars"] = round(total_chars / total_chunks, 1) if total_chunks else 0
    for row in per_file.values():
        row["avg_chars"] = round(row.pop("_char_sum") / row["chunk_count"], 1) if row["chunk_count"] else 0
        summary["empty_chunks"] += int(row["empty_chunks"])
        summary["image_hint_chunks"] += int(row["image_hint_chunks"])
        summary["ocr_hint_chunks"] += int(row["ocr_hint_chunks"])
        summary["vision_hint_chunks"] += int(row["vision_hint_chunks"])
        if int(row.get("chunk_count") or 0) > 0:
            summary["indexed_files"] += 1
        else:
            summary["zero_chunk_files"] += 1
        doc_class = str(row.get("pdf_doc_class") or "")
        if doc_class in ("scan_suspected", "scan_recoverable"):
            summary["scan_doc_files"] += 1
        elif doc_class == "ocr_polluted":
            summary["ocr_polluted_files"] += 1
        elif doc_class == "extract_failed":
            summary["extract_failed_files"] += 1
    summary["total_files"] = len(per_file)
    out["files"] = sorted(
        per_file.values(),
        key=lambda x: (-int(x["chunk_count"]), _norm_name(x["file_name"])),
    )
    return out


def _expected_uploaded_filenames(cfg: AppConfig) -> list[str]:
    return [str(x.get("name") or "").strip() for x in uploaded_files_snapshot(cfg) if str(x.get("name") or "").strip()]


def _missing_uploaded_filenames(expected_names: list[str], diag_files: list[dict[str, Any]]) -> list[str]:
    present = {
        _filename_sig(str(x.get("file_name") or ""))
        for x in diag_files
        if str(x.get("file_name") or "").strip() and str(x.get("file_name") or "").strip() != "unknown"
    }
    out: list[str] = []
    for name in expected_names:
        if _filename_sig(name) not in present:
            out.append(name)
    return sorted(out, key=_norm_name)



_DEFAULT_EXCLUDED_DOC_CLASSES = ("ocr_polluted", "extract_failed")


def build_excluded_file_set(
    cfg: AppConfig,
    index: Any | None = None,
    *,
    prefer_index: bool = True,
    excluded_doc_classes: list[str] | tuple[str, ...] | None = None,
    include_zero_chunk: bool = True,
) -> dict[str, Any]:
    classes = [
        str(x or "").strip().lower()
        for x in (excluded_doc_classes or _DEFAULT_EXCLUDED_DOC_CLASSES)
        if str(x or "").strip()
    ]
    diag = chunk_diagnostics(cfg, index=index, prefer_index=prefer_index)
    excluded: list[str] = []
    seen: set[str] = set()
    counts = {
        "by_doc_class": 0,
        "by_zero_chunk": 0,
    }
    for row in diag.get("files") or []:
        if not isinstance(row, dict):
            continue
        name = str(row.get("file_name") or "").strip()
        if not name:
            continue
        doc_class = str(row.get("pdf_doc_class") or "").strip().lower()
        index_status = str(row.get("index_status") or "").strip().lower()
        should_exclude = False
        if doc_class and doc_class in classes:
            should_exclude = True
            counts["by_doc_class"] += 1
        elif include_zero_chunk and index_status == "zero_chunk":
            should_exclude = True
            counts["by_zero_chunk"] += 1
        if not should_exclude:
            continue
        sig = _filename_sig(name)
        if sig in seen:
            continue
        seen.add(sig)
        excluded.append(name)
    excluded.sort(key=_norm_name)
    return {
        "excluded_files": excluded,
        "excluded_doc_classes": classes,
        "include_zero_chunk": bool(include_zero_chunk),
        "excluded_count": len(excluded),
        "excluded_by_doc_class_count": int(counts["by_doc_class"]),
        "excluded_by_zero_chunk_count": int(counts["by_zero_chunk"]),
        "chunk_diag_summary": dict(diag.get("summary") or {}),
    }


def fetch_chunks_for_file(
    cfg: AppConfig,
    filename: str,
    *,
    max_chunks: int = 500,
    index: Any | None = None,
    prefer_index: bool = True,
) -> tuple[list[dict[str, Any]], int, dict[str, Any]]:
    """
    从当前 Chroma 集合读取指定文件名的切片正文（与最近一次成功构建写入的集合一致）。
    返回 (列表项, 匹配到的总块数, 集合来源信息)；列表项为 {"i", "node_id", "text", "meta"}，
    按文档原文顺序排序：优先 start_char_idx、其次 end_char_idx（顶层或 _node_content 内），最后 node_id；
    超过 max_chunks 时截断列表，总数仍为匹配总数。
    """
    fn = _nfc(Path(str(filename).strip()).name)
    coll_info = _open_chroma_collection_info(cfg, index=index, prefer_index=prefer_index)
    source_info = {
        "source": str(coll_info.get("source") or "none"),
        "active_chroma_dir": str(coll_info.get("active_chroma_dir") or ""),
        "fallback_used": bool(coll_info.get("fallback_used")),
        "fallback_reason": str(coll_info.get("fallback_reason") or ""),
        "file_diag": {},
    }
    manifest_meta = _manifest_file_scan_meta(cfg)
    sig = _filename_sig(fn) if fn else ""
    if sig and sig in manifest_meta:
        row = _empty_diag_row(fn)
        row["index_status"] = "zero_chunk"
        row["index_status_label"] = _ZERO_CHUNK_STATUS_LABELS["zero_chunk"]
        _apply_scan_meta_to_diag_row(row, manifest_meta.get(sig) or {})
        source_info["file_diag"] = row
    if not fn:
        return [], 0, source_info

    try:
        coll = coll_info.get("collection")
        if coll is None:
            return [], 0, source_info
    except Exception:
        return [], 0, source_info

    rows: list[tuple[str, str, dict[str, Any]]] = []
    res = None
    for fn_try in _filename_query_variants(fn):
        try:
            res = coll.get(
                where={"file_name": fn_try},
                include=["documents", "metadatas", "ids"],
                limit=max(1, int(max_chunks)),
            )
        except Exception:
            res = None
            continue
        if res and res.get("ids"):
            break
    if res and res.get("ids"):
        _ids = [str(x) for x in (res.get("ids") or [])]
        cand = _chroma_get_docs_metas_by_ids(coll, _ids, batch_size=64)
        filtered = [
            (nid, doc, m)
            for nid, doc, m in cand
            if _metadata_matches_filename(m, fn) or _filename_appears_in_meta_blob(m, fn)
        ]
        rows = filtered if filtered else list(cand)

    if not rows:
        try:
            all_ids = _chroma_all_ids_paginated(coll)
            triples = _chroma_get_docs_metas_by_ids(coll, all_ids, batch_size=48)
        except Exception:
            return [], 0, source_info
        flags = [
            _metadata_matches_filename(m, fn) or _filename_appears_in_meta_blob(m, fn)
            for _, _, m in triples
        ]
        matched_rids: set[str] = set()
        for (_, _, m), ok in zip(triples, flags):
            if ok:
                rid = _ref_doc_id_from_meta(m)
                if rid:
                    matched_rids.add(rid)
        for (nid, doc, m), ok in zip(triples, flags):
            if ok:
                rows.append((nid, doc, m))
            else:
                rid = _ref_doc_id_from_meta(m)
                if rid and rid in matched_rids:
                    rows.append((nid, doc, m))

    def _sort_key(item: tuple[str, str, dict[str, Any]]) -> tuple:
        nid, _, m = item
        s, e = _order_keys_from_meta(m)
        return (s, e, str(nid))

    rows.sort(key=_sort_key)
    total_found = len(rows)
    if total_found > max_chunks:
        rows = rows[:max_chunks]

    out: list[dict[str, Any]] = []
    file_diag_meta: dict[str, Any] = {}
    for i, (nid, text, m) in enumerate(rows, 1):
        if not file_diag_meta and isinstance(m, dict):
            file_diag_meta = {
                "pdf_doc_class": str(m.get("pdf_doc_class") or "").strip(),
                "pdf_doc_class_label": _scan_doc_class_label(m.get("pdf_doc_class")),
                "pdf_doc_class_reason": str(m.get("pdf_doc_class_reason") or "").strip(),
                "pdf_doc_class_reason_label": _scan_doc_reason_label(m.get("pdf_doc_class_reason")),
                "pdf_text_pages": int(m.get("pdf_text_pages") or 0),
                "pdf_page_count": int(m.get("pdf_page_count") or 0),
                "pdf_text_chars": int(m.get("pdf_text_chars") or 0),
                "pdf_text_lines": int(m.get("pdf_text_lines") or 0),
                "pdf_text_page_ratio_pct": int(m.get("pdf_text_page_ratio_pct") or 0),
                "pdf_fallback_text_ratio_pct": int(m.get("pdf_fallback_text_ratio_pct") or 0),
                "pdf_suspicious_page_ratio_pct": int(m.get("pdf_suspicious_page_ratio_pct") or 0),
            }
        out.append({"i": i, "node_id": nid, "text": text, "meta": m})
    source_info["file_diag"] = file_diag_meta
    return out, total_found, source_info


def self_check_index(
    cfg: AppConfig,
    *,
    embed_model_override: str | None = None,
    chroma_dir_override: Path | None = None,
    require_query: bool = True,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "ok": False,
        "count_ok": False,
        "get_ok": False,
        "query_ok": False,
        "count": 0,
        "sample_id": "",
        "sample_query": "",
        "diagnostics_total_chunks": 0,
        "diagnostics_total_files": 0,
        "missing_uploaded_files": [],
        "top_files": [],
        "error": "",
    }
    try:
        coll = _open_chroma_collection(
            cfg,
            index=None,
            prefer_index=False,
            chroma_dir_override=chroma_dir_override,
        )
        if coll is None:
            out["error"] = "无法打开 Chroma 集合"
            return out

        all_ids: list[str] = []
        count_error = ""
        compacted_count = 0
        try:
            compacted_count = _chroma_count_index_only(coll)
            count = compacted_count
        except Exception as e:
            count_error = f"{type(e).__name__}: {e}"
            all_ids = _chroma_all_ids_paginated(coll)
            count = len(all_ids)
        out["count"] = count
        out["count_ok"] = count > 0
        if count <= 0:
            wal_count = 0
            try:
                wal_count = int(coll.count())
            except Exception:
                wal_count = 0
            if wal_count > 0:
                # 可靠模式（高 sync_threshold）下，集合可能长期只在 WAL 可见；
                # 对可用性校验而言，把 WAL 计数视为有效记录数。
                count = wal_count
                out["count"] = count
                out["count_ok"] = True
            else:
                out["error"] = count_error or "集合为空"
                return out

        if not all_ids:
            all_ids = _chroma_all_ids_paginated(coll)
        if not all_ids:
            out["error"] = "集合可计数但无法枚举样本 id"
            return out
        if count != len(all_ids):
            # 不把计数差异作为硬失败；在 WAL/压实切换窗口里两者可能短时不一致。
            out["error"] = f"计数偏差提示：index_only={count}, ids={len(all_ids)}"

        sample_rows = _chroma_get_docs_metas_by_ids(coll, all_ids[:1], batch_size=1)
        if not sample_rows:
            out["error"] = "集合可计数但无法读取样本"
            return out
        sample_id, sample_doc, _ = sample_rows[0]
        sample_doc = str(sample_doc or "").strip()
        out["sample_id"] = str(sample_id)
        out["get_ok"] = True
        query_text = sample_doc[:120].strip() or out["sample_id"]
        out["sample_query"] = query_text

        last_query_error = ""
        query_attempts = 18 if chroma_dir_override is not None else 6
        index = None
        if require_query:
            for attempt in range(query_attempts):
                try:
                    index = load_index_from_disk(
                        cfg,
                        embed_model_override=embed_model_override,
                        chroma_dir_override=chroma_dir_override,
                    )
                    if index is None:
                        last_query_error = "无法从磁盘加载索引"
                    else:
                        retriever = VectorIndexRetriever(
                            index,
                            similarity_top_k=1,
                            embed_model=getattr(index, "_embed_model", None),
                            node_ids=None,
                            callback_manager=getattr(index, "_callback_manager", None),
                            object_map=getattr(index, "_object_map", None),
                        )
                        nodes = retriever.retrieve(query_text)
                        out["query_ok"] = len(nodes) > 0
                        if out["query_ok"]:
                            break
                        last_query_error = "query 无返回结果"
                except Exception as e:
                    last_query_error = f"{type(e).__name__}: {e}"
                if attempt + 1 < query_attempts:
                    _clear_chroma_process_cache()
                    time.sleep(1.0 if chroma_dir_override is not None else 0.35)
            if not out["query_ok"]:
                out["error"] = last_query_error or "query 无返回结果"
                return out

        diag_cfg = cfg
        diag_index = index
        diag_prefer_index = True
        if chroma_dir_override is not None:
            diag_cfg = AppConfig(
                root=cfg.root,
                raw=dict(cfg.raw),
            )
            diag_cfg.raw["data_dir"] = str(chroma_dir_override.parent)
            diag_cfg.raw["chroma_subdir"] = chroma_dir_override.name
            diag_index = None
            diag_prefer_index = False
        try:
            diag = chunk_diagnostics(diag_cfg, index=diag_index, prefer_index=diag_prefer_index)
            summary = diag.get("summary") or {}
            files = list(diag.get("files") or [])
            out["diagnostics_total_chunks"] = int(summary.get("total_chunks") or 0)
            out["diagnostics_total_files"] = int(summary.get("total_files") or 0)
            out["top_files"] = [
                {
                    "file_name": str(x.get("file_name") or ""),
                    "chunk_count": int(x.get("chunk_count") or 0),
                }
                for x in files[:10]
            ]
            expected_names = _expected_uploaded_filenames(cfg)
            out["missing_uploaded_files"] = _missing_uploaded_filenames(expected_names, files)
        except Exception as e:
            # schema 差异（如 KeyError: '_type'）不再阻断索引上线
            out["error"] = f"diagnostics skipped: {type(e).__name__}: {e}"
        out["ok"] = True
        return out
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        return out


def build_index(
    cfg: AppConfig,
    chunk_size: int,
    chunk_overlap: int,
    chunk_mode: str = "sentence",
) -> tuple[bool, str]:
    """同步构建（消费 iter_build_index）；供非 UI 调用。"""
    last = ""
    for line in iter_build_index(cfg, chunk_size, chunk_overlap, chunk_mode=chunk_mode):
        last = line
    if not last:
        return False, "构建未返回任何状态。"
    return last.startswith("完成："), last


def load_index_from_disk(
    cfg: AppConfig,
    embed_model_override: str | None = None,
    chroma_dir_override: Path | None = None,
) -> VectorStoreIndex | None:
    o = dict(cfg.ollama)
    if embed_model_override:
        o["embed_model"] = embed_model_override
    embed = OllamaEmbedding(
        model_name=o["embed_model"],
        base_url=o["base_url"],
        ollama_additional_kwargs={},
    )
    Settings.embed_model = embed

    chroma_dir = chroma_dir_override or resolve_active_chroma_dir(cfg, allow_legacy=True)
    if chroma_dir is None or not chroma_dir.is_dir():
        return None
    try:
        chroma_collection = _open_chroma_collection(
            cfg,
            index=None,
            prefer_index=False,
            chroma_dir_override=chroma_dir,
        )
    except Exception:
        return None
    if chroma_collection is None:
        return None
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context,
        embed_model=embed,
    )
