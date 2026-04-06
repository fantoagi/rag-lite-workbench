from __future__ import annotations

import concurrent.futures
import json
import os
import re
import time
import shutil
import unicodedata
from collections.abc import Iterator
from itertools import zip_longest
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
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
        n for n in nodes if n.get_content(metadata_mode=MetadataMode.EMBED) != ""
    ]
    skipped = len(nodes) - len(content_nodes)
    if skipped:
        yield f"    └ 提示：跳过 {skipped} 个无嵌入内容的空块。"
    n_total = len(content_nodes)
    if n_total == 0:
        yield "失败：没有可写入向量的文本块。"
        return

    import chromadb
    from chromadb.config import Settings as ChromaSettings

    yield "[4/5] 连接 Chroma、清空旧集合并准备写入 …"
    chroma_client = chromadb.PersistentClient(
        path=str(cfg.chroma_dir),
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    try:
        chroma_client.delete_collection(cfg.collection_name)
    except Exception:
        pass
    chroma_collection = chroma_client.get_or_create_collection(cfg.collection_name)
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
        yield f"失败：向量化或写入 Chroma 时出错：{e}"
        return

    snap = uploaded_files_snapshot(cfg)
    built_iso = _utc_now_iso()
    for item in snap:
        item["indexed_at"] = built_iso
    if store is not None:
        try:
            store.save_index_manifest(
                embed_model=em_name,
                chunk_mode=(chunk_mode or "sentence").strip().lower(),
                chunk_size=int(chunk_size),
                chunk_overlap=int(chunk_overlap),
                files=snap,
            )
        except Exception:
            pass

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
    """仅分页收集全部 id（顺序与集合默认迭代一致），避免 offset+并行字段错位。"""
    n_total = int(coll.count())
    if n_total <= 0:
        return []
    out: list[str] = []
    offset = 0
    while offset < n_total:
        batch = coll.get(include=["metadatas"], limit=page, offset=offset)
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


def chroma_collection_count(cfg: AppConfig) -> int:
    """当前 Chroma 集合中的向量条数；-1 表示无法读取。"""
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    if not cfg.chroma_dir.is_dir():
        return -1
    try:
        client = chromadb.PersistentClient(
            path=str(cfg.chroma_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        coll = client.get_collection(cfg.collection_name)
        return int(coll.count())
    except Exception:
        return -1


def chroma_sample_distinct_filenames(cfg: AppConfig, limit: int = 30) -> list[str]:
    """
    扫描当前集合，从元数据中提取可见的文件名（用于预览失败时的排查提示）。
    """
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    if not cfg.chroma_dir.is_dir():
        return []
    try:
        client = chromadb.PersistentClient(
            path=str(cfg.chroma_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        coll = client.get_collection(cfg.collection_name)
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


def fetch_chunks_for_file(
    cfg: AppConfig,
    filename: str,
    *,
    max_chunks: int = 500,
) -> tuple[list[dict[str, Any]], int]:
    """
    从当前 Chroma 集合读取指定文件名的切片正文（与最近一次成功构建写入的集合一致）。
    返回 (列表项, 匹配到的总块数)；列表项为 {\"i\", \"node_id\", \"text\", \"meta\"}，
    按文档原文顺序排序：优先 start_char_idx、其次 end_char_idx（顶层或 _node_content 内），最后 node_id；
    超过 max_chunks 时截断列表，总数仍为匹配总数。
    """
    fn = _nfc(Path(str(filename).strip()).name)
    if not fn:
        return [], 0

    import chromadb
    from chromadb.config import Settings as ChromaSettings

    if not cfg.chroma_dir.is_dir():
        return [], 0
    client = chromadb.PersistentClient(
        path=str(cfg.chroma_dir),
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    try:
        coll = client.get_collection(cfg.collection_name)
    except Exception:
        return [], 0

    n_total = int(coll.count())
    rows: list[tuple[str, str, dict[str, Any]]] = []
    res = None
    for fn_try in _filename_query_variants(fn):
        try:
            res = coll.get(
                where={"file_name": fn_try},
                include=["documents", "metadatas", "ids"],
                limit=max(n_total, 1),
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
        # where 已命中时优先用结构化+子串匹配；若元数据异常导致全被滤掉，仍展示 where 拉回的块（信任 Chroma）
        rows = filtered if filtered else list(cand)

    if not rows:
        try:
            all_ids = _chroma_all_ids_paginated(coll)
            triples = _chroma_get_docs_metas_by_ids(coll, all_ids, batch_size=48)
        except Exception:
            return [], 0
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
    for i, (nid, text, m) in enumerate(rows, 1):
        out.append({"i": i, "node_id": nid, "text": text, "meta": m})
    return out, total_found


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


def load_index_from_disk(cfg: AppConfig, embed_model_override: str | None = None) -> VectorStoreIndex | None:
    o = dict(cfg.ollama)
    if embed_model_override:
        o["embed_model"] = embed_model_override
    embed = OllamaEmbedding(
        model_name=o["embed_model"],
        base_url=o["base_url"],
        ollama_additional_kwargs={},
    )
    Settings.embed_model = embed

    import chromadb
    from chromadb.config import Settings as ChromaSettings

    if not cfg.chroma_dir.is_dir():
        return None
    chroma_client = chromadb.PersistentClient(
        path=str(cfg.chroma_dir),
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    try:
        chroma_collection = chroma_client.get_collection(cfg.collection_name)
    except Exception:
        return None
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context,
        embed_model=embed,
    )
