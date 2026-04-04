from __future__ import annotations

import os
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
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
    if not any(cfg.uploads_dir.iterdir()):
        yield "失败：上传目录为空，请先上传文档。"
        return

    yield "[2/5] 加载文档（本地解析 PDF/TXT/Markdown/DOCX）…"
    reader = SimpleDirectoryReader(
        input_dir=str(cfg.uploads_dir),
        recursive=True,
        required_exts=[".pdf", ".txt", ".md", ".markdown", ".docx"],
        file_extractor=default_local_file_extractors(),
    )
    docs = reader.load_data()
    if not docs:
        yield "失败：未能解析出任何文档内容。"
        return

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
                yield f"    └ 已向量化并写入 {done}/{n_total} 块 …"
    except Exception as e:
        yield f"失败：向量化或写入 Chroma 时出错：{e}"
        return

    snap = uploaded_files_snapshot(cfg)
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
