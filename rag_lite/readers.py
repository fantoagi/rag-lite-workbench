"""Local file readers for formats that SimpleDirectoryReader would otherwise open as UTF-8 text."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

from rag_lite.config import AppConfig
from rag_lite.image_text import ImageEnrichOptions, image_enrich_options_from_config


def _extract_docx_plain_text(path: Path) -> str:
    from docx import Document as DocxDocument

    docx = DocxDocument(str(path))
    parts: list[str] = []
    for p in docx.paragraphs:
        t = (p.text or "").strip()
        if t:
            parts.append(t)
    for table in docx.tables:
        for row in table.rows:
            cells = [(c.text or "").strip() for c in row.cells]
            cells = [c for c in cells if c]
            if cells:
                parts.append(" | ".join(cells))
    return "\n\n".join(parts)


class PlainDocxReader(BaseReader):
    """Parse .docx via python-docx (paragraphs and tables)."""

    def lazy_load_data(
        self,
        input_file: Path,
        extra_info: dict | None = None,
        **kwargs: Any,
    ) -> Iterable[Document]:
        path = Path(input_file).resolve()
        text = _extract_docx_plain_text(path)
        meta = dict(extra_info or {})
        meta.setdefault("file_name", path.name)
        meta.setdefault("file_path", str(path))
        yield Document(text=text, metadata=meta)


class PlainPdfReader(BaseReader):
    """Extract PDF text via pypdf (matches project requirements)."""

    def lazy_load_data(
        self,
        input_file: Path,
        extra_info: dict | None = None,
        **kwargs: Any,
    ) -> Iterable[Document]:
        from pypdf import PdfReader

        path = Path(input_file).resolve()
        reader = PdfReader(str(path))
        texts: list[str] = []
        for page in reader.pages:
            t = page.extract_text() or ""
            t = t.strip()
            if t:
                texts.append(t)
        text = "\n\n".join(texts)
        meta = dict(extra_info or {})
        meta.setdefault("file_name", path.name)
        meta.setdefault("file_path", str(path))
        yield Document(text=text, metadata=meta)


class HybridPdfReader(BaseReader):
    """PyMuPDF 文本 + 页内图片 OCR / 视觉模型补充。"""

    def __init__(self, opts: ImageEnrichOptions) -> None:
        self._opts = opts

    def lazy_load_data(
        self,
        input_file: Path,
        extra_info: dict | None = None,
        **kwargs: Any,
    ) -> Iterable[Document]:
        import fitz

        from rag_lite.image_text import hybrid_image_to_text

        path = Path(input_file).resolve()
        opts = self._opts
        doc = fitz.open(str(path))
        try:
            all_parts: list[str] = []
            seen_xref: set[int] = set()
            processed = 0
            # 成功写入正文的内嵌图数量；与 raster_xrefs 区分，便于排查「页面上有图但统计为 0」
            raster_xrefs = 0
            empty_after_hybrid = 0
            for page_idx in range(doc.page_count):
                page = doc[page_idx]
                t = (page.get_text() or "").strip()
                if t:
                    all_parts.append(t)
                if processed >= opts.max_images_per_file:
                    continue
                for img_info in page.get_images(full=True) or []:
                    if processed >= opts.max_images_per_file:
                        break
                    xref = int(img_info[0])
                    if xref in seen_xref:
                        continue
                    seen_xref.add(xref)
                    try:
                        base = doc.extract_image(xref)
                    except Exception:
                        continue
                    image_bytes = base.get("image")
                    if not image_bytes:
                        continue
                    raster_xrefs += 1
                    block, tag = hybrid_image_to_text(image_bytes, opts)
                    if not block:
                        empty_after_hybrid += 1
                        continue
                    processed += 1
                    label = tag or "text"
                    all_parts.append(
                        f"\n\n[第{page_idx + 1}页 图片 {processed} · {label}]\n{block}"
                    )
            text = "\n\n".join(all_parts)
            meta = dict(extra_info or {})
            meta.setdefault("file_name", path.name)
            meta.setdefault("file_path", str(path))
            # pdf_images=写入索引的内嵌图条数；raster_xrefs=解析到的 PDF 内嵌位图对象数；
            # empty_hybrid=位图已提取但 OCR+视觉 均无可用文本（模型/阈值/过小图等）
            meta["image_enrichment_summary"] = (
                f"pdf_images={processed},raster_xrefs={raster_xrefs},empty_hybrid={empty_after_hybrid}"
            )
            yield Document(text=text, metadata=meta)
        finally:
            doc.close()


class HybridDocxReader(BaseReader):
    """段落/表格文本 + 压缩包内 word/media 图片（顺序与正文不完全一致）。"""

    def __init__(self, opts: ImageEnrichOptions) -> None:
        self._opts = opts

    def lazy_load_data(
        self,
        input_file: Path,
        extra_info: dict | None = None,
        **kwargs: Any,
    ) -> Iterable[Document]:
        import zipfile

        from rag_lite.image_text import hybrid_image_to_text

        path = Path(input_file).resolve()
        opts = self._opts
        body = _extract_docx_plain_text(path)
        extra_parts: list[str] = []
        n = 0
        try:
            with zipfile.ZipFile(path) as zf:
                names = sorted(
                    n
                    for n in zf.namelist()
                    if n.startswith("word/media/") and not n.endswith("/")
                )
                for name in names:
                    if n >= opts.max_images_per_file:
                        break
                    try:
                        data = zf.read(name)
                    except Exception:
                        continue
                    if not data:
                        continue
                    block, tag = hybrid_image_to_text(data, opts)
                    if not block:
                        continue
                    n += 1
                    label = tag or "text"
                    extra_parts.append(f"\n\n[DOCX 内嵌图 {n} · {label}]\n{block}")
        except (zipfile.BadZipFile, OSError):
            pass
        text = body + "".join(extra_parts)
        meta = dict(extra_info or {})
        meta.setdefault("file_name", path.name)
        meta.setdefault("file_path", str(path))
        meta["image_enrichment_summary"] = f"docx_images={n}"
        yield Document(text=text, metadata=meta)


def default_local_file_extractors(
    cfg: AppConfig | None = None,
    *,
    image_enrichment: bool | None = None,
    pipeline_overrides: dict[str, Any] | None = None,
) -> dict[str, BaseReader]:
    """Use with SimpleDirectoryReader(file_extractor=...)."""
    if cfg is None:
        return {
            ".docx": PlainDocxReader(),
            ".pdf": PlainPdfReader(),
        }
    opts = image_enrich_options_from_config(
        cfg,
        image_enrichment=image_enrichment,
        pipeline_overrides=pipeline_overrides,
    )
    if opts is None:
        return {
            ".docx": PlainDocxReader(),
            ".pdf": PlainPdfReader(),
        }
    return {
        ".docx": HybridDocxReader(opts),
        ".pdf": HybridPdfReader(opts),
    }
