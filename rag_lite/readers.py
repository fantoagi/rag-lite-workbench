"""Local file readers for formats that SimpleDirectoryReader would otherwise open as UTF-8 text."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class PlainDocxReader(BaseReader):
    """Parse .docx via python-docx (paragraphs and tables)."""

    def lazy_load_data(
        self,
        input_file: Path,
        extra_info: dict | None = None,
        **kwargs: Any,
    ) -> Iterable[Document]:
        from docx import Document as DocxDocument

        path = Path(input_file).resolve()
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
        text = "\n\n".join(parts)
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


def default_local_file_extractors() -> dict[str, BaseReader]:
    """Use with SimpleDirectoryReader(file_extractor=...)."""
    return {
        ".docx": PlainDocxReader(),
        ".pdf": PlainPdfReader(),
    }
