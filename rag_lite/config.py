from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
from typing import Any

import yaml


def _is_ascii_only_path(p: Path) -> bool:
    try:
        str(p).encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def _stable_ascii_project_slug(root: Path) -> str:
    base = re.sub(r"[^A-Za-z0-9_.-]+", "-", root.name).strip("-._") or "project"
    digest = hashlib.sha1(str(root).encode("utf-8")).hexdigest()[:10]
    return f"{base}-{digest}"


@dataclass
class AppConfig:
    root: Path
    raw: dict[str, Any]

    @property
    def data_dir(self) -> Path:
        p = Path(self.raw["data_dir"])
        return p if p.is_absolute() else (self.root / p).resolve()

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / self.raw["uploads_subdir"]

    @property
    def chroma_dir(self) -> Path:
        raw_chroma_dir = self.raw.get("chroma_dir")
        if raw_chroma_dir:
            p = Path(str(raw_chroma_dir))
            candidate = p if p.is_absolute() else (self.root / p).resolve()
        else:
            candidate = self.data_dir / self.raw["chroma_subdir"]
        if os.name == "nt" and not _is_ascii_only_path(candidate):
            local_appdata = Path(os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local"))
            safe_root = local_appdata / "ClaudeCode" / "rag_lite_chroma" / _stable_ascii_project_slug(self.root)
            candidate = safe_root / str(self.raw["chroma_subdir"])
        return candidate.resolve()

    @property
    def sqlite_path(self) -> Path:
        return self.data_dir / self.raw["sqlite_filename"]

    @property
    def collection_name(self) -> str:
        return str(self.raw["collection_name"])

    @property
    def ollama(self) -> dict[str, Any]:
        return dict(self.raw.get("ollama", {}))

    @property
    def ingest(self) -> dict[str, Any]:
        return dict(self.raw.get("ingest", {}))

    @property
    def rerank(self) -> dict[str, Any]:
        return dict(self.raw.get("rerank", {}))

    @property
    def retrieval(self) -> dict[str, Any]:
        return dict(self.raw.get("retrieval", {}))

    @property
    def chunking(self) -> dict[str, Any]:
        return dict(self.raw.get("chunking", {}))

    @property
    def prompt(self) -> dict[str, Any]:
        return dict(self.raw.get("prompt", {}))

    @property
    def ui(self) -> dict[str, Any]:
        return dict(self.raw.get("ui", {}))

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)


def load_config(root: Path | None = None) -> AppConfig:
    if root is None:
        root = Path(__file__).resolve().parent.parent
    path = root / "config.yaml"
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = AppConfig(root=root, raw=raw)
    cfg.ensure_dirs()
    return cfg
