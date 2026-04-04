from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


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
        return self.data_dir / self.raw["chroma_subdir"]

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
