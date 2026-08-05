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
    def chroma_dir_redirect_info(self) -> dict[str, Any] | None:
        """If the chroma_dir was redirected to %LOCALAPPDATA% (because the
        project path contains non-ASCII characters), return
        ``{"original": str, "final": str, "reason": "non-ascii-path"}``;
        otherwise return ``None``. The UI uses this to surface a one-time
        warning so the user is never surprised by a hidden index location.
        """
        raw_chroma_dir = self.raw.get("chroma_dir")
        if raw_chroma_dir:
            raw = Path(str(raw_chroma_dir))
            original = raw if raw.is_absolute() else (self.root / raw).resolve()
        else:
            original = self.data_dir / self.raw["chroma_subdir"]
        if os.name == "nt" and not _is_ascii_only_path(original):
            local_appdata = Path(os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local"))
            safe_root = local_appdata / "ClaudeCode" / "rag_lite_chroma" / _stable_ascii_project_slug(self.root)
            final = safe_root / str(self.raw["chroma_subdir"])
            try:
                original_resolved = str(original.resolve())
                final_resolved = str(final.resolve())
            except Exception:
                original_resolved, final_resolved = str(original), str(final)
            if original_resolved == final_resolved:
                return None
            return {"original": original_resolved, "final": final_resolved, "reason": "non-ascii-path"}
        return None

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
    def prompts_dir(self) -> Path:
        raw = self.raw.get("prompt") or {}
        sub = raw.get("dir") or "./prompts"
        p = Path(str(sub))
        return p if p.is_absolute() else (self.root / p).resolve()

    @property
    def ui(self) -> dict[str, Any]:
        return dict(self.raw.get("ui", {}))

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)


def _resolve_prompt_block(raw: dict[str, Any], root: Path) -> dict[str, Any]:
    """Load system prompt text from an external file when configured.

    Precedence:
    1. ``prompt.system_file`` (absolute or relative to project root)
    2. ``prompt.dir`` + ``system_{version}.txt`` when ``version`` is set
    3. Inline ``prompt.system_default`` (legacy fallback)
    """
    prompt = dict(raw.get("prompt") or {})
    version = str(prompt.get("version") or "").strip() or "v1"
    prompt["version"] = version

    prompts_dir_raw = prompt.get("dir") or "./prompts"
    prompts_dir = Path(str(prompts_dir_raw))
    if not prompts_dir.is_absolute():
        prompts_dir = (root / prompts_dir).resolve()

    system_path: Path | None = None
    system_file = prompt.get("system_file")
    if system_file:
        p = Path(str(system_file))
        system_path = p if p.is_absolute() else (root / p).resolve()
    else:
        candidate = prompts_dir / f"system_{version}.txt"
        if candidate.is_file():
            system_path = candidate

    if system_path is not None:
        if not system_path.is_file():
            raise FileNotFoundError(f"prompt system_file not found: {system_path}")
        text = system_path.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"prompt system_file is empty: {system_path}")
        prompt["system_default"] = text
        prompt["system_file"] = str(system_path.relative_to(root)) if system_path.is_relative_to(root) else str(system_path)
        prompt["system_file_resolved"] = str(system_path)
    elif not str(prompt.get("system_default") or "").strip():
        raise ValueError(
            "prompt.system_default is empty and no prompt file was found; "
            f"set prompt.system_file or place {prompts_dir / f'system_{version}.txt'}"
        )

    prompt["dir"] = str(prompts_dir.relative_to(root)) if prompts_dir.is_relative_to(root) else str(prompts_dir)
    return prompt


def load_config(root: Path | None = None) -> AppConfig:
    if root is None:
        root = Path(__file__).resolve().parent.parent
    path = root / "config.yaml"
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    raw["prompt"] = _resolve_prompt_block(raw, root)
    cfg = AppConfig(root=root, raw=raw)
    cfg.ensure_dirs()
    return cfg
