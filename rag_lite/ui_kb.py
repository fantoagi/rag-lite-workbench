from __future__ import annotations

import html
import shutil
from pathlib import Path
from typing import Any

from rag_lite.config import AppConfig
from rag_lite.ingest import resolve_active_chroma_dir
from rag_lite.store import ExperimentStore


def _dir_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += int(p.stat().st_size)
        except OSError:
            continue
    return total


def _fmt_bytes(n: int) -> str:
    size = float(max(0, int(n)))
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024.0
    return f"{size:.1f} GB"


def _versions_root(cfg: AppConfig) -> Path:
    return cfg.chroma_dir.parent / f"{cfg.chroma_dir.name}.__versions__"


def _building_dirs(cfg: AppConfig) -> list[Path]:
    parent = cfg.chroma_dir.parent
    return sorted(
        [p for p in parent.glob(f"{cfg.chroma_dir.name}.__building__*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime if p.exists() else 0,
        reverse=True,
    )


def _version_dirs(cfg: AppConfig) -> list[Path]:
    root = _versions_root(cfg)
    if not root.is_dir():
        return []
    return sorted(
        [p for p in root.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime if p.exists() else 0,
        reverse=True,
    )


def index_ops_diagnostics(cfg: AppConfig, store: ExperimentStore | None = None, keep_recent: int = 3) -> dict[str, Any]:
    store = store or ExperimentStore(cfg.sqlite_path)
    manifest = store.get_index_manifest() or {}
    active_dir = resolve_active_chroma_dir(cfg, store=store, allow_legacy=True)
    active_resolved = active_dir.resolve() if active_dir is not None else None
    versions = _version_dirs(cfg)
    buildings = _building_dirs(cfg)
    active_subdir = str(manifest.get("active_chroma_subdir") or "")
    deletable_versions: list[Path] = []
    retained_nonactive = 0
    for path in versions:
        try:
            is_active = bool(active_resolved and path.resolve() == active_resolved)
        except OSError:
            is_active = False
        if is_active:
            continue
        if retained_nonactive < max(0, int(keep_recent)):
            retained_nonactive += 1
            continue
        deletable_versions.append(path)
    delete_candidates = deletable_versions + buildings
    return {
        "active_dir": str(active_resolved or ""),
        "manifest_active_chroma_subdir": active_subdir,
        "versions_root": str(_versions_root(cfg).resolve()),
        "version_count": len(versions),
        "building_count": len(buildings),
        "active_size_bytes": _dir_size(active_resolved) if active_resolved else 0,
        "versions_size_bytes": sum(_dir_size(p) for p in versions),
        "building_size_bytes": sum(_dir_size(p) for p in buildings),
        "deletable_version_count": len(deletable_versions),
        "deletable_building_count": len(buildings),
        "deletable_size_bytes": sum(_dir_size(p) for p in delete_candidates),
        "delete_candidates": [str(p.resolve()) for p in delete_candidates],
        "keep_recent": max(0, int(keep_recent)),
    }


def index_ops_summary_html(diag: dict[str, Any] | None) -> str:
    d = diag or {}
    active = html.escape(str(d.get("active_dir") or "(none)"))
    manifest = html.escape(str(d.get("manifest_active_chroma_subdir") or "(none)"))
    versions_root = html.escape(str(d.get("versions_root") or "(none)"))
    return (
        '<div class="rag-tip-block rag-tip-block--tight" style="margin-bottom:0;">'
        f"<p style='margin:0;'><strong>真实活跃目录：</strong><code>{active}</code></p>"
        f"<p style='margin:0.35em 0 0 0;'><strong>Manifest 指向：</strong><code>{manifest}</code></p>"
        f"<p style='margin:0.35em 0 0 0;'><strong>版本根目录：</strong><code>{versions_root}</code></p>"
        f"<p style='margin:0.35em 0 0 0;'><strong>版本数：</strong>{int(d.get('version_count') or 0)}"
        f"　<strong>残留 building：</strong>{int(d.get('building_count') or 0)}"
        f"　<strong>active 大小：</strong>{_fmt_bytes(int(d.get('active_size_bytes') or 0))}"
        f"　<strong>历史版本大小：</strong>{_fmt_bytes(int(d.get('versions_size_bytes') or 0))}"
        f"　<strong>building 大小：</strong>{_fmt_bytes(int(d.get('building_size_bytes') or 0))}</p>"
        f"<p style='margin:0.35em 0 0 0;color:#92400e;'><strong>可清理：</strong>"
        f"{int(d.get('deletable_version_count') or 0)} 个旧版本 + {int(d.get('deletable_building_count') or 0)} 个残留 building"
        f"（约 {_fmt_bytes(int(d.get('deletable_size_bytes') or 0))}，保留最近 {int(d.get('keep_recent') or 0)} 个非 active 版本）</p>"
        "</div>"
    )


def index_ops_rows(diag: dict[str, Any] | None) -> list[list[Any]]:
    candidates = list((diag or {}).get("delete_candidates") or [])
    if not candidates:
        return [["(暂无可清理目录)", "", ""]]
    rows: list[list[Any]] = []
    for p in candidates[:80]:
        path = Path(str(p))
        rows.append([path.name, str(path), _fmt_bytes(_dir_size(path))])
    return rows


def cleanup_old_index_versions(
    cfg: AppConfig,
    store: ExperimentStore | None = None,
    *,
    keep_recent: int = 3,
    dry_run: bool = True,
) -> tuple[str, dict[str, Any]]:
    diag = index_ops_diagnostics(cfg, store=store, keep_recent=keep_recent)
    candidates = [Path(str(p)) for p in diag.get("delete_candidates") or []]
    parent = cfg.chroma_dir.parent.resolve()
    if dry_run:
        return f"dry-run：将清理 {len(candidates)} 个目录，预计释放 {_fmt_bytes(int(diag.get('deletable_size_bytes') or 0))}。", diag
    removed = 0
    errors: list[str] = []
    for path in candidates:
        try:
            resolved = path.resolve()
            resolved.relative_to(parent)
            if resolved == cfg.chroma_dir.resolve():
                raise ValueError("refusing to remove active chroma_dir")
            shutil.rmtree(resolved)
            removed += 1
        except Exception as exc:
            errors.append(f"{path}: {type(exc).__name__}: {exc}")
    next_diag = index_ops_diagnostics(cfg, store=store, keep_recent=keep_recent)
    msg = f"已清理 {removed}/{len(candidates)} 个目录。"
    if errors:
        msg += " 部分失败：" + "；".join(errors[:5])
    return msg, next_diag
