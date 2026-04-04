"""Persist UI-selected models (and similar) under data_dir."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def prefs_path(data_dir: Path) -> Path:
    return data_dir / "ui_preferences.json"


def load_prefs(data_dir: Path) -> dict[str, Any]:
    p = prefs_path(data_dir)
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_prefs(data_dir: Path, data: dict[str, Any]) -> None:
    p = prefs_path(data_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
