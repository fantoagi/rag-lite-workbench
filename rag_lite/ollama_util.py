"""Ollama HTTP helpers: list local models for UI dropdowns."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


def list_ollama_models(base_url: str, timeout: float = 5.0) -> list[str]:
    """
    GET /api/tags -> model names. Returns [] on failure.
    """
    url = base_url.rstrip("/") + "/api/tags"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, OSError):
        return []
    try:
        data: dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError:
        return []
    models = data.get("models") or []
    names: list[str] = []
    for m in models:
        if isinstance(m, dict) and m.get("name"):
            names.append(str(m["name"]))
    return sorted(set(names))


def merge_model_choices(ollama_names: list[str], config_candidates: list[str], current: str) -> list[str]:
    """Union ollama + yaml list, preserve order, ensure current is included."""
    seen: set[str] = set()
    out: list[str] = []
    for x in list(config_candidates) + list(ollama_names):
        s = str(x).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    if current and current not in seen:
        out.insert(0, current)
    return out
