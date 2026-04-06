"""
When image enrichment is enabled: try to install Tesseract (Windows winget).
Does not run ollama pull; missing vision models must be pulled manually.
Called from run.ps1 / main.py; failures are non-fatal.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _model_in_ollama_tags(base_url: str, want: str, timeout: float = 5.0) -> bool:
    url = base_url.rstrip("/") + "/api/tags"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError, ValueError):
        return False
    names = [str(m.get("name", "")) for m in (data.get("models") or []) if m.get("name")]
    wb = want.split(":")[0].strip()
    for n in names:
        if n == want or n.split(":")[0] == wb:
            return True
    return False


def _tesseract_present() -> bool:
    if shutil.which("tesseract"):
        return True
    if sys.platform == "win32":
        for p in (
            Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
            Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
        ):
            if p.is_file():
                return True
    return False


def _winget_install_tesseract() -> bool:
    winget = shutil.which("winget")
    if not winget or sys.platform != "win32":
        return False
    # UB-Mannheim build includes language packs; --silent may still show UAC once
    r = subprocess.run(
        [
            winget,
            "install",
            "-e",
            "--id",
            "UB-Mannheim.TesseractOCR",
            "--accept-package-agreements",
            "--accept-source-agreements",
            "--silent",
        ],
        cwd=str(ROOT),
    )
    return r.returncode == 0


def main() -> int:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)

    from rag_lite.config import load_config
    from rag_lite import prefs as prefs_mod

    cfg = load_config(ROOT)
    ing = cfg.ingest
    pr = prefs_mod.load_prefs(cfg.data_dir)
    if "image_enrichment" in pr:
        enabled = bool(pr["image_enrichment"])
    else:
        enabled = bool(ing.get("image_enrichment"))
    if not enabled:
        return 0

    oe = pr.get("image_ocr_engine")
    if oe is None or (isinstance(oe, str) and not str(oe).strip()):
        oe = ing.get("ocr_engine") or "tesseract"
    oe = str(oe).strip().lower()
    if oe not in ("tesseract", "paddleocr"):
        oe = "tesseract"

    if oe == "paddleocr":
        print("[RAG-Lite] ocr_engine=paddleocr: skipping Tesseract / winget; checking paddleocr ...", flush=True)
        try:
            import paddleocr  # noqa: F401

            print("[RAG-Lite] paddleocr import OK.", flush=True)
        except ImportError:
            print(
                "[RAG-Lite] paddleocr not installed. See requirements-paddleocr.txt "
                "(install paddlepaddle for your platform, then paddleocr).",
                flush=True,
            )
    else:
        print("[RAG-Lite] image_enrichment=true: checking Tesseract and vision model ...", flush=True)

        if not _tesseract_present():
            print("[RAG-Lite] Tesseract not found; trying winget ...", flush=True)
            if _winget_install_tesseract():
                print("[RAG-Lite] winget reported success; restart the app if OCR still fails (PATH).", flush=True)
            else:
                print(
                    "[RAG-Lite] Could not auto-install Tesseract. Install from "
                    "https://github.com/UB-Mannheim/tesseract/wiki or set ingest.tesseract_cmd.",
                    flush=True,
                )
        else:
            print("[RAG-Lite] Tesseract is available.", flush=True)

    o = cfg.ollama
    base = str(o.get("base_url", "http://127.0.0.1:11434")).rstrip("/")
    vm = str(pr.get("image_vision_model") or ing.get("vision_model") or "").strip() or "llava"

    if not shutil.which("ollama"):
        print("[RAG-Lite] ollama not in PATH; install Ollama and pull the vision model manually.", flush=True)
        return 0

    if _model_in_ollama_tags(base, vm):
        print(f"[RAG-Lite] Ollama already has vision model: {vm}", flush=True)
        return 0

    print(
        f"[RAG-Lite] Vision model not found locally: {vm}. "
        f"Pull it manually when ready: ollama pull {vm}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
