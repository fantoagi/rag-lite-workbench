# -*- coding: utf-8 -*-
"""
RAG-Lite self-test: no Web UI. Checks Python, Gradio, config/SQLite, Ollama, optional heavy imports.

Usage (from ragZone/):
  python self_test.py
  python self_test.py --import-all
"""
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _ok(msg: str) -> None:
    print(f"  [OK] {msg}", flush=True)


def _warn(msg: str) -> None:
    print(f"  [!!] {msg}", flush=True)


def _fail(msg: str) -> None:
    print(f"  [XX] {msg}", flush=True)


def check_python() -> None:
    print("\n=== Python ===", flush=True)
    print(f"  executable: {sys.executable}", flush=True)
    print(f"  version: {sys.version.split()[0]}", flush=True)


def check_gradio_meta() -> tuple[str | None, int | None]:
    print("\n=== Gradio (metadata only, no package import) ===", flush=True)
    spec = importlib.util.find_spec("gradio")
    if spec is None:
        _fail("gradio not installed")
        return None, None
    try:
        v = importlib.metadata.version("gradio")
    except Exception as e:
        _fail(f"cannot read version: {e}")
        return None, None
    try:
        major = int(v.split(".", 1)[0])
    except Exception:
        major = None
    _ok(f"gradio {v} (major={major})")
    if major is not None and major >= 6:
        _warn(
            "Gradio 6+ is slow to import; suggest: "
            'pip install "gradio>=4.44.0,<5.0.0" --force-reinstall'
        )
    return v, major


def check_gradio_import() -> bool:
    print("\n=== import gradio (timed) ===", flush=True)
    t0 = time.perf_counter()
    try:
        import gradio as gr  # noqa: F401

        dt = time.perf_counter() - t0
        _ok(f"import ok in {dt:.1f}s")
        if dt > 30:
            _warn("very slow; likely Gradio 6.x or cold cache ?? pin Gradio 4.x per README")
        return True
    except Exception as e:
        _fail(f"import failed: {e}")
        return False


def check_config_store() -> bool:
    print("\n=== config + SQLite ===", flush=True)
    try:
        from rag_lite.config import load_config
        from rag_lite.store import ExperimentStore

        t0 = time.perf_counter()
        cfg = load_config(ROOT)
        _ = ExperimentStore(cfg.sqlite_path)
        _ok(f"load_config + ExperimentStore ok ({time.perf_counter() - t0:.2f}s)")
        _ok(f"data_dir={cfg.data_dir}")
        _ok(f"sqlite={cfg.sqlite_path}")
        _ok(f"chroma_dir={cfg.chroma_dir}")
        return True
    except Exception as e:
        _fail(str(e))
        return False


def check_ollama(base_url: str, timeout: float = 5.0) -> None:
    print("\n=== Ollama HTTP ===", flush=True)
    url = base_url.rstrip("/") + "/api/tags"
    try:
        t0 = time.perf_counter()
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        data = json.loads(raw)
        models = data.get("models") or []
        names = [m.get("name", "?") for m in models[:12]]
        _ok(f"{url} in {time.perf_counter() - t0:.2f}s, models={len(models)}")
        if names:
            tail = "..." if len(models) > 12 else ""
            print("       sample:", ", ".join(names) + tail, flush=True)
    except urllib.error.URLError as e:
        _fail(f"cannot reach {url}: {e}")
        _warn("start Ollama and check ollama.base_url in config.yaml")
    except Exception as e:
        _fail(str(e))


def check_heavy_imports() -> None:
    print("\n=== heavy imports (ingest / engine) ===", flush=True)
    for label, mod in [("rag_lite.ingest", "rag_lite.ingest"), ("rag_lite.engine", "rag_lite.engine")]:
        t0 = time.perf_counter()
        try:
            importlib.import_module(mod)
            _ok(f"{label} {time.perf_counter() - t0:.1f}s")
        except Exception as e:
            _fail(f"{label}: {e}")


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG-Lite self test")
    parser.add_argument("--import-all", action="store_true", help="also import ingest and engine")
    parser.add_argument("--skip-ollama", action="store_true", help="skip Ollama probe")
    parser.add_argument(
        "--no-gradio-import",
        action="store_true",
        help="skip timed `import gradio` (use when Gradio 6 hangs minutes)",
    )
    args = parser.parse_args()

    check_python()
    check_gradio_meta()
    print("\n(hint) To skip auto pip in main.py, set env RAG_LITE_SKIP_AUTO_PIP=1", flush=True)

    if args.no_gradio_import:
        print("\n=== import gradio (skipped) ===", flush=True)
        _warn("--no-gradio-import: not testing import speed")
        g_import_ok = True
    else:
        g_import_ok = check_gradio_import()
    cfg_ok = check_config_store()

    if not args.skip_ollama:
        try:
            import yaml

            with (ROOT / "config.yaml").open(encoding="utf-8") as f:
                raw = yaml.safe_load(f)
            base = (raw.get("ollama") or {}).get("base_url", "http://127.0.0.1:11434")
            check_ollama(str(base))
        except Exception as e:
            _fail(f"ollama probe via config failed: {e}")

    if args.import_all:
        check_heavy_imports()

    print("\n=== summary ===", flush=True)
    if g_import_ok and cfg_ok:
        print("  Core checks passed. If main.py hangs, note the last log line:", flush=True)
        print("  - stuck at pip: network / torch download; or pre-install gradio 4.x only", flush=True)
        print("  - stuck at import gradio: still on Gradio 6.x", flush=True)
        print("  - stuck at launch: port in use / firewall", flush=True)
        return 0
    print("  Fix failures above before running main.py.", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
