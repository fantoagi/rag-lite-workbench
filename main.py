from __future__ import annotations

import csv
import difflib
import html
import importlib.metadata
import importlib.util
import json
import logging
import operator
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_ANONYMIZED_TELEMETRY", "False")


class _ChromaTelemetryNoiseFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "Failed to send telemetry event" not in record.getMessage()


_CHROMA_TELEMETRY_FILTER = _ChromaTelemetryNoiseFilter()


def _quiet_chroma_telemetry_logs() -> None:
    for handler in logging.getLogger().handlers:
        handler.addFilter(_CHROMA_TELEMETRY_FILTER)
    for name in ("chromadb.telemetry", "chromadb.telemetry.product", "chromadb.telemetry.product.posthog", "posthog"):
        logger = logging.getLogger(name)
        logger.addFilter(_CHROMA_TELEMETRY_FILTER)
        logger.setLevel(logging.CRITICAL)


_quiet_chroma_telemetry_logs()

if sys.version_info >= (3, 14):
    print(
        "[RAG-Lite] 当前 Python 为 {}.{}，版本过新：请在 Windows 上改用 Python 3.12（推荐）或 3.11，"
        "删除 ragZone/.venv 后重装依赖。详见 README。".format(sys.version_info.major, sys.version_info.minor),
        file=sys.stderr,
        flush=True,
    )
    raise SystemExit(1)

# 无缓冲输出，避免 IDE「运行」时长时间看不到任何反应
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

print("[RAG-Lite] 脚本已开始执行…", flush=True)

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 与 PyPI gradio 5.50.0 元数据一致；勿单独升级 gradio-client。
# Gradio 4.44 依赖 HfFolder，与 transformers 5 所需的 huggingface-hub 1.x 互斥；统一用 5.50。
_GRADIO_VERSION_PIN = "gradio==5.50.0"
_GRADIO_CLIENT_PIN = "gradio-client==1.14.0"
_TOMLKIT_PIN = "tomlkit>=0.12.0,<0.14.0"
_PYDANTIC_GRADIO_PIN = "pydantic>=2.0,<=2.12.3"
_CHROMADB_PIN = "chromadb==0.5.23"


def _pip_uninstall_gradio_stack() -> None:
    """拆掉混装后再装，避免新版 gradio-client 与 gradio 5.50.0 不匹配。"""
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", "gradio", "gradio-client"],
        stdin=subprocess.DEVNULL,
    )


def _pip_install(*pip_args: str) -> None:
    """子进程继承终端 stdout/stderr，便于看到 pip 进度（长时间无 newline 时至少知道未卡死）。"""
    cmd = [sys.executable, "-m", "pip", "install", "--default-timeout=120", *pip_args]
    print(f"[RAG-Lite] 执行: {' '.join(cmd)}\n", flush=True)
    r = subprocess.run(cmd, stdin=subprocess.DEVNULL)
    if r.returncode != 0:
        raise SystemExit(f"pip 失败（退出码 {r.returncode}），请检查网络/权限后重试或手动安装依赖。")


def _gradio_version_str() -> str | None:
    try:
        if importlib.util.find_spec("gradio") is None:
            return None
        return importlib.metadata.version("gradio")
    except Exception:
        return None


def _ensure_dependencies() -> None:
    """缺依赖或 Gradio 版本不在 5.x（含误装 6+、残留 4.x）时安装/对齐。"""
    skip = os.environ.get("RAG_LITE_SKIP_AUTO_PIP", "").strip().lower() in ("1", "true", "yes")
    if skip:
        gv = _gradio_version_str()
        if gv:
            try:
                gmaj = int(gv.split(".", 1)[0])
            except Exception:
                gmaj = 5
            if gmaj >= 6:
                print(
                    "[RAG-Lite] 已设置 RAG_LITE_SKIP_AUTO_PIP，跳过自动安装；"
                    f"当前 Gradio {gv} 导入较慢，建议手动: pip install {_GRADIO_CLIENT_PIN} {_GRADIO_VERSION_PIN} --force-reinstall",
                    flush=True,
                )
            elif gmaj < 5:
                print(
                    "[RAG-Lite] 已设置 RAG_LITE_SKIP_AUTO_PIP；"
                    f"当前 Gradio {gv} 与 huggingface-hub 1.x / transformers 5 不兼容，建议升级到 5.x：\n"
                    f"  pip install {_PYDANTIC_GRADIO_PIN} {_GRADIO_CLIENT_PIN} {_GRADIO_VERSION_PIN} {_TOMLKIT_PIN} --force-reinstall",
                    flush=True,
                )
        return

    gver = _gradio_version_str()
    if gver is None:
        req = ROOT / "requirements.txt"
        if not req.is_file():
            raise SystemExit(f"未找到依赖清单: {req}")
        print(f"[RAG-Lite] 未检测到 Gradio。当前 Python:\n  {sys.executable}\n[RAG-Lite] 正在完整安装 requirements（首次可能较久）…", flush=True)
        _pip_install("-r", str(req))
        return

    try:
        major = int(gver.split(".", 1)[0])
    except Exception:
        major = 5

    if 5 <= major < 6:
        return

    if major >= 6:
        print(
            f"[RAG-Lite] 当前 Gradio {gver}（6+）依赖树较大、import 较慢。\n"
            "[RAG-Lite] 尝试仅对齐到 Gradio 5.50（与 transformers / huggingface-hub 1.x 一致）…",
            flush=True,
        )
    else:
        print(
            f"[RAG-Lite] 当前 Gradio {gver}（4.x）与 huggingface-hub 1.x 不兼容（HfFolder）。\n"
            "[RAG-Lite] 正在升级到 Gradio 5.50 …",
            flush=True,
        )

    req = ROOT / "requirements.txt"
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--default-timeout=120", _PYDANTIC_GRADIO_PIN, "--upgrade"],
        stdin=subprocess.DEVNULL,
    )
    _pip_uninstall_gradio_stack()
    _gcmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--default-timeout=120",
        _GRADIO_CLIENT_PIN,
        _GRADIO_VERSION_PIN,
        _TOMLKIT_PIN,
        "--force-reinstall",
    ]
    print(f"[RAG-Lite] 执行: {' '.join(_gcmd)}\n", flush=True)
    gradio_only_ok = subprocess.run(_gcmd, stdin=subprocess.DEVNULL).returncode == 0
    if not gradio_only_ok:
        if not req.is_file():
            raise SystemExit(f"未找到依赖清单: {req}")
        print("[RAG-Lite] 仅对齐 Gradio 栈未成功，改为完整 requirements 安装 …", flush=True)
        _pip_install("-r", str(req))

    gver2 = _gradio_version_str()
    if gver2 is None:
        raise SystemExit("安装后仍无法读取 Gradio 版本，请检查虚拟环境是否一致。")
    try:
        maj2 = int(gver2.split(".", 1)[0])
    except Exception:
        maj2 = 5
    if maj2 >= 6:
        raise SystemExit(
            f"Gradio 仍为 {gver2}。请在本解释器下手动执行:\n"
            f'  {sys.executable} -m pip install "{_GRADIO_CLIENT_PIN}" "{_GRADIO_VERSION_PIN}" --force-reinstall'
        )
    if maj2 < 5:
        raise SystemExit(
            f"Gradio 仍为 {gver2}。请在本解释器下手动执行:\n"
            f"  {sys.executable} -m pip install {_PYDANTIC_GRADIO_PIN} \"{_GRADIO_CLIENT_PIN}\" \"{_GRADIO_VERSION_PIN}\" --force-reinstall"
        )
    print(f"[RAG-Lite] Gradio 已对齐为 {gver2}。", flush=True)


_ensure_dependencies()


def _ensure_optional_ingest_deps() -> None:
    """pymupdf / pytesseract 等在 requirements 中；Gradio 已就绪后补装，避免仅跑 pip 时漏装。"""
    skip = os.environ.get("RAG_LITE_SKIP_AUTO_PIP", "").strip().lower() in ("1", "true", "yes")
    if skip:
        return
    try:
        import fitz  # noqa: F401
        import pytesseract  # noqa: F401
    except ImportError:
        req = ROOT / "requirements.txt"
        if not req.is_file():
            return
        print("[RAG-Lite] 补装 PDF/图片相关依赖（pymupdf、pytesseract 等）…", flush=True)
        _pip_install("-r", str(req))


_ensure_optional_ingest_deps()


def _chromadb_version_str() -> str | None:
    try:
        if importlib.util.find_spec("chromadb") is None:
            return None
        return importlib.metadata.version("chromadb")
    except Exception:
        return None


def _ensure_chromadb_pin() -> None:
    """
    Keep Chroma on the project's validated line.
    Newer 1.x builds can intermittently fail on HNSW reopen in this workstation setup.
    """
    skip = os.environ.get("RAG_LITE_SKIP_AUTO_PIP", "").strip().lower() in ("1", "true", "yes")
    cv = _chromadb_version_str()
    if cv == "0.5.23":
        return
    if skip:
        if cv:
            print(
                f"[RAG-Lite] 检测到 chromadb {cv}，建议切换到 {_CHROMADB_PIN} 以避免 HNSW 重开异常。",
                flush=True,
            )
        return
    req = ROOT / "requirements.txt"
    print(
        f"[RAG-Lite] 检测到 chromadb 版本为 {cv or '未安装'}，正在对齐到 {_CHROMADB_PIN} …",
        flush=True,
    )
    _pip_install(_CHROMADB_PIN, "chroma-hnswlib>=0.7.6")
    if req.is_file():
        # 回装项目顶层依赖，修复 Chroma 降级过程中可能被带偏的传递依赖版本。
        _pip_install("-r", str(req))
    cv2 = _chromadb_version_str()
    if cv2 != "0.5.23":
        raise SystemExit(
            f"chromadb 版本仍为 {cv2 or '未知'}，请手动执行：\n"
            f"  {sys.executable} -m pip install {_CHROMADB_PIN} chroma-hnswlib>=0.7.6 -r requirements.txt"
        )


_ensure_chromadb_pin()


def _ensure_chroma_hnswlib() -> None:
    """Chroma 1.x 在本地持久化索引重开时仍可能触发 hnswlib 依赖。"""
    skip = os.environ.get("RAG_LITE_SKIP_AUTO_PIP", "").strip().lower() in ("1", "true", "yes")
    try:
        has_chromadb = importlib.util.find_spec("chromadb") is not None
        has_hnswlib = importlib.util.find_spec("hnswlib") is not None
    except Exception:
        return
    if not has_chromadb or has_hnswlib:
        return
    if skip:
        print(
            "[RAG-Lite] 检测到 chromadb 但缺少 hnswlib（常见报错: Error loading hnsw index）。\n"
            "[RAG-Lite] 已启用 RAG_LITE_SKIP_AUTO_PIP，请手动执行：\n"
            f"  {sys.executable} -m pip install chroma-hnswlib",
            flush=True,
        )
        return
    print("[RAG-Lite] 检测到 chromadb 缺少 hnswlib，正在补装 chroma-hnswlib …", flush=True)
    _pip_install("chroma-hnswlib")
    if importlib.util.find_spec("hnswlib") is None:
        raise SystemExit(
            "已安装 chroma-hnswlib，但仍无法导入 hnswlib。\n"
            "请确认解释器为项目 .venv，并手动执行：\n"
            f"  {sys.executable} -m pip install chroma-hnswlib"
        )


_ensure_chroma_hnswlib()


def _purge_gradio_modules() -> None:
    """部分导入失败时 sys.modules 里会留下坏模块，重装后必须先清掉再 import。"""
    for k in list(sys.modules):
        if k == "gradio" or k.startswith("gradio."):
            del sys.modules[k]


def _purge_pydantic_stack() -> None:
    """Pydantic v1 / 混装时 Gradio 4 会在 base.py 等处 ImportError；重装前清缓存。"""
    prefixes = ("pydantic", "pydantic_core", "annotated_types")
    for k in list(sys.modules):
        if k in prefixes or k.startswith("pydantic.") or k.startswith("pydantic_core."):
            del sys.modules[k]


def _purge_huggingface_hub_modules() -> None:
    for k in list(sys.modules):
        if k == "huggingface_hub" or k.startswith("huggingface_hub."):
            del sys.modules[k]


def _purge_after_failed_gradio_import() -> None:
    _purge_gradio_modules()
    _purge_huggingface_hub_modules()
    _purge_pydantic_stack()


def _import_gradio_with_auto_repair():
    """混装或缓存坏包导致 import 中途崩时，自动 pip 修复并重试。"""
    skip = os.environ.get("RAG_LITE_SKIP_AUTO_PIP", "").strip().lower() in ("1", "true", "yes")
    print(
        "[RAG-Lite] 正在 import Gradio（5.x；失败会自动尝试 pip reinstall）…",
        flush=True,
    )
    for round_i in range(3):
        try:
            import gradio as gr_mod

            return gr_mod
        except Exception as e:
            if skip or round_i >= 2:
                traceback.print_exc()
                raise SystemExit(
                    "[RAG-Lite] 无法 import gradio。可删除 ragZone\\.venv 后重新运行 start.bat，"
                    "或关闭 RAG_LITE_SKIP_AUTO_PIP 后重试。"
                ) from e
            print(f"[RAG-Lite] import gradio 异常: {type(e).__name__}: {e}", flush=True)
            _purge_after_failed_gradio_import()
            if round_i == 0:
                print(
                    "[RAG-Lite] 自动重装：Pydantic（与 Gradio 5 一致）+ gradio / gradio-client 锁定对 …",
                    flush=True,
                )
                _pip_install(
                    _PYDANTIC_GRADIO_PIN,
                    "typing-extensions>=4.8.0",
                    "--upgrade",
                    "--force-reinstall",
                    "--no-cache-dir",
                )
                _pip_uninstall_gradio_stack()
                _pip_install(
                    "httpx>=0.24.1",
                    _GRADIO_CLIENT_PIN,
                    _GRADIO_VERSION_PIN,
                    _TOMLKIT_PIN,
                    "--force-reinstall",
                    "--no-cache-dir",
                )
            else:
                req = ROOT / "requirements.txt"
                if not req.is_file():
                    raise SystemExit(f"未找到依赖清单: {req}") from e
                print("[RAG-Lite] 仍未恢复，按 requirements.txt --force-reinstall 重装依赖…", flush=True)
                _pip_install("-r", str(req), "--force-reinstall", "--no-cache-dir")


_t_gradio = time.perf_counter()
gr = _import_gradio_with_auto_repair()


def _gradio_major() -> int:
    try:
        return int(gr.__version__.split(".", 1)[0])
    except Exception:
        return 5


_GRADIO_MAJOR = _gradio_major()
CHAT_USE_OPENAI_MESSAGES = True
if _GRADIO_MAJOR >= 6:
    print(
        "[RAG-Lite] 提示: 检测到 Gradio 6+，导入会较慢。建议: pip install "
        f'{_GRADIO_CLIENT_PIN} {_GRADIO_VERSION_PIN} --force-reinstall',
        flush=True,
    )

print(
    f"[RAG-Lite] Gradio {gr.__version__} 已就绪，本步耗时 {time.perf_counter() - _t_gradio:.1f} 秒；"
    " Chatbot 数据模式=messages",
    flush=True,
)

from rag_lite.config import load_config
from rag_lite.ingest import (
    _filename_equiv,
    active_chroma_label,
    build_excluded_file_set,
    chroma_collection_count,
    chroma_sample_distinct_filenames,
    resolve_active_chroma_dir,
    uploaded_files_snapshot,
)
from rag_lite.ollama_util import list_ollama_models, merge_model_choices
from rag_lite import prefs as prefs_mod
from rag_lite.store import ExperimentStore
from rag_lite import eval_judge
from rag_lite.eval_platform import (
    attribution_code,
    attribution_rows,
    dataset_quality_rows,
    dataset_quality_summary,
)
from rag_lite.eval_runner import (
    effective_llm_model,
    normalize_generation_mode,
    normalize_retrieval_mode,
    retrieval_flags,
)
from rag_lite.platform_ops import (
    ollama_health_rows,
    ollama_health_snapshot,
    write_eval_case_compare,
    write_eval_run_report,
)
from rag_lite.ui_kb import (
    cleanup_old_index_versions,
    index_ops_diagnostics,
    index_ops_rows,
    index_ops_summary_html,
)

print("[RAG-Lite] 加载 config.yaml 与本地目录 …", flush=True)
cfg = load_config(ROOT)
try:
    from rag_lite.ensure_image_deps import main as _ensure_image_deps_main

    _ensure_image_deps_main()
except Exception as e:
    print(f"[RAG-Lite] 图片增强环境检测跳过或未完成: {e}", flush=True)
store = ExperimentStore(cfg.sqlite_path)
print("[RAG-Lite] 配置与 SQLite 就绪。", flush=True)

# 对话页三列对齐：左侧会话表与 Chatbot 同高；引用区在输入框下方且限高；右侧「生成与检索」总高不超过中间区（至引用区下沿）
RAG_CHATBOT_HEIGHT_PX = 400
RAG_SOURCES_PANEL_MAX_PX = 200
RAG_CHAT_INPUT_ROW_PX = 80
# 右侧栏最大高度 ≈ Chatbot + 引用区上限 + 输入行 + 组件边距（与中间可视栈对齐，避免右侧低于输入框）
# 额外留出高度，避免「LLM 上下文 num_ctx」等控件把「检索参数」挤出首屏
RAG_CHAT_RIGHT_EXTRA_PX = 140
RAG_CHAT_MIDDLE_STACK_PX = (
    RAG_CHATBOT_HEIGHT_PX + RAG_SOURCES_PANEL_MAX_PX + RAG_CHAT_INPUT_ROW_PX + 48 + RAG_CHAT_RIGHT_EXTRA_PX
)

# 知识库页三列（3:5:2）统一可视高度：左侧文档表 + 构建日志区域
RAG_KB_MAIN_MIN_HEIGHT_PX = 640
RAG_KB_FILES_TABLE_MAX_PX = 280
# 构建日志文本框固定可视高度（px），超出部分在框内滚动并由脚本跟到底部
RAG_KB_BUILD_LOG_TEXTAREA_PX = 260


def _ollama_base() -> str:
    return str(cfg.ollama.get("base_url", "http://127.0.0.1:11434"))


def _image_pipeline_prefs_merged() -> tuple[str, str, int, str]:
    """prefs 与 config 合并后的视觉模型、OCR 语言、跳过阈值、OCR 引擎。"""
    p = prefs_mod.load_prefs(cfg.data_dir)
    ing = cfg.ingest
    o = cfg.ollama
    vm = p.get("image_vision_model")
    if vm is None or (isinstance(vm, str) and not str(vm).strip()):
        vm = ing.get("vision_model") or o.get("llm_model") or "llava"
    vm = str(vm).strip() or "llava"
    lang = p.get("image_tesseract_lang")
    if lang is None or (isinstance(lang, str) and not str(lang).strip()):
        lang = ing.get("tesseract_lang") or "chi_sim+eng"
    lang = str(lang).strip() or "chi_sim+eng"
    skip = p.get("image_ocr_skip_vlm_min_chars")
    if skip is None:
        skip = ing.get("ocr_skip_vlm_min_chars", 20)
    try:
        skip = int(skip)
    except (TypeError, ValueError):
        skip = 20
    oe = p.get("image_ocr_engine")
    if oe is None or (isinstance(oe, str) and not str(oe).strip()):
        oe = ing.get("ocr_engine") or "tesseract"
    oe = str(oe).strip().lower()
    if oe not in ("tesseract", "paddleocr"):
        oe = "tesseract"
    return vm, lang, max(0, min(500, skip)), oe


def _vision_model_choices_for_ui(current: str) -> list[str]:
    from rag_lite.ollama_util import list_ollama_models, merge_model_choices

    names = list_ollama_models(_ollama_base())
    cur = (current or "").strip() or "llava"
    cand = [str(cfg.ingest.get("vision_model") or "llava").strip(), cur]
    return merge_model_choices(names, [c for c in cand if c], cur)


def _config_llm_candidates() -> list[str]:
    raw = cfg.raw.get("ollama") or {}
    v = raw.get("llm_models")
    if isinstance(v, list) and v:
        return [str(x) for x in v]
    m = raw.get("llm_model")
    return [str(m)] if m else []


def _config_embed_candidates() -> list[str]:
    raw = cfg.raw.get("ollama") or {}
    v = raw.get("embed_models")
    if isinstance(v, list) and v:
        return [str(x) for x in v]
    m = raw.get("embed_model")
    return [str(m)] if m else []


def refresh_model_choices() -> tuple[list[str], list[str], str, str]:
    ollama_names = list_ollama_models(_ollama_base())
    p = prefs_mod.load_prefs(cfg.data_dir)
    cur_llm = str(p.get("llm_model") or cfg.ollama.get("llm_model", ""))
    cur_emb = str(p.get("embed_model") or cfg.ollama.get("embed_model", ""))
    llm_choices = merge_model_choices(ollama_names, _config_llm_candidates(), cur_llm)
    emb_choices = merge_model_choices(ollama_names, _config_embed_candidates(), cur_emb)
    return llm_choices, emb_choices, cur_llm, cur_emb


def initial_llm_num_ctx_for_ui() -> int:
    """对话页「LLM 上下文窗口 num_ctx」初始值：优先 data/ui_preferences.json，其次 config.yaml。"""
    p = prefs_mod.load_prefs(cfg.data_dir)
    if "llm_num_ctx" in p and p["llm_num_ctx"] is not None:
        try:
            v = int(p["llm_num_ctx"])
            return max(2048, min(v, 262144))
        except (TypeError, ValueError):
            pass
    raw = cfg.raw.get("ollama") or {}
    if raw.get("num_ctx") is not None:
        try:
            v = int(raw["num_ctx"])
            if v > 0:
                return max(2048, min(v, 262144))
        except (TypeError, ValueError):
            pass
    return 16384


def _fmt_session_label(sid: int, title: str | None) -> str:
    t = (title or "未命名").strip() or "未命名"
    if len(t) > 42:
        t = t[:39] + "…"
    return f"#{sid} · {t}"


def _all_session_labels() -> list[str]:
    rows = store.list_sessions(limit=400)
    return [_fmt_session_label(int(r["id"]), r.get("title")) for r in rows]


def _format_session_time(iso_ts: str | None) -> str:
    if not iso_ts:
        return "—"
    s = str(iso_ts).strip()
    if "T" in s and len(s) >= 19:
        return s[:19].replace("T", " ")
    return s[:32] + ("…" if len(s) > 32 else "")


def _sessions_table_value() -> list[list[str]]:
    """与 store.list_sessions 同序：按 updated_at 倒序。供 Dataframe 展示。"""
    rows = store.list_sessions(limit=400)
    if not rows:
        return [["（暂无会话，请点击新建）", "—"]]
    out: list[list[str]] = []
    for r in rows:
        sid = int(r["id"])
        label = _fmt_session_label(sid, r.get("title"))
        out.append([label, _format_session_time(r.get("updated_at"))])
    return out


def session_history_for_chatbot(session_id: int) -> list:
    rows = store.fetch_qa_rows_for_session(session_id)
    out: list[dict[str, str]] = []
    for r in rows:
        out.append({"role": "user", "content": str(r.get("question") or "")})
        diag = r.get('diagnostics') or {}
        ui_progress = str(diag.get('ui_progress') or '').strip()
        ans = str(r.get('answer') or '')
        if ui_progress:
            ans = f'{ui_progress}\n\n---\n\n{ans}'
        out.append({"role": "assistant", "content": ans})
    return out


def kb_embed_warning_markdown(embed_selected: str) -> str:
    manifest = store.get_index_manifest()
    em = manifest.get("embed_model") if manifest else None
    parts: list[str] = []
    if em and em != embed_selected:
        parts.append(
            "**提示**：磁盘索引由嵌入模型 `" + str(em) + "` 构建，当前选择为 `" + str(embed_selected) + "`。"
            "若检索报错或效果异常，请改回一致模型或重新构建索引。"
        )
    index_state = _build_index_status_snapshot()
    parts.append(
        "**当前索引状态**："
        f"build_id=`{index_state.get('manifest_build_id')}`；"
        f"uploads=`{int(index_state.get('uploads_file_count') or 0)}`；"
        f"active一致=`{'是' if index_state.get('active_dir_matches_manifest') else '否'}`；"
        f"readiness=`{'通过' if index_state.get('readiness_ok') else '未通过'}`；"
        f"diagnostics=`{int(index_state.get('diagnostics_total_files') or 0)} 文件 / {int(index_state.get('diagnostics_total_chunks') or 0)} 切片`。"
    )
    if bool(index_state.get("uploads_changed_since_manifest")):
        parts.append("**警告**：uploads 已变化但 manifest/索引未同步，请先重建索引。")
    if int(index_state.get("missing_uploaded_files_count") or 0) > 0:
        parts.append(
            f"**警告**：readiness 检测到 {int(index_state.get('missing_uploaded_files_count') or 0)} 个上传文件未进入当前索引。"
        )
    return "\n\n".join(parts)



def _chroma_dir_ack_path() -> Path:
    """Path of the small JSON file used to remember that the user has
    acknowledged the chroma_dir silent-redirect warning."""
    return cfg.data_dir / ".chroma_dir_redirect_acknowledged.json"


def _chroma_dir_redirect_acknowledged() -> bool:
    info = cfg.chroma_dir_redirect_info
    if not info:
        return True
    p = _chroma_dir_ack_path()
    if not p.is_file():
        return False
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(d.get("acknowledged")) and d.get("redirect_final") == info["final"]


def kb_chroma_dir_redirect_warning_markdown() -> str:
    """Return a Markdown warning explaining that the chroma index has been
    silently relocated to %LOCALAPPDATA% because the project path contains
    non-ASCII characters. Returns an empty string when no redirect is
    active, or when the user has already acknowledged it for this exact
    target path."""
    info = cfg.chroma_dir_redirect_info
    if not info or _chroma_dir_redirect_acknowledged():
        return ""
    return (
        "**⚠️ 索引路径已重定向**：项目根目录含非 ASCII 字符，Chroma 索引实际存储于\n\n"
        f"  `{info['final']}`\n\n"
        f"原计划位置 `{info['original']}` 不会被使用。如需改回项目目录内，"
        "请在 `config.yaml` 把 `chroma_dir` 显式指向一个 ASCII 路径，"
        "或在 `%LOCALAPPDATA%` 之外选择一个 ASCII 目录。"
    )


def do_ack_chroma_dir_redirect():
    """Dismiss handler: record the user's acknowledgement and hide the
    warning + button. The ack file is keyed on the redirect target path,
    so if the project path changes (different slug), the warning will
    re-appear — which is the desired behavior."""
    info = cfg.chroma_dir_redirect_info
    if not info:
        return gr.update(visible=False), gr.update(visible=False)
    p = _chroma_dir_ack_path()
    try:
        payload = {
            "acknowledged": True,
            "redirect_final": info["final"],
            "redirect_original": info["original"],
            "reason": info.get("reason", ""),
            "acknowledged_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        print(f"[RAG-Lite] 写入 chroma 重定向确认文件失败: {exc}", file=sys.stderr, flush=True)
    return gr.update(visible=False), gr.update(visible=False)


def _norm_path_for_compare(path_text: str | None) -> str:
    t = str(path_text or "").strip()
    if not t:
        return ""
    t = t.replace("\\", "/")
    while "//" in t:
        t = t.replace("//", "/")
    return t.casefold()


def _build_index_status_snapshot() -> dict[str, Any]:
    manifest = store.get_index_manifest() or {}
    uploads = uploaded_files_snapshot(cfg)
    active_dir = resolve_active_chroma_dir(cfg, store=store)
    active_label = ""
    if active_dir is not None:
        try:
            active_label = str(active_dir.resolve())
        except Exception:
            active_label = str(active_dir)
    manifest_active = str(manifest.get("active_chroma_subdir") or "").strip()
    manifest_files = {
        str(x.get("name") or "").strip(): x
        for x in (manifest.get("files") or [])
        if isinstance(x, dict) and str(x.get("name") or "").strip()
    }
    uploads_changed = False
    upload_names = {str(x.get("name") or "").strip() for x in uploads if str(x.get("name") or "").strip()}
    for item in uploads:
        name = str(item.get("name") or "").strip()
        prev = manifest_files.get(name)
        if prev is None:
            uploads_changed = True
            break
        if int(prev.get("size", -1)) != int(item.get("size", -2)) or int(prev.get("mtime_ns", -1)) != int(item.get("mtime_ns", -2)):
            uploads_changed = True
            break
    if not uploads_changed:
        for name in manifest_files:
            if name not in upload_names:
                uploads_changed = True
                break
    active_norm = _norm_path_for_compare(active_label)
    manifest_norm = _norm_path_for_compare(manifest_active)
    active_matches_manifest = bool(
        manifest_norm
        and active_norm
        and (active_norm == manifest_norm or active_norm.endswith(manifest_norm))
    )
    readiness = dict(manifest.get("readiness") or {})
    final_health = dict(readiness.get("final_health") or {}) if isinstance(readiness.get("final_health"), dict) else {}
    return {
        "active_label": active_label or "（未发现可用索引目录）",
        "manifest_build_id": str(manifest.get("build_id") or "").strip() or "—",
        "manifest_active_chroma_subdir": manifest_active or "—",
        "readiness_ok": bool((readiness.get("ok") or final_health.get("ok")) and readiness.get("diagnostics_ok", True)),
        "uploads_changed_since_manifest": uploads_changed,
        "active_dir_matches_manifest": active_matches_manifest,
        "missing_uploaded_files_count": len(
            list(readiness.get("missing_uploaded_files") or final_health.get("missing_uploaded_files") or [])
        ),
        "diagnostics_total_chunks": int(
            readiness.get("diagnostics_total_chunks") or final_health.get("diagnostics_total_chunks") or 0
        ),
        "diagnostics_total_files": int(
            readiness.get("diagnostics_total_files") or final_health.get("diagnostics_total_files") or 0
        ),
        "uploads_file_count": len(uploads),
    }


def _index_state_blocking_message(index_state: dict[str, Any] | None) -> str:
    state = index_state or {}
    if bool(state.get("uploads_changed_since_manifest")):
        return "当前 uploads 已变更但索引未同步，请先重建索引后再继续。"
    if not bool(state.get("active_dir_matches_manifest")):
        return "当前激活索引与 manifest 不一致，请先确认当前激活版本。"
    if not bool(state.get("readiness_ok")):
        return "当前索引 readiness 未通过，请先修复索引健康状态。"
    return ""


def _index_state_summary_text(index_state: dict[str, Any] | None) -> str:
    state = index_state or {}
    return (
        f"build_id={state.get('manifest_build_id') or '—'}；"
        f"uploads={int(state.get('uploads_file_count') or 0)}；"
        f"active一致={'是' if state.get('active_dir_matches_manifest') else '否'}；"
        f"readiness={'通过' if state.get('readiness_ok') else '未通过'}；"
        f"diagnostics={int(state.get('diagnostics_total_files') or 0)} 文件 / {int(state.get('diagnostics_total_chunks') or 0)} 切片"
    )


def kb_files_table_value(embed_selected: str) -> list[list[str]]:
    manifest = store.get_index_manifest()
    disk = uploaded_files_snapshot(cfg)
    index_state = _build_index_status_snapshot()
    status_rows = [
        ["[索引状态] build_id", str(index_state.get("manifest_build_id") or "—"), "—", "—", "—"],
        ["[索引状态] active一致", "是" if index_state.get("active_dir_matches_manifest") else "否", "—", "—", "—"],
        ["[索引状态] readiness", "通过" if index_state.get("readiness_ok") else "未通过", "—", "—", "—"],
        ["[索引状态] uploads变化", "是" if index_state.get("uploads_changed_since_manifest") else "否", "—", "—", "—"],
        ["[索引状态] diagnostics", f"{int(index_state.get('diagnostics_total_files') or 0)} 文件 / {int(index_state.get('diagnostics_total_chunks') or 0)} 切片", "—", "—", "—"],
        ["[索引状态] 缺失上传文件", str(int(index_state.get("missing_uploaded_files_count") or 0)), "—", "—", "—"],
    ]
    if not disk:
        return status_rows + [["（暂无文件）", "—", "—", "—", "—"]]
    mf_files: dict[str, dict] = {}
    built_at_global = manifest.get("built_at") if manifest else None
    if manifest:
        for x in manifest.get("files") or []:
            if isinstance(x, dict) and x.get("name"):
                mf_files[str(x["name"])] = x
    out: list[list[str]] = []
    for f in disk:
        name = f["name"]
        sz = int(f["size"])
        sz_s = f"{sz / 1024:.1f} KB" if sz < 1024 * 1024 else f"{sz / (1024 * 1024):.1f} MB"
        up_t = _format_session_time(store.get_latest_upload_time_for_filename(name))
        prev = mf_files.get(name)
        vec_t = "—"
        st = "未索引"
        if manifest:
            active_dir = resolve_active_chroma_dir(cfg, store=store)
            manifest_active = str(manifest.get("active_chroma_subdir") or "").strip()
            active_label = ""
            if active_dir is not None:
                try:
                    active_label = str(active_dir)
                except Exception:
                    active_label = ""
            active_matches_manifest = bool(
                _norm_path_for_compare(manifest_active)
                and _norm_path_for_compare(active_label)
                and (
                    _norm_path_for_compare(active_label) == _norm_path_for_compare(manifest_active)
                    or _norm_path_for_compare(active_label).endswith(_norm_path_for_compare(manifest_active))
                )
            )
            if prev and prev.get("mtime_ns") == f.get("mtime_ns") and int(prev.get("size", -1)) == sz:
                if active_matches_manifest:
                    st = "✓ 已入库（与当前激活索引一致）"
                else:
                    st = "⚠ Manifest 已记录，但当前激活索引版本待确认"
                vec_t = _format_session_time(prev.get("indexed_at") or built_at_global)
            elif prev:
                st = "⚠ 文件已变更，需重建索引"
                vec_t = _format_session_time(prev.get("indexed_at") or built_at_global)
            else:
                st = "未索引（相对上次构建为新增）"
        out.append([name, sz_s, up_t, vec_t, st])
    return status_rows + out


def remove_kb_file(
    row_idx: Any,
    embed_selected: str,
) -> tuple[str, list[list[str]], None]:
    es = embed_selected or ""
    disk = uploaded_files_snapshot(cfg)
    ri = _coerce_kb_row_index(row_idx)
    if ri is None or not (0 <= ri < len(disk)):
        return "请先在表格中选中要移除的文件。", kb_files_table_value(es), None
    fn = disk[ri]["name"]
    p = (cfg.uploads_dir / fn).resolve()
    uploads_resolved = cfg.uploads_dir.resolve()
    try:
        p.relative_to(uploads_resolved)
    except ValueError:
        return "路径非法。", kb_files_table_value(es), None
    if not p.is_file():
        return f"文件不存在：{fn}", kb_files_table_value(es), None
    try:
        p.unlink()
    except OSError as e:
        return f"删除失败：{e}", kb_files_table_value(es), None
    return (
        f"已从上传目录删除「{fn}」。若此前建过索引，请重新点击「构建向量索引」以同步向量库。",
        kb_files_table_value(es),
        None,
    )


_KB_CHUNK_PREVIEW_EMPTY_HTML = (
    '<p style="color:#64748b;font-size:0.88em;margin:0;">在上方表格中<strong>点击一行</strong>选中文件，再点「预览切片」'
    "查看当前 Chroma 向量库中的文本块。</p>"
)


def _canonical_upload_filename(sel: str | None, disk: list[dict]) -> str | None:
    """将 State 或截断后的显示名还原为上传目录中的真实文件名。"""
    if not sel or not disk:
        return None
    s = str(sel).strip()
    if s.startswith("（暂无"):
        return None
    for f in disk:
        if f["name"] == s or _filename_equiv(f["name"], s):
            return f["name"]
    base = s.rstrip("…").rstrip(".").strip()
    for f in disk:
        fn = f["name"]
        if fn.startswith(base):
            return fn
        if len(base) >= 6 and (base in fn or fn in base):
            return fn
    for f in disk:
        if Path(f["name"]).stem == Path(s).stem:
            return f["name"]
    return Path(s).name


def _coerce_kb_row_index(x: Any) -> int | None:
    """Gradio / numpy 可能给出 numpy.int64 等，不能用 isinstance(..., int) 判断。"""
    if x is None:
        return None
    try:
        i = operator.index(x)
    except (TypeError, ValueError):
        return None
    return int(i)


def _kb_chunk_preview_html(row_idx: Any, embed_selected: str | None = None) -> str:
    """知识库：按当前表格行对应的上传目录真实文件名展示切片（行号解析，避免 State 字符串与向量库不一致）。"""
    ing = _lazy_ingest()
    eng = _lazy_engine()
    disk = uploaded_files_snapshot(cfg)
    fn = None
    ri = _coerce_kb_row_index(row_idx)
    if ri is not None and 0 <= ri < len(disk):
        fn = disk[ri]["name"]
    if not fn:
        return '<p style="color:#92400e;margin:0;">请先在表格中<strong>点击一行</strong>选中文件。</p>'
    em = (embed_selected or "").strip() or None
    index = eng.get_index(cfg, embed_model=em)
    # Preview should reflect the currently activated on-disk index, not a possibly stale cached collection.
    chunks, total_found, source_info = ing.fetch_chunks_for_file(cfg, fn, index=index, prefer_index=False)
    chroma_path = html.escape(str(source_info.get("active_chroma_dir") or active_chroma_label(cfg, store=store)))
    source = str(source_info.get("source") or "none")
    fallback_used = bool(source_info.get("fallback_used"))
    fallback_reason = str(source_info.get("fallback_reason") or "")
    file_diag = dict(source_info.get("file_diag") or {})
    source_bits = [f"<strong>当前读取来源：</strong>{html.escape(source)}"]
    if fallback_used:
        source_bits.append("<strong>已回退到磁盘集合</strong>")
    if fallback_reason:
        source_bits.append(f"原因：<code>{html.escape(fallback_reason)}</code>")
    source_html = f"<p style='margin:0.35em 0 0 0;color:#92400e;font-size:0.9em;'>{'；'.join(source_bits)}</p>" if fallback_used else ""
    if total_found == 0:
        cc = chroma_collection_count(cfg, index=index, prefer_index=False)
        if cc == 0:
            return (
                f'<p style="color:#92400e;margin:0;">当前向量库为空（共 0 条，路径：<code>{chroma_path}</code>）。'
                "请先<strong>构建向量索引</strong>。</p>"
                f"{source_html}"
            )
        sample = chroma_sample_distinct_filenames(cfg, limit=24, index=index, prefer_index=True)
        sample_html = ""
        if sample:
            esc = [html.escape(x) for x in sample]
            sample_html = (
                "<br /><br />当前向量库中解析到的文件名示例（供核对）："
                "<br /><code style=\"font-size:0.85em;word-break:break-all;\">"
                + "； ".join(esc)
                + "</code>"
            )
        return (
            '<p style="color:#92400e;margin:0;">未找到该文件的切片（向量库中共有 '
            f"<strong>{cc}</strong> 条记录，当前激活路径：<code>{chroma_path}</code>）。"
            f"<br /><strong>文档状态：</strong>{html.escape(str(file_diag.get('index_status_label') or '未入库（0 块）'))}"
            f"　<strong>分类：</strong>{html.escape(str(file_diag.get('pdf_doc_class_label') or '—'))}"
            f"　<strong>依据：</strong>{html.escape(str(file_diag.get('pdf_doc_class_reason_label') or '—'))}"
            "<br />请先在表格中<strong>重新点击该行</strong>再预览；"
            f"若仍失败，请<strong>重新构建向量索引</strong>，并核对下方文件名是否与「{html.escape(fn)}」一致。"
            f"{sample_html}</p>"
            f"{source_html}"
        )
    note = ""
    if total_found > len(chunks):
        note = f"（数据库中共 {total_found} 块，以下仅展示前 {len(chunks)} 块）"
    file_diag_html = ""
    if file_diag:
        file_diag_html = (
            "<p style='margin:0.35em 0 0 0;color:#475569;font-size:0.9em;'>"
            f"<strong>文档分类：</strong>{html.escape(str(file_diag.get('pdf_doc_class_label') or '—'))}"
            f"　<strong>依据：</strong>{html.escape(str(file_diag.get('pdf_doc_class_reason_label') or '—'))}"
            f"　<strong>文本页/总页：</strong>{int(file_diag.get('pdf_text_pages') or 0)}/{int(file_diag.get('pdf_page_count') or 0)}"
            f"　<strong>可疑页占比：</strong>{int(file_diag.get('pdf_suspicious_page_ratio_pct') or 0)}%"
            "</p>"
        )
    head = (
        f'<p style="margin:0 0 10px 0;color:#475569;font-size:0.9em;">'
        f"匹配到 <strong>{total_found}</strong> 个文本块{note}."
        f"<br /><strong>当前激活路径：</strong><code>{chroma_path}</code></p>"
        f"{source_html}{file_diag_html}"
    )
    parts: list[str] = []
    for c in chunks:
        t = html.escape(c["text"])
        nid = html.escape(str(c.get("node_id", "")))
        parts.append(
            '<details style="margin:8px 0;border:1px solid #e5e7eb;border-radius:8px;padding:6px 10px;'
            'background:#fff;">'
            f'<summary style="cursor:pointer;font-weight:600;color:#1565c0;">切片 {c["i"]} '
            f'<span style="color:#64748b;font-weight:400;font-size:0.85em;">({nid})</span></summary>'
            f'<pre style="white-space:pre-wrap;word-break:break-word;margin:10px 0 0 0;font-size:0.88em;'
            f'line-height:1.5;color:#1f2937;">{t}</pre>'
            "</details>"
        )
    return f'<div class="rag-kb-chunk-preview-inner">{head}{"".join(parts)}</div>'


_ingest_mod = None
_engine_mod = None


def _lazy_ingest():
    """Chroma / 解析 / 建索引 —— 首次点「保存」「构建索引」时才加载。"""
    global _ingest_mod
    if _ingest_mod is None:
        print("[RAG-Lite] 首次知识库操作：加载 ingest（Chroma、LlamaIndex）…", flush=True)
        t0 = time.perf_counter()
        from rag_lite import ingest

        _ingest_mod = ingest
        print(f"[RAG-Lite] ingest 已加载（{time.perf_counter() - t0:.1f}s）", flush=True)
    return _ingest_mod


def _lazy_engine():
    """检索与 Ollama 流式 —— 首次对话时才加载。"""
    global _engine_mod
    if _engine_mod is None:
        print("[RAG-Lite] 首次对话：加载 engine（检索、Ollama）…", flush=True)
        t0 = time.perf_counter()
        from rag_lite import engine

        _engine_mod = engine
        print(f"[RAG-Lite] engine 已加载（{time.perf_counter() - t0:.1f}s）", flush=True)
    return _engine_mod

LAST_QA_ID: int | None = None

# 对话页「引用」面板初始/清空时的占位 HTML（与 Gradio HTML 组件配合）
_SOURCES_EMPTY_HTML = (
    '<div class="rag-tip-block rag-tip-block--tight" style="margin-bottom:0;">'
    '<p style="margin:0;color:#64748b;">回答生成后，此处展示检索到的文档片段与相似度 / 重排得分。</p>'
    "</div>"
)

_DIAG_EMPTY_HTML = (
    '<div class="rag-tip-block rag-tip-block--tight" style="margin-bottom:0;">'
    '<p style="margin:0;color:#64748b;">发送问题后，这里会展示本轮的检索诊断：Top-N 候选、最终送入上下文的 Top-K、是否发生重排，以及用于效果排查的摘要。</p>'
    "</div>"
)

# 页头：「!」+ checkbox；浮层 fixed，位置由 launch(head) 注入的 rag_tip_popover.js 算在按钮右侧
_RAG_HEADER_HTML = """
<div class="rag-workbench-header">
<div class="rag-header-title-row">
<h1>RAG 验证工作台</h1>
<div class="rag-tip-anchor rag-tip-anchor--page">
<input type="checkbox" id="rag-tip-cb-page" class="rag-tip-cb" tabindex="-1" aria-hidden="true"/>
<label for="rag-tip-cb-page" class="rag-tip-trigger rag-tip-trigger--page" title="使用说明">!</label>
<div id="rag-tip-pop-page" class="rag-tip-pop rag-tip-pop--wide">
<div class="rag-tip-block rag-tip-block--header rag-tip-block--in-pop">
<p>本地数据与模型，用于检索与生成效果对比。请先运行
<a href="https://ollama.com" target="_blank" rel="noopener noreferrer">Ollama</a>
并拉取所需模型。向量模型在<strong>知识库</strong>选择；对话模型在<strong>对话</strong>选择；
二者可与 <code>config.yaml</code> 默认值不同，并会记住上次选择。</p>
<p class="rag-workbench-tip">流程：上传文档 → 选择嵌入模型并构建索引 → 在对话中选择 LLM 提问；
左侧可切换历史会话；导出备份在<strong>对话</strong>页侧栏。</p>
</div>
</div>
</div>
</div>
</div>
"""

_KB_SECTION_UPLOADED_HTML = """
<div class="rag-kb-section-head rag-kb-section-head--h5">
<h5>已上传文档</h5>
<div class="rag-tip-anchor">
<input type="checkbox" id="rag-tip-cb-kb-files" class="rag-tip-cb" tabindex="-1" aria-hidden="true"/>
<label for="rag-tip-cb-kb-files" class="rag-tip-trigger rag-tip-trigger--h5" title="表格说明">!</label>
<div id="rag-tip-pop-kb-files" class="rag-tip-pop">
<div class="rag-tip-block rag-tip-block--tight rag-tip-block--in-pop">点击表格<strong>任一格</strong>选中整行，再点下方按钮从上传目录删除。</div>
</div>
</div>
</div>
"""
_KB_EMBED_HEAD_HTML = """
<div class="rag-kb-embed-head">
<span class="rag-kb-embed-label">嵌入模型（Ollama）</span>
<div class="rag-tip-anchor rag-tip-anchor--align-end">
<input type="checkbox" id="rag-tip-cb-kb-embed" class="rag-tip-cb" tabindex="-1" aria-hidden="true"/>
<label for="rag-tip-cb-kb-embed" class="rag-tip-trigger rag-tip-trigger--h5" title="嵌入模型说明">!</label>
<div id="rag-tip-pop-kb-embed" class="rag-tip-pop">
<div class="rag-tip-block rag-tip-block--tight rag-tip-block--in-pop">若仅有一个候选，将自动固定为该模型。切换嵌入模型后请重新构建索引。</div>
</div>
</div>
</div>
"""
_CHAT_SESSION_HEAD_HTML = """
<div class="rag-kb-section-head rag-kb-section-head--h5">
<h5>历史会话</h5>
<div class="rag-tip-anchor">
<input type="checkbox" id="rag-tip-cb-chat-session" class="rag-tip-cb" tabindex="-1" aria-hidden="true"/>
<label for="rag-tip-cb-chat-session" class="rag-tip-trigger rag-tip-trigger--h5" title="会话列表说明">!</label>
<div id="rag-tip-pop-chat-session" class="rag-tip-pop">
<div class="rag-tip-block rag-tip-block--tight rag-tip-block--in-pop">按更新时间倒序；点击一行切换当前会话。</div>
</div>
</div>
</div>
"""
_CHAT_EXPORT_HEAD_HTML = """
<div class="rag-kb-section-head rag-kb-section-head--h5">
<h5>导出备份</h5>
<div class="rag-tip-anchor">
<input type="checkbox" id="rag-tip-cb-chat-export" class="rag-tip-cb" tabindex="-1" aria-hidden="true"/>
<label for="rag-tip-cb-chat-export" class="rag-tip-trigger rag-tip-trigger--h5" title="导出说明">!</label>
<div id="rag-tip-pop-chat-export" class="rag-tip-pop">
<div class="rag-tip-block rag-tip-block--tight rag-tip-block--in-pop">将本地 SQLite 中的 <code>qa_log</code> 导出为 json / csv 全库备份，路径见下方。</div>
</div>
</div>
</div>
"""


def _flatten_content(content) -> str:
    """Gradio 6 的 content 可能是 str，也可能是多段 list（如 {\"type\":\"text\",\"text\":...}）。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text" and "text" in part:
                    parts.append(str(part["text"]))
                elif "text" in part:
                    parts.append(str(part["text"]))
            elif isinstance(part, str):
                parts.append(part)
        return "".join(parts)
    return str(content)


def _one_history_item_to_messages(item) -> list[dict]:
    """将 Gradio 传入的一条记录转为 0～2 条标准 message dict。"""
    if item is None:
        return []
    if isinstance(item, str):
        return [{"role": "user", "content": item}]
    if isinstance(item, (list, tuple)) and len(item) >= 2:
        return [
            {"role": "user", "content": str(item[0]) if item[0] is not None else ""},
            {"role": "assistant", "content": str(item[1]) if item[1] is not None else ""},
        ]
    if hasattr(item, "model_dump"):
        try:
            item = item.model_dump()
        except Exception:
            pass
    if isinstance(item, dict) and "role" in item and "content" in item:
        role = str(item["role"])
        text = _flatten_content(item["content"])
        return [{"role": role, "content": text}]
    return []


def _as_messages(history: list | None) -> list[dict]:
    """在内存中统一为 role/content 列表，便于流式改写；输出给 Chatbot 时再换格式。"""
    if not history:
        return []
    out: list[dict] = []
    for slot in list(history):
        out.extend(_one_history_item_to_messages(slot))
    return out


def _chatbot_value(messages: list[dict]) -> list:
    return messages


def _preview_chunk_text(raw: str, max_chars: int = 160) -> str:
    """单行化并截断，用于列表缩略预览。"""
    t = " ".join(str(raw).split())
    if len(t) > max_chars:
        return t[: max_chars - 1] + "…"
    return t


def _sources_panel_html(sources: list) -> str:
    """引用区：缩略列表 + 点击「查看全文」用 dialog 浮层展示完整片段（避免默认铺满过长）。"""
    intro = (
        '<p style="margin:0 0 8px 0;color:#444;font-size:0.95em;">'
        "以下为本次回答所<strong>引用</strong>的源文档；列表为片段摘要，点击<strong>查看全文</strong>在浮窗中阅读完整内容。"
        "</p>"
    )
    if not sources:
        return (
            intro
            + '<p style="color:#888;font-style:italic;">（本轮无检索结果或未命中片段）</p>'
        )

    def _score_badge_class(kind: str) -> str:
        if kind == "rerank":
            return "#e8f5e9"
        return "#e3f2fd"

    cards: list[str] = []
    for i, s in enumerate(sources, 1):
        kind_key = s.get("score_kind") or ""
        kind_label = (
            "Cross-Encoder 重排（相关性 0–1）"
            if kind_key == "rerank"
            else "向量检索得分"
        )
        score = s.get("score")
        score_s = f"{score:.4f}" if score is not None else "—"
        vec_sc = s.get("vector_score")
        vec_s = f"{vec_sc:.4f}" if vec_sc is not None else None
        fname = html.escape(str(s.get("file_name", "")))
        chunk_raw = str(s.get("chunk", ""))
        chunk_esc = html.escape(chunk_raw).replace("\n", "<br />\n")
        preview_plain = _preview_chunk_text(chunk_raw, 160)
        preview_esc = html.escape(preview_plain)
        badge_bg = _score_badge_class(kind_key)
        dlg_id = f"rag-src-dlg-{i}"
        if kind_key == "rerank" and vec_s is not None:
            score_line = (
                f'<span style="background:#e3f2fd;padding:2px 10px;border-radius:999px;'
                f'font-size:0.8em;border:1px solid rgba(0,0,0,0.06);white-space:nowrap;">'
                f"向量检索：<strong>{html.escape(vec_s)}</strong></span>"
                f'<span style="background:{badge_bg};padding:2px 10px;border-radius:999px;'
                f'font-size:0.8em;border:1px solid rgba(0,0,0,0.06);white-space:nowrap;">'
                f"{html.escape(kind_label)}：<strong>{html.escape(score_s)}</strong></span>"
            )
            dlg_scores = f"向量检索：{html.escape(vec_s)}　{html.escape(kind_label)}：{html.escape(score_s)}"
        else:
            score_line = (
                f'<span style="background:{badge_bg};padding:2px 10px;border-radius:999px;'
                f'font-size:0.8em;border:1px solid rgba(0,0,0,0.06);white-space:nowrap;">'
                f"{html.escape(kind_label)}：<strong>{html.escape(score_s)}</strong></span>"
            )
            dlg_scores = f"{html.escape(kind_label)}：{html.escape(score_s)}"
        cards.append(
            f'<div style="border:1px solid #e5e7eb;border-radius:8px;padding:10px 12px;margin-bottom:8px;'
            f'background:#fff;box-shadow:0 1px 2px rgba(0,0,0,0.04);">'
            f'<div style="display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin-bottom:6px;">'
            f'<span style="font-weight:600;color:#1565c0;font-size:0.95em;">[{i}] {fname}</span>'
            f"{score_line}"
            f"</div>"
            f'<p style="margin:0 0 8px 0;font-size:0.88em;color:#4b5563;line-height:1.45;'
            f"overflow:hidden;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;\">"
            f"{preview_esc}</p>"
            f'<button type="button" style="font-size:0.85em;padding:5px 12px;border-radius:6px;'
            f'border:1px solid #1565c0;background:#fff;color:#1565c0;cursor:pointer;"'
            f"onclick=\"document.getElementById('{dlg_id}').showModal()\">查看全文</button>"
            f"</div>"
            f'<dialog id="{dlg_id}" class="rag-src-dialog" style="max-width:min(92vw,560px);width:100%;'
            f"border:none;border-radius:14px;padding:0;box-shadow:0 16px 48px rgba(0,0,0,0.2);\">"
            f'<div style="padding:12px 16px;border-bottom:1px solid #eee;background:#fafafa;border-radius:14px 14px 0 0;">'
            f'<div style="font-weight:600;color:#1565c0;font-size:0.95em;">[{i}] {fname}</div>'
            f'<div style="font-size:0.8em;color:#64748b;margin-top:4px;">'
            f"{dlg_scores}</div></div>"
            f'<div style="padding:14px 16px;max-height:min(58vh,440px);overflow:auto;line-height:1.65;'
            f'font-size:0.9em;color:#1f2937;background:#fff;">{chunk_esc}</div>'
            f'<form method="dialog" style="margin:0;padding:10px 14px;border-top:1px solid #eee;'
            f'text-align:right;background:#f9fafb;border-radius:0 0 14px 14px;">'
            f'<button type="submit" style="padding:6px 18px;border-radius:8px;border:none;'
            f'background:#1565c0;color:#fff;cursor:pointer;font-size:0.9em;">关闭</button>'
            f"</form></dialog>"
        )

    body = "".join(cards)
    return (
        intro
        + '<details open style="margin-top:4px;border:1px solid #e0e0e0;border-radius:8px;padding:8px 12px;background:#fafafa;">'
        + '<summary style="cursor:pointer;font-weight:600;color:#333;user-select:none;">'
        "参考来源（缩略列表 · 点击查看全文）"
        "</summary>"
        + '<div style="margin-top:10px;">'
        + body
        + "</div></details>"
    )


def _compact_diag_sources(rows: list[dict[str, Any]], limit: int = 12) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, s in enumerate(rows[: max(1, int(limit))], start=1):
        out.append(
            {
                "rank": i,
                "file_name": str(s.get("file_name") or "unknown"),
                "score": s.get("score"),
                "score_kind": str(s.get("score_kind") or ""),
                "vector_score": s.get("vector_score"),
                "keyword_score": s.get("keyword_score"),
                "merged_score": s.get("merged_score"),
                "retrieval_sources": list(s.get("retrieval_sources") or []),
                "preview": _preview_chunk_text(str(s.get("chunk") or ""), 220),
            }
        )
    return out


def _retrieval_diag_payload(
    *,
    query: str,
    vector_candidates: list[dict[str, Any]],
    final_sources: list[dict[str, Any]],
    keyword_candidates: list[dict[str, Any]] | None = None,
    merged_candidates: list[dict[str, Any]] | None = None,
    top_n: int,
    top_k: int,
    use_rerank: bool,
    score_kind: str,
    excluded_files: list[str] | None = None,
    excluded_doc_classes: list[str] | None = None,
    include_zero_chunk: bool | None = None,
    excluded_candidate_count: int = 0,
    raw_query: str | None = None,
    query_anchors: list[str] | None = None,
    retrieval_mode: str | None = None,
) -> dict[str, Any]:
    keyword_candidates = list(keyword_candidates or [])
    merged_candidates = list(merged_candidates or vector_candidates or [])
    before_names = [str(x.get("file_name") or "") for x in merged_candidates[: max(1, min(top_k, len(merged_candidates)))]]
    after_names = [str(x.get("file_name") or "") for x in final_sources]
    excluded_clean = [str(x) for x in (excluded_files or []) if str(x or "").strip()]
    excluded_classes_clean = [str(x) for x in (excluded_doc_classes or []) if str(x or "").strip()]
    anchor_list = [str(x).strip() for x in (query_anchors or []) if str(x or "").strip()]
    return {
        "query": query,
        "raw_query": str(raw_query or query),
        "retrieval_query": query,
        "query_anchors": anchor_list,
        "top_n": int(top_n),
        "top_k": int(top_k),
        "use_rerank": bool(use_rerank),
        "score_kind": score_kind,
        "candidate_count": len(merged_candidates),
        "vector_candidate_count": len(vector_candidates),
        "keyword_candidate_count": len(keyword_candidates),
        "merged_candidate_count": len(merged_candidates),
        "final_count": len(final_sources),
        "retrieval_mode": retrieval_mode or ("hybrid" if keyword_candidates else "vector"),
        "rerank_changed_order": before_names != after_names if bool(use_rerank) else False,
        "excluded_files": excluded_clean,
        "excluded_file_count": len(excluded_clean),
        "excluded_file_preview": excluded_clean[:10],
        "excluded_doc_classes": excluded_classes_clean,
        "include_zero_chunk": bool(include_zero_chunk) if include_zero_chunk is not None else None,
        "excluded_candidate_count": int(excluded_candidate_count or 0),
        "vector_candidates": _compact_diag_sources(vector_candidates, limit=20),
        "keyword_candidates": _compact_diag_sources(keyword_candidates, limit=20),
        "merged_candidates": _compact_diag_sources(merged_candidates, limit=20),
        "final_sources": _compact_diag_sources(final_sources, limit=12),
        "final_contexts": _compact_diag_sources(final_sources, limit=12),
    }


def _retrieval_diag_html(diag: dict[str, Any] | None) -> str:
    if not diag:
        return _DIAG_EMPTY_HTML
    cand = list(diag.get("vector_candidates") or [])
    keyword = list(diag.get("keyword_candidates") or [])
    merged = list(diag.get("merged_candidates") or [])
    finals = list(diag.get("final_contexts") or [])
    use_rerank = bool(diag.get("use_rerank"))
    score_kind = str(diag.get("score_kind") or "vector")
    summary = (
        f'<div class="rag-tip-block rag-tip-block--tight" style="margin-bottom:8px;">'
        f'<p style="margin:0;"><strong>Merged candidates:</strong>{int(diag.get("candidate_count") or 0)}'
        f' <strong>Vector:</strong>{int(diag.get("vector_candidate_count") or 0)}'
        f' <strong>Keyword:</strong>{int(diag.get("keyword_candidate_count") or 0)}'
        f' <strong>Context:</strong>{int(diag.get("final_count") or 0)}'
        f' <strong>Rerank:</strong>{"on" if use_rerank else "off"}'
        f' <strong>Order changed:</strong>{"yes" if diag.get("rerank_changed_order") else "no"}</p>'
        f'<p style="margin:0.35em 0 0 0;"><strong>Excluded candidates:</strong>{int(diag.get("excluded_candidate_count") or 0)}'
        f' <strong>Excluded files:</strong>{int(diag.get("excluded_file_count") or 0)}'
        f' <strong>Excluded classes:</strong>{html.escape("/".join([str(x) for x in (diag.get("excluded_doc_classes") or [])]) or "-")}'
        f' <strong>Includes zero-chunk:</strong>{"yes" if diag.get("include_zero_chunk") else "no"}</p>'
        f"</div>"
    )

    def _table(title: str, rows: list[dict[str, Any]], show_vector: bool) -> str:
        if not rows:
            return f'<p style="margin:0 0 8px 0;color:#64748b;">{html.escape(title)}：无数据。</p>'
        body: list[str] = []
        for row in rows:
            rank = int(row.get("rank") or 0)
            fname = html.escape(str(row.get("file_name") or "unknown"))
            score = row.get("score")
            score_s = "—" if score is None else f"{float(score):.4f}"
            extras = ""
            if show_vector:
                vec = row.get("vector_score")
                vec_s = "—" if vec is None else f"{float(vec):.4f}"
                extras = f" / 向量 {vec_s}"
            preview = html.escape(str(row.get("preview") or ""))
            body.append(
                "<tr>"
                f"<td style='padding:6px 8px;border-top:1px solid #e5e7eb;'>{rank}</td>"
                f"<td style='padding:6px 8px;border-top:1px solid #e5e7eb;'>{fname}</td>"
                f"<td style='padding:6px 8px;border-top:1px solid #e5e7eb;white-space:nowrap;'>{score_s}{extras}</td>"
                f"<td style='padding:6px 8px;border-top:1px solid #e5e7eb;'>{preview}</td>"
                "</tr>"
            )
        return (
            f"<details open style='margin-top:8px;border:1px solid #e5e7eb;border-radius:8px;background:#fff;'>"
            f"<summary style='padding:8px 12px;cursor:pointer;font-weight:600;color:#334155;'>{html.escape(title)}</summary>"
            "<div style='padding:0 8px 8px 8px;overflow-x:auto;'>"
            "<table style='width:100%;border-collapse:collapse;font-size:0.88em;'>"
            "<thead><tr>"
            "<th style='text-align:left;padding:6px 8px;'>#</th>"
            "<th style='text-align:left;padding:6px 8px;'>文件</th>"
            "<th style='text-align:left;padding:6px 8px;'>得分</th>"
            "<th style='text-align:left;padding:6px 8px;'>片段摘要</th>"
            "</tr></thead>"
            f"<tbody>{''.join(body)}</tbody></table></div></details>"
        )

    final_title = "Final Top-K contexts"
    if score_kind == "rerank":
        final_title += " (after rerank)"
    return (
        summary
        + _table("Vector Top-N candidates", cand, show_vector=True)
        + _table("Keyword/BM25 Top-N candidates", keyword, show_vector=True)
        + _table("Merged candidates", merged, show_vector=True)
        + _table(final_title, finals, show_vector=True)
    )

def _chunk_diag_summary_html(diag: dict[str, Any]) -> str:
    summary = dict(diag.get("summary") or {})
    chroma_path = html.escape(str(diag.get("collection_active_dir") or diag.get("active_chroma_label") or "（未发现可用索引目录）"))
    source = html.escape(str(diag.get("collection_source") or "none"))
    fallback_used = bool(diag.get("collection_fallback_used"))
    fallback_reason = str(diag.get("collection_fallback_reason") or "")
    source_line = f"<p style='margin:0.35em 0 0 0;'><strong>当前读取来源：</strong>{source}</p>"
    if fallback_used:
        extra = "；<strong>已回退到磁盘集合</strong>"
        if fallback_reason:
            extra += f"；原因：<code>{html.escape(fallback_reason)}</code>"
        source_line = f"<p style='margin:0.35em 0 0 0;color:#92400e;'><strong>当前读取来源：</strong>{source}{extra}</p>"
    if int(summary.get("total_chunks") or 0) <= 0:
        return (
            '<div class="rag-tip-block rag-tip-block--tight" style="margin-bottom:0;">'
            '<p style="margin:0;color:#92400e;">当前向量库为空，或暂时无法读取切片统计。请先完成构建。</p>'
            f"<p style='margin:0.35em 0 0 0;'><strong>当前激活路径：</strong><code>{chroma_path}</code></p>"
            f"{source_line}"
            "</div>"
        )
    return (
        '<div class="rag-tip-block rag-tip-block--tight" style="margin-bottom:0;">'
        f"<p style='margin:0;'><strong>总块数：</strong>{int(summary.get('total_chunks') or 0)}"
        f"　<strong>文件数：</strong>{int(summary.get('total_files') or 0)}"
        f"　<strong>已入库文件：</strong>{int(summary.get('indexed_files') or 0)}"
        f"　<strong>未入库文件：</strong>{int(summary.get('zero_chunk_files') or 0)}"
        f"　<strong>平均字符数：</strong>{summary.get('avg_chars') or 0}</p>"
        f"<p style='margin:0.35em 0 0 0;'><strong>空块：</strong>{int(summary.get('empty_chunks') or 0)}"
        f"　<strong>图片提示块：</strong>{int(summary.get('image_hint_chunks') or 0)}"
        f"　<strong>OCR 提示块：</strong>{int(summary.get('ocr_hint_chunks') or 0)}"
        f"　<strong>视觉提示块：</strong>{int(summary.get('vision_hint_chunks') or 0)}</p>"
        f"<p style='margin:0.35em 0 0 0;'><strong>疑似扫描文件：</strong>{int(summary.get('scan_doc_files') or 0)}"
        f"　<strong>OCR 污染文件：</strong>{int(summary.get('ocr_polluted_files') or 0)}"
        f"　<strong>抽取失败文件：</strong>{int(summary.get('extract_failed_files') or 0)}</p>"
        f"<p style='margin:0.35em 0 0 0;'><strong>当前激活路径：</strong><code>{chroma_path}</code></p>"
        f"{source_line}"
        "</div>"
    )


def _experiment_compare_rows(rows: list[dict[str, Any]]) -> list[list[Any]]:
    groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        p = dict(row.get("params") or {})
        key = (
            str(p.get("embed_model") or ""),
            str(p.get("llm_model") or ""),
            str(p.get("chunk_mode") or ""),
            int(p.get("chunk_size") or 0),
            int(p.get("chunk_overlap") or 0),
            bool(p.get("use_rerank")),
            int(p.get("top_n") or 0),
            int(p.get("top_k") or 0),
        )
        g = groups.setdefault(
            key,
            {
                "count": 0,
                "rated_count": 0,
                "rating_sum": 0.0,
                "no_ctx_count": 0,
                "session_ids": set(),
            },
        )
        g["count"] += 1
        if row.get("session_id") is not None:
            g["session_ids"].add(int(row["session_id"]))
        if row.get("rating") is not None:
            g["rated_count"] += 1
            g["rating_sum"] += float(row["rating"])
        ans = str(row.get("answer") or "")
        if "未检索到任何文档片段" in ans or "根据已知材料无法回答" in ans:
            g["no_ctx_count"] += 1
    out: list[list[Any]] = []
    for key, g in groups.items():
        avg = round(g["rating_sum"] / g["rated_count"], 2) if g["rated_count"] else "—"
        out.append(
            [
                key[0],
                key[1],
                key[2],
                key[3],
                key[4],
                "是" if key[5] else "否",
                key[6],
                key[7],
                g["count"],
                g["rated_count"],
                avg,
                g["no_ctx_count"],
                len(g["session_ids"]),
            ]
        )
    out.sort(key=lambda x: (-int(x[8]), str(x[0]), str(x[1]), str(x[2])))
    return out


def do_save_uploads(files, max_mb: float, embed_selected: str):
    ing = _lazy_ingest()
    _, msg, paths = ing.save_uploads(cfg, files, max_size_mb=float(max_mb))
    for p in paths:
        pp = Path(p)
        if pp.is_file():
            try:
                store.log_upload(p, pp.name, pp.stat().st_size)
            except Exception:
                pass
    return msg, kb_files_table_value(embed_selected or "")


def do_build_index(
    chunk_size: int,
    overlap: int,
    chunk_mode: str,
    embed_model: str,
    image_enrichment: bool,
    image_vision_model: str,
    image_tesseract_lang: str,
    image_ocr_skip_vlm: float,
    image_ocr_engine: str,
):
    """生成器：逐条刷新「状态」文本框，展示向量构建阶段与分批进度。"""
    ing = _lazy_ingest()
    eng = _lazy_engine()
    pipeline: dict[str, Any] = {}
    vm = str(image_vision_model or "").strip()
    if vm:
        pipeline["vision_model"] = vm
    lang = str(image_tesseract_lang or "").strip()
    if lang:
        pipeline["tesseract_lang"] = lang
    try:
        pipeline["ocr_skip_vlm_min_chars"] = max(0, min(500, int(round(float(image_ocr_skip_vlm)))))
    except (TypeError, ValueError):
        pipeline["ocr_skip_vlm_min_chars"] = int(cfg.ingest.get("ocr_skip_vlm_min_chars", 20))
    oe = str(image_ocr_engine or "").strip().lower()
    if oe not in ("tesseract", "paddleocr"):
        oe = str(cfg.ingest.get("ocr_engine") or "tesseract").strip().lower()
        if oe not in ("tesseract", "paddleocr"):
            oe = "tesseract"
    pipeline["ocr_engine"] = oe
    p = prefs_mod.load_prefs(cfg.data_dir)
    if vm:
        p["image_vision_model"] = vm
    if lang:
        p["image_tesseract_lang"] = lang
    p["image_ocr_skip_vlm_min_chars"] = pipeline["ocr_skip_vlm_min_chars"]
    p["image_ocr_engine"] = oe
    prefs_mod.save_prefs(cfg.data_dir, p)

    lines: list[str] = []
    t0 = time.perf_counter()
    final_ok = False
    eng.clear_all_index_caches()
    for step in ing.iter_build_index(
        cfg,
        int(chunk_size),
        int(overlap),
        chunk_mode=chunk_mode or "sentence",
        embed_model=embed_model or None,
        store=store,
        image_enrichment_override=bool(image_enrichment),
        image_pipeline_overrides=pipeline,
    ):
        lines.append(f"{step}  （累计 {time.perf_counter() - t0:.1f}s）")
        final_ok = step.startswith("完成：")
        yield "\n".join(lines)
    if final_ok:
        eng.clear_all_index_caches()
        eng.refresh_index_cache(cfg, embed_model=embed_model or None)


def do_chat_stream(
    message: str,
    history: list,
    system_prompt: str,
    top_n: int,
    top_k: int,
    use_rerank: bool,
    chunk_size: int,
    chunk_overlap: int,
    chunk_mode: str,
    llm_model: str,
    llm_num_ctx: float,
    embed_model: str,
    session_id: int,
):
    """流式输出三段：chatbot、参考来源、输入框。不要用 .then 接在流式事件后，Gradio 6 下易导致无响应。"""
    global LAST_QA_ID
    clear_pending = True

    def _out_msg():
        nonlocal clear_pending
        if clear_pending:
            clear_pending = False
            return ""
        return gr.update()

    sid = int(session_id) if session_id is not None else store.ensure_default_session()
    history = _as_messages(history)
    qtext = str(message).strip() if message is not None else ""
    if not qtext:
        yield _chatbot_value(history), _sources_panel_html([]), _DIAG_EMPTY_HTML, gr.update()
        return

    eng = _lazy_engine()
    em = (embed_model or "").strip() or None
    index = eng.get_index(cfg, embed_model=em)
    top_n = max(1, int(top_n))
    top_k = max(1, min(int(top_k), top_n))
    cm = (chunk_mode or "sentence").strip().lower()
    lm = (llm_model or "").strip() or None
    exclusion_info = build_excluded_file_set(cfg, index=index, prefer_index=True)
    excluded_files = list(exclusion_info.get("excluded_files") or [])
    excluded_doc_classes = list(exclusion_info.get("excluded_doc_classes") or [])
    include_zero_chunk = bool(exclusion_info.get("include_zero_chunk"))
    try:
        nctx = int(round(float(llm_num_ctx)))
    except (TypeError, ValueError):
        nctx = 16384
    nctx = max(2048, min(nctx, 262144))

    if index is None:
        err = "索引不存在：请先在「知识库」页上传文档并点击「构建向量索引」（或检查嵌入模型是否与构建时一致）。"
        new_hist_err = history + [
            {"role": "user", "content": qtext},
            {"role": "assistant", "content": err},
        ]
        yield _chatbot_value(new_hist_err), _sources_panel_html([]), _DIAG_EMPTY_HTML, _out_msg()
        return

    index_state = _build_index_status_snapshot()
    index_block_msg = _index_state_blocking_message(index_state)
    index_summary = _index_state_summary_text(index_state)

    new_hist = history + [
        {"role": "user", "content": qtext},
        {
            "role": "assistant",
            "content": f"**① 向量检索中…**（初筛 Top-{top_n}）",
        },
    ]
    yield _chatbot_value(new_hist), _sources_panel_html([]), _DIAG_EMPTY_HTML, _out_msg()

    try:
        prior_rows = store.fetch_qa_rows_for_session(sid)
        if len(prior_rows) == 0:
            store.update_session_meta(sid, title=qtext[:60])

        retrieval_result = eng.hybrid_retrieve(
            cfg,
            index,
            qtext,
            top_n,
            top_k,
            bool(use_rerank),
            excluded_files,
        )
        vector_diag = eng.nodes_to_source_dicts(retrieval_result.vector_nodes, "vector")
        keyword_diag = eng.nodes_to_source_dicts(retrieval_result.keyword_nodes, "keyword")
        merged_diag = eng.nodes_to_source_dicts(retrieval_result.merged_nodes, "hybrid")
        excluded_candidate_count = retrieval_result.excluded_candidate_count
        line1 = (
            f"**Hybrid retrieval complete** (vector {len(retrieval_result.vector_nodes)} / "
            f"keyword {len(retrieval_result.keyword_nodes)} / merged {len(retrieval_result.merged_nodes)} / "
            f"excluded {excluded_candidate_count} / Top-{top_n})"
        )
        new_hist[-1]["content"] = line1
        yield _chatbot_value(new_hist), _sources_panel_html([]), _retrieval_diag_html(
            _retrieval_diag_payload(
                query=qtext,
                vector_candidates=vector_diag,
                keyword_candidates=keyword_diag,
                merged_candidates=merged_diag,
                final_sources=[],
                top_n=top_n,
                top_k=top_k,
                use_rerank=bool(use_rerank),
                score_kind="hybrid",
                excluded_files=excluded_files,
                excluded_doc_classes=excluded_doc_classes,
                include_zero_chunk=include_zero_chunk,
                excluded_candidate_count=excluded_candidate_count,
            )
        ), _out_msg()

        nodes, kind = retrieval_result.final_nodes, retrieval_result.score_kind
        nodes_for_answer, context_truncated = eng.select_nodes_for_answer(
            cfg,
            qtext,
            nodes,
            llm_model=lm,
            llm_num_ctx=nctx,
        )
        sources = eng.nodes_to_source_dicts(nodes_for_answer, kind)
        src_html = _sources_panel_html(sources)
        diag_payload = _retrieval_diag_payload(
            query=qtext,
            vector_candidates=vector_diag,
            keyword_candidates=keyword_diag,
            merged_candidates=merged_diag,
            final_sources=sources,
            top_n=top_n,
            top_k=top_k,
            use_rerank=bool(use_rerank),
            score_kind=kind,
            excluded_files=excluded_files,
            excluded_doc_classes=excluded_doc_classes,
            include_zero_chunk=include_zero_chunk,
            excluded_candidate_count=excluded_candidate_count,
        )
        diag_payload["context_selected_count"] = len(nodes_for_answer)
        diag_payload["context_truncated"] = bool(context_truncated)
        diag_payload["vector_error"] = str(retrieval_result.vector_error or "")
        diag_payload["index_state"] = dict(index_state)
        diag_payload["index_state_summary"] = index_summary
        diag_html = _retrieval_diag_html(diag_payload)
        if index_block_msg:
            blocked = (
                "**当前索引状态异常，已停止生成回答。**\n\n"
                f"{index_block_msg}\n\n"
                f"当前状态：{index_summary}"
            )
            new_hist[-1]["content"] = blocked
            yield _chatbot_value(new_hist), src_html, diag_html, _out_msg()
            params = eng.build_params_snapshot(
                cfg,
                int(chunk_size),
                int(chunk_overlap),
                top_n,
                top_k,
                bool(use_rerank),
                chunk_mode=cm,
                llm_model=lm,
                embed_model=em,
                llm_num_ctx=nctx,
                excluded_files=excluded_files,
                excluded_doc_classes=excluded_doc_classes,
                include_zero_chunk=include_zero_chunk,
            )
            LAST_QA_ID = store.insert_qa(
                question=qtext,
                answer=blocked,
                sources=sources,
                params=params,
                diagnostics=diag_payload,
                session_id=sid,
            )
            yield _chatbot_value(new_hist), src_html, diag_html, _out_msg()
            return
        if not nodes_for_answer:
            no_ctx = (
                "**根据已知材料无法回答。**\n\n"
                "本轮**未检索到任何文档片段**，因此**没有**把「已知上下文」发给大模型，回答不应来自你的上传文件。\n\n"
                "**请排查：**①「知识库」是否已成功构建索引；② 嵌入模型是否与构建时一致且已在 Ollama 就绪；"
                "③ 问题与文档主题是否相关。"
            )
            new_hist[-1]["content"] = no_ctx
            yield _chatbot_value(new_hist), src_html, diag_html, _out_msg()
            params = eng.build_params_snapshot(
                cfg,
                int(chunk_size),
                int(chunk_overlap),
                top_n,
                top_k,
                bool(use_rerank),
                chunk_mode=cm,
                llm_model=lm,
                embed_model=em,
                llm_num_ctx=nctx,
                excluded_files=excluded_files,
                excluded_doc_classes=excluded_doc_classes,
                include_zero_chunk=include_zero_chunk,
            )
            LAST_QA_ID = store.insert_qa(
                question=qtext,
                answer=no_ctx,
                sources=sources,
                params=params,
                diagnostics=diag_payload,
                session_id=sid,
            )
            yield _chatbot_value(new_hist), src_html, diag_html, _out_msg()
            return

        kind_zh = "重排" if kind == "rerank" else "向量截断"
        if bool(use_rerank) and retrieval_result.merged_nodes:
            progress_head = (
                f"{line1}\n\n"
                f"**② 重排完成**（{kind_zh}，送入上下文 {len(nodes_for_answer)} 条）\n\n"
                f"**③ 正在生成回答…**"
            )
        else:
            progress_head = (
                f"{line1}\n\n"
                f"**③ 正在生成回答…**（未启用重排，送入上下文 {len(nodes_for_answer)} 条）"
            )
        sep = "\n\n---\n\n"
        new_hist[-1]["content"] = progress_head + sep
        yield _chatbot_value(new_hist), src_html, diag_html, _out_msg()
        partial = ""
        for token in eng.stream_answer(
            cfg, qtext, system_prompt, nodes_for_answer, llm_model=lm, llm_num_ctx=nctx
        ):
            partial += token
            new_hist[-1]["content"] = progress_head + sep + partial
            yield _chatbot_value(new_hist), src_html, diag_html, _out_msg()
        params = eng.build_params_snapshot(
            cfg,
            int(chunk_size),
            int(chunk_overlap),
            top_n,
            top_k,
            bool(use_rerank),
            chunk_mode=cm,
            llm_model=lm,
            embed_model=em,
            llm_num_ctx=nctx,
            excluded_files=excluded_files,
            excluded_doc_classes=excluded_doc_classes,
            include_zero_chunk=include_zero_chunk,
        )
        # Store clean answer in qa_log.answer; the rich pre-answer scaffolding
        # (Hybrid retrieval / rerank / generation markers) goes into
        # diag_payload['ui_progress'] and is reconstructed by
        # session_history_for_chatbot when the user switches back to this
        # session, so the rich bubble still renders for replay.
        diag_payload['ui_progress'] = progress_head
        LAST_QA_ID = store.insert_qa(
            question=qtext,
            answer=partial.strip(),
            sources=sources,
            params=params,
            diagnostics=diag_payload,
            session_id=sid,
        )
        yield _chatbot_value(new_hist), src_html, diag_html, _out_msg()
    except Exception:
        tb = traceback.format_exc()
        err = f"生成失败:\n```\n{tb}\n```"
        try:
            params = eng.build_params_snapshot(
                cfg,
                int(chunk_size),
                int(chunk_overlap),
                top_n,
                top_k,
                bool(use_rerank),
                chunk_mode=cm,
                llm_model=lm,
                embed_model=em,
                llm_num_ctx=nctx,
                excluded_files=excluded_files,
                excluded_doc_classes=excluded_doc_classes,
                include_zero_chunk=include_zero_chunk,
            )
            LAST_QA_ID = store.insert_qa(
                question=qtext,
                answer=err,
                sources=[],
                params=params,
                diagnostics={"index_state": dict(index_state), "error": "chat_exception"},
                session_id=sid,
            )
        except Exception:
            LAST_QA_ID = None
        new_hist[-1]["content"] = err
        yield _chatbot_value(new_hist), _sources_panel_html([]), _DIAG_EMPTY_HTML, _out_msg()


def do_rate_last(rating: float | None, note: str, session_id: int | None):
    sid = int(session_id) if session_id is not None else store.ensure_default_session()
    row_id = store.get_latest_qa_id_for_session(sid)
    if row_id is None:
        return "没有可打分的对话（先完成一次问答）。"
    r = int(rating) if rating is not None else None
    store.update_qa_rating(row_id, r, note or None)
    return f"已保存评分：id={row_id}"


def do_export(fmt: str):
    out_dir = cfg.data_dir / "exports"
    out_dir.mkdir(parents=True, exist_ok=True)
    if fmt == "json":
        dest = out_dir / "qa_export.json"
        store.export_json(dest, include_eval=False)
    else:
        dest = out_dir / "qa_export.csv"
        store.export_csv(dest, include_eval=False)
    return str(dest.resolve())


def do_chunk_diagnostics(embed_selected: str | None = None):
    ing = _lazy_ingest()
    eng = _lazy_engine()
    em = (embed_selected or "").strip() or None
    index = eng.get_index(cfg, embed_model=em)
    # Diagnostics should inspect the active disk-backed collection so the panel matches current index state.
    diag = ing.chunk_diagnostics(cfg, index=index, prefer_index=False)
    diag["active_chroma_label"] = active_chroma_label(cfg, store=store)
    rows = [
        [
            str(x.get("file_name") or ""),
            str(x.get("index_status_label") or "—"),
            int(x.get("chunk_count") or 0),
            x.get("avg_chars") or 0,
            int(x.get("max_chars") or 0),
            int(x.get("empty_chunks") or 0),
            int(x.get("image_hint_chunks") or 0),
            int(x.get("ocr_hint_chunks") or 0),
            int(x.get("vision_hint_chunks") or 0),
            str(x.get("pdf_doc_class_label") or "—"),
            str(x.get("pdf_doc_class_reason_label") or "—"),
            f"{int(x.get('pdf_text_pages') or 0)}/{int(x.get('pdf_page_count') or 0)}",
            int(x.get("pdf_suspicious_page_ratio_pct") or 0),
        ]
        for x in diag.get("files") or []
    ]
    if not rows:
        rows = [["（当前无切片统计）", "—", 0, 0, 0, 0, 0, 0, 0, "—", "—", "0/0", 0]]
    return _chunk_diag_summary_html(diag), rows


def do_index_ops_diagnostics():
    diag = index_ops_diagnostics(cfg, store=store, keep_recent=3)
    return index_ops_summary_html(diag), index_ops_rows(diag)


def do_index_cleanup_dry_run():
    msg, diag = cleanup_old_index_versions(cfg, store=store, keep_recent=3, dry_run=True)
    return msg, index_ops_summary_html(diag), index_ops_rows(diag)


def do_index_cleanup_apply(confirmed: bool = False):
    if not confirmed:
        diag = index_ops_diagnostics(cfg, store=store, keep_recent=3)
        return (
            "\u672a\u786e\u8ba4\uff1a\u8bf7\u52fe\u9009\u201c\u6211\u5df2\u786e\u8ba4\u8981\u6e05\u7406\u4ee5\u4e0a\u76ee\u5f55\uff08\u4e0d\u53ef\u6062\u590d\uff09\u201d\u540e\u518d\u70b9 Apply\u3002",
            index_ops_summary_html(diag),
            index_ops_rows(diag),
        )
    msg, diag = cleanup_old_index_versions(cfg, store=store, keep_recent=3, dry_run=False)
    return msg, index_ops_summary_html(diag), index_ops_rows(diag)


def do_experiment_compare():
    rows = _experiment_compare_rows(store.fetch_all_qa(include_eval=False))
    if not rows:
        rows = [["（暂无问答记录）", "", "", 0, 0, "否", 0, 0, 0, 0, "—", 0, 0]]
    return rows, _eval_run_compare_rows(), _eval_run_diff_rows(), _eval_baseline_compare_rows()


def _eval_run_compare_rows() -> list[list[Any]]:
    rows: list[list[Any]] = []
    for run in store.list_eval_runs(limit=100):
        summary = run.get("summary") or {}
        params = run.get("params") or {}
        attrs = summary.get("attribution_counts") or {}
        top_attr = ", ".join(
            f"{k}:{v}" for k, v in sorted(attrs.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))[:3]
        ) or "-"
        rows.append(
            [
                int(run.get("id") or 0),
                str(run.get("dataset_name") or ""),
                str(run.get("name") or ""),
                str(params.get("run_fingerprint") or ""),
                str(params.get("retrieval_mode") or ""),
                bool(params.get("vector_enabled", True)),
                str(params.get("llm_model") or ""),
                str(params.get("embed_model") or ""),
                int(params.get("top_n") or 0),
                int(params.get("top_k") or 0),
                summary.get("candidate_hit_rate") or "-",
                summary.get("context_hit_rate") or "-",
                summary.get("chunk_hit_rate") or "-",
                summary.get("answer_hit_rate") or "-",
                int(summary.get("ok_count") or 0),
                top_attr,
            ]
        )
    return rows or [[0, "（暂无评测运行）", "", "", "", False, "", "", 0, 0, "-", "-", "-", "-", 0, "-"]]


def _pct_text_to_float(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text or text in {"-", "—"}:
        return None
    try:
        return float(text.rstrip("%"))
    except ValueError:
        return None


def _format_metric_delta(current: Any, previous: Any) -> str:
    cur = _pct_text_to_float(current)
    prev = _pct_text_to_float(previous)
    if cur is None or prev is None:
        return "—"
    delta = cur - prev
    return f"{delta:+.1f}pp"


def _nested_eval_value(data: dict[str, Any], dotted_key: str) -> Any:
    cur: Any = data
    for part in dotted_key.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _short_eval_value(value: Any) -> str:
    if value is None or value == "":
        return "-"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)[:260]
    text = str(value)
    return text if len(text) <= 260 else text[:257] + "..."


def _top_eval_counts_text(counts: dict[str, Any], limit: int = 4) -> str:
    if not counts:
        return "-"
    return ", ".join(
        f"{k}:{v}" for k, v in sorted(counts.items(), key=lambda kv: (-int(kv[1] or 0), str(kv[0])))[:limit]
    ) or "-"


def _eval_run_diff_rows() -> list[list[Any]]:
    runs = store.list_eval_runs(limit=2)
    if len(runs) < 2:
        return [["（需要至少两个评测运行）", "-", "-", "-", "-"]]

    current, previous = runs[0], runs[1]
    cur_params = current.get("params") or {}
    prev_params = previous.get("params") or {}
    cur_summary = current.get("summary") or {}
    prev_summary = previous.get("summary") or {}

    rows: list[list[Any]] = [["RUN", f"#{current.get('id')} {current.get('name') or ''}", f"#{previous.get('id')} {previous.get('name') or ''}", "-", "按最近两次评测运行对比"]]

    metric_specs = [
        ("candidate_hit_rate", "候选命中率"),
        ("context_hit_rate", "上下文命中率"),
        ("chunk_hit_rate", "片段命中率"),
        ("answer_hit_rate", "答案命中率"),
        ("abstain_accuracy", "拒答准确率"),
    ]
    for key, label in metric_specs:
        cur_val = cur_summary.get(key) or "-"
        prev_val = prev_summary.get(key) or "-"
        rows.append([label, cur_val, prev_val, _format_metric_delta(cur_val, prev_val), "核心指标变化"])
    rows.append([
        "OK 数",
        int(cur_summary.get("ok_count") or 0),
        int(prev_summary.get("ok_count") or 0),
        int(cur_summary.get("ok_count") or 0) - int(prev_summary.get("ok_count") or 0),
        "综合判定通过样本变化",
    ])

    changed_params = 0
    param_specs = [
        ("run_fingerprint", "Run fingerprint", "整体实验指纹变化"),
        ("retrieval_mode", "检索模式", "检索管线变化会直接影响召回"),
        ("generation_mode", "生成模式", "retrieval-only 仅跳过 LLM 生成"),
        ("vector_enabled", "向量召回", "由 Retrieval mode 显式控制"),
        ("keyword_enabled", "关键词召回", "由 Retrieval mode 显式控制"),
        ("top_n", "Top-N", "候选池大小变化"),
        ("top_k", "Top-K", "进入上下文数量变化"),
        ("use_rerank", "重排", "重排启停可能改变最终片段"),
        ("query_anchoring_enabled", "查询锚定", "金融文件名/问题锚定变化"),
        ("llm_model", "LLM", "生成模型变化影响答案与拒答"),
        ("embed_model", "Embedding", "向量模型变化影响候选召回"),
        ("rerank_model", "Rerank 模型", "重排模型变化影响 Top-K"),
        ("chunk_mode", "切分策略", "切片方式变化影响片段命中"),
        ("chunk_size", "Chunk", "切片大小变化影响上下文粒度"),
        ("chunk_overlap", "Overlap", "重叠变化影响跨片段信息保留"),
    ]
    for key, label, note in param_specs:
        cur_val = cur_params.get(key)
        prev_val = prev_params.get(key)
        if cur_val != prev_val:
            changed_params += 1
            rows.append([label, _short_eval_value(cur_val), _short_eval_value(prev_val), "已变化", note])

    fp_specs = [
        ("run_fingerprint_payload.index_build_id", "索引 Build ID", "索引重建会影响候选空间"),
        ("run_fingerprint_payload.index_active_dir", "活跃索引目录", "活跃 Chroma 目录不同需确认 manifest 指向"),
        ("run_fingerprint_payload.index_chunk_count", "索引切片数", "索引切片数量变化会影响召回覆盖"),
        ("run_fingerprint_payload.index_file_count", "索引文件数", "索引文件覆盖变化会影响目标文件命中"),
    ]
    for key, label, note in fp_specs:
        cur_val = _nested_eval_value(cur_params, key)
        prev_val = _nested_eval_value(prev_params, key)
        if cur_val != prev_val:
            changed_params += 1
            rows.append([label, _short_eval_value(cur_val), _short_eval_value(prev_val), "已变化", note])

    cur_quality = ((cur_summary.get("dataset_quality") or cur_params.get("dataset_quality") or {}).get("issue_counts") or {})
    prev_quality = ((prev_summary.get("dataset_quality") or prev_params.get("dataset_quality") or {}).get("issue_counts") or {})
    if cur_quality != prev_quality:
        rows.append([
            "评测集质量问题",
            _top_eval_counts_text(cur_quality),
            _top_eval_counts_text(prev_quality),
            "已变化",
            "样本治理变化会改变可评测样本口径",
        ])

    cur_attrs = cur_summary.get("attribution_counts") or {}
    prev_attrs = prev_summary.get("attribution_counts") or {}
    if cur_attrs != prev_attrs:
        rows.append([
            "Top 归因",
            _top_eval_counts_text(cur_attrs),
            _top_eval_counts_text(prev_attrs),
            "已变化",
            "用于定位是数据集、召回、重排、答案还是拒答策略导致",
        ])

    if changed_params == 0:
        rows.append(["参数/索引", "未发现关键变化", "未发现关键变化", "一致", "若指标变化，优先查看本地模型波动、样本导入或运行时异常"])
    return rows


def _eval_baseline_path() -> Path:
    return cfg.data_dir / "eval_baselines.json"


_EVAL_LOAD_ERRORS: dict[str, str] = {}

def _eval_load_errors_text() -> str:
    lines = []
    for k, v in _EVAL_LOAD_ERRORS.items():
        lines.append("\u26a0 " + str(Path(k).name) + ": " + str(v))
    return "\n".join(lines)

def _load_json_or_backup(path: Path) -> Any:
    """Read JSON; on parse error, copy bytes to <name>.corrupt-<ts> and record a warning.
    Returns the parsed JSON on success, None on missing file or parse error.
    The caller is expected to fall back to a safe default (empty dict/list) when
    None is returned. The backup lets the user inspect the broken file afterwards
    while unblocking the app from silently losing state.
    """
    if not path.exists():
        _EVAL_LOAD_ERRORS.pop(str(path), None)
        return None
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        _EVAL_LOAD_ERRORS.pop(str(path), None)
        return result
    except Exception as exc:
        try:
            backup = path.with_name(path.name + ".corrupt-" + str(int(time.time())))
            backup.write_bytes(path.read_bytes())
            msg = "JSON \u635f\u574f\uff08" + type(exc).__name__ + ": " + str(exc) + "\uff09\uff0c\u5df2\u5907\u4efd\u5230 " + backup.name + "\uff0c\u5f53\u524d\u4e3a\u7a7a\u72b6\u6001"
        except OSError as ioexc:
            msg = "JSON \u635f\u574f\uff08" + type(exc).__name__ + ": " + str(exc) + "\uff09\uff0c\u5907\u4efd\u5931\u8d25\uff08" + str(ioexc) + "\uff09\uff0c\u5f53\u524d\u4e3a\u7a7a\u72b6\u6001"
        _EVAL_LOAD_ERRORS[str(path)] = msg
        logging.getLogger("rag_lite").warning("eval config %s corrupted: %s", path, exc)
        return None


def _load_eval_baselines() -> dict[str, int]:
    path = _eval_baseline_path()
    payload = _load_json_or_backup(path)
    if not isinstance(payload, (dict, list)):
        return {}
    data = payload.get("baselines") if isinstance(payload, dict) else payload
    out: dict[str, int] = {}
    if isinstance(data, dict):
        for k, v in data.items():
            try:
                out[str(int(k))] = int(v)
            except (TypeError, ValueError):
                continue
    return out


def _atomic_write_json(path, payload: dict) -> None:
    """Atomically write a JSON file via temp + os.replace.
    Prevents partial/corrupt files if the process is killed mid-write.
    Raises on any I/O error after removing the temp file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(str(tmp_path), str(path))
    except BaseException:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise


def _save_eval_baselines(data: dict[str, int]) -> None:
    _atomic_write_json(_eval_baseline_path(), {"baselines": data})


def _eval_run_by_id(run_id: int | None) -> dict[str, Any] | None:
    if not run_id:
        return None
    return next((r for r in store.list_eval_runs(limit=500) if int(r.get("id") or 0) == int(run_id)), None)


def _latest_eval_run(dataset_id: int | None = None) -> dict[str, Any] | None:
    return next(
        (
            r
            for r in store.list_eval_runs(limit=500)
            if dataset_id is None or int(r.get("dataset_id") or 0) == int(dataset_id)
        ),
        None,
    )


def _eval_baseline_compare_rows(dataset_id: int | None = None) -> list[list[Any]]:
    latest = _latest_eval_run(dataset_id)
    if not latest:
        return [["（暂无评测运行）", "-", "-", "-", "-"]]
    did = int(latest.get("dataset_id") or 0)
    baseline_id = _load_eval_baselines().get(str(did))
    baseline = _eval_run_by_id(baseline_id)
    if not baseline:
        return [[f"Dataset #{did}", f"latest RUN#{latest.get('id')}", "未设置 baseline", "-", "点击“设为Baseline”后建立稳定对比口径"]]
    cur_summary = latest.get("summary") or {}
    base_summary = baseline.get("summary") or {}
    rows = [[
        "RUN",
        f"#{latest.get('id')} {latest.get('name') or ''}",
        f"#{baseline.get('id')} {baseline.get('name') or ''}",
        "-",
        "当前最新 RUN vs baseline",
    ]]
    for key, label in [
        ("candidate_hit_rate", "候选命中率"),
        ("context_hit_rate", "上下文命中率"),
        ("chunk_hit_rate", "片段命中率"),
        ("answer_hit_rate", "答案命中率"),
        ("abstain_accuracy", "拒答准确率"),
    ]:
        cur_val = cur_summary.get(key) or "-"
        base_val = base_summary.get(key) or "-"
        rows.append([label, cur_val, base_val, _format_metric_delta(cur_val, base_val), "相对 baseline 变化"])
    return rows


def _eval_dataset_blocking_issues(dataset_id: int | None) -> tuple[bool, list[str]]:
    """Return (is_clean, blocker_codes) for a dataset.
    Blocking issues are ones that would make a baseline run a poor reference:
    the cases themselves are not trustworthy (unresolved filenames, ambiguous
    matches, empty questions, or no expected answer/keywords for non-abstain cases).
    Quality issues that are useful to know about (e.g. NO_TAG, NO_EXPECTED_CHUNK)
    are intentionally non-blocking: a baseline can still be set on a dataset that
    is missing tags or expected-chunk strings.
    """
    if not dataset_id:
        return True, []
    cases = store.fetch_eval_cases(int(dataset_id))
    summary = dataset_quality_summary(cases, resolve_expected_file_details=_resolve_expected_eval_file_details)
    counts = dict(summary.get("issue_counts") or {})
    blocking = (
        "EXPECTED_FILE_UNRESOLVED",
        "EXPECTED_FILE_AMBIGUOUS",
        "EMPTY_QUESTION",
        "NO_ANSWER_OR_KEYWORDS",
    )
    blockers = [code for code in blocking if int(counts.get(code) or 0) > 0]
    return (not blockers), blockers


def do_set_latest_eval_baseline(dataset_choice: str | None, force: bool = False):
    dataset_id = _parse_eval_dataset_id(dataset_choice)
    latest = _latest_eval_run(dataset_id)
    if not latest:
        return "\u5f53\u524d\u8bc4\u6d4b\u96c6\u6682\u65e0 RUN\uff0c\u65e0\u6cd5\u8bbe\u7f6e baseline\u3002", _eval_baseline_compare_rows(dataset_id)
    latest_dataset_id = int(latest.get("dataset_id") or 0)
    if not force:
        is_clean, blockers = _eval_dataset_blocking_issues(latest_dataset_id)
        if not is_clean:
            names = "\u3001".join(blockers)
            return (
                "\u26a0 \u6570\u636e\u96c6\u5b58\u5728\u963b\u585e\u9879\uff08"
                + names
                + "\uff09\uff0c\u8bf7\u5148\u5728\u300c\u6570\u636e\u96c6\u6cbb\u7406\u300d\u9762\u677f\u5904\u7406\u540e\u518d\u8bbe baseline\u3002",
                _eval_baseline_compare_rows(latest_dataset_id),
            )
    run_id = int(latest.get("id") or 0)
    baselines = _load_eval_baselines()
    baselines[str(latest_dataset_id)] = run_id
    _save_eval_baselines(baselines)
    suffix = " (\u5f3a\u5236\u8986\u76d6)" if force else ""
    return "\u5df2\u5c06 RUN#" + str(run_id) + " \u8bbe\u4e3a dataset #" + str(latest_dataset_id) + " \u7684 baseline" + suffix + "\u3002", _eval_baseline_compare_rows(latest_dataset_id)

def do_ollama_health_check():
    return ollama_health_rows(ollama_health_snapshot(str(cfg.ollama.get("base_url") or "")))


def _latest_eval_run_id(dataset_id: int | None = None) -> int | None:
    for run in store.list_eval_runs(limit=100):
        if dataset_id is None or int(run.get("dataset_id") or 0) == int(dataset_id):
            rid = int(run.get("id") or 0)
            return rid if rid > 0 else None
    return None


def do_export_latest_eval_report(dataset_choice: str | None):
    dataset_id = _parse_eval_dataset_id(dataset_choice)
    run_id = _latest_eval_run_id(dataset_id)
    if run_id is None:
        return None
    return str(write_eval_run_report(store, run_id, cfg.data_dir / "exports"))


def do_export_latest_eval_case_compare(dataset_choice: str | None):
    """导出「原题字段 + 评测结果」逐题对比表（xlsx，附 csv）。"""
    dataset_id = _parse_eval_dataset_id(dataset_choice)
    run_id = _latest_eval_run_id(dataset_id)
    if run_id is None:
        return None
    return str(write_eval_case_compare(store, run_id, cfg.data_dir / "exports"))


def do_run_retrieval_regression(dataset_choice: str | None):
    # 真实召回口径：retrieval-only + anchoring OFF；切片取自索引 manifest
    out = do_run_eval_dataset(
        dataset_choice,
        f"retrieval regression {time.strftime('%Y-%m-%d %H:%M:%S')}",
        str(cfg.prompt.get("system_default", "")).strip(),
        3,
        1,
        False,
        None,
        None,
        None,
        str(cfg.ollama.get("llm_model") or ""),
        2048,
        str(cfg.ollama.get("embed_model") or ""),
        "retrieval_only",
        "hybrid",
        False,
    )
    try:
        status, result_rows, summary_rows, recent_rows, param_rows, error_rows, tag_rows, file_rows, chunk_diag_rows, funnel_rows = out
        summary_map = {str(k): v for k, v in summary_rows}
        candidate_rate = str(summary_map.get("Candidate hit rate") or summary_map.get("候选命中率") or "-")
        context_rate = str(summary_map.get("Context hit rate") or summary_map.get("最终上下文命中率") or "-")
        status = (
            str(status)
            + "\n\n[Regression gate] retrieval-only baseline complete. "
            + f"candidate_hit_rate={candidate_rate}; context_hit_rate={context_rate}. "
            + "若低于上一条稳定基线，请优先查看归因表和失败样本详情。"
        )
        return status, result_rows, summary_rows, recent_rows, param_rows, error_rows, tag_rows, file_rows, chunk_diag_rows, funnel_rows
    except Exception:
        return out


def do_batch_replay(
    raw_questions: str,
    batch_title: str,
    system_prompt: str,
    top_n: int,
    top_k: int,
    use_rerank: bool,
    chunk_size: int,
    chunk_overlap: int,
    chunk_mode: str,
    llm_model: str,
    llm_num_ctx: float,
    embed_model: str,
    generation_mode: str = "llm",
    retrieval_mode: str = "hybrid",
):
    eng = _lazy_engine()
    questions = [line.strip() for line in str(raw_questions or "").splitlines() if line.strip()]
    if not questions:
        return "没有可回放的问题。请按“每行一个问题”输入。", [["（暂无结果）", "—", 0, 0, "—", "—"]], gr.update()

    em = (embed_model or "").strip() or None
    generation_mode = normalize_generation_mode(generation_mode, llm_model)
    retrieval_only = generation_mode == "retrieval_only"
    lm = effective_llm_model(llm_model, generation_mode)
    retrieval_mode = normalize_retrieval_mode(retrieval_mode)
    vector_enabled, keyword_enabled = retrieval_flags(retrieval_mode)
    try:
        nctx = int(round(float(llm_num_ctx)))
    except (TypeError, ValueError):
        nctx = 16384
    nctx = max(2048, min(nctx, 262144))
    top_n = max(1, int(top_n))
    top_k = max(1, int(top_k))
    cm = (chunk_mode or "sentence").strip().lower()

    index = eng.get_index(cfg, embed_model=em)
    if index is None:
        return "当前没有可用索引。请先在“知识库”页完成索引构建。", [["（暂无结果）", "—", 0, 0, "—", "—"]], gr.update()
    index_state = _build_index_status_snapshot()
    index_block_msg = _index_state_blocking_message(index_state)
    if index_block_msg:
        return index_block_msg, [["（暂无结果）", "—", 0, 0, "—", "—"]], gr.update()
    exclusion_info = build_excluded_file_set(cfg, index=index, prefer_index=True)
    excluded_files = list(exclusion_info.get("excluded_files") or [])
    excluded_doc_classes = list(exclusion_info.get("excluded_doc_classes") or [])
    include_zero_chunk = bool(exclusion_info.get("include_zero_chunk"))

    title = str(batch_title or "").strip() or f"批量回放 {time.strftime('%Y-%m-%d %H:%M:%S')}"
    sid = store.create_session(title)
    lines: list[str] = [f"已创建批量回放会话：#{sid} · {title}"]
    result_rows: list[list[Any]] = []

    for i, qtext in enumerate(questions, start=1):
        lines.append(f"[{i}/{len(questions)}] {qtext}")
        retrieval_result = eng.hybrid_retrieve(
            cfg,
            index,
            qtext,
            top_n,
            top_k,
            bool(use_rerank),
            excluded_files,
            vector_enabled=vector_enabled,
            keyword_enabled=keyword_enabled,
        )
        vector_diag = eng.nodes_to_source_dicts(retrieval_result.vector_nodes, "vector")
        keyword_diag = eng.nodes_to_source_dicts(retrieval_result.keyword_nodes, "keyword")
        merged_diag = eng.nodes_to_source_dicts(retrieval_result.merged_nodes, "hybrid")
        excluded_candidate_count = retrieval_result.excluded_candidate_count
        nodes, kind = retrieval_result.final_nodes, retrieval_result.score_kind
        nodes_for_answer, context_truncated = eng.select_nodes_for_answer(
            cfg,
            qtext,
            nodes,
            llm_model=lm,
            llm_num_ctx=nctx,
        )
        sources = eng.nodes_to_source_dicts(nodes_for_answer, kind)
        diag_payload = _retrieval_diag_payload(
            query=qtext,
            vector_candidates=vector_diag,
            keyword_candidates=keyword_diag,
            merged_candidates=merged_diag,
            final_sources=sources,
            top_n=top_n,
            top_k=top_k,
            use_rerank=bool(use_rerank),
            score_kind=kind,
            excluded_files=excluded_files,
            excluded_doc_classes=excluded_doc_classes,
            include_zero_chunk=include_zero_chunk,
            excluded_candidate_count=excluded_candidate_count,
            retrieval_mode=retrieval_mode,
        )
        diag_payload["context_selected_count"] = len(nodes_for_answer)
        diag_payload["context_truncated"] = bool(context_truncated)
        diag_payload["vector_error"] = str(retrieval_result.vector_error or "")
        diag_payload["index_state"] = dict(index_state)
        diag_payload["index_state_summary"] = _index_state_summary_text(index_state)
        params = eng.build_params_snapshot(
            cfg,
            int(chunk_size),
            int(chunk_overlap),
            top_n,
            top_k,
            bool(use_rerank),
            chunk_mode=cm,
            llm_model=lm,
            embed_model=em,
            llm_num_ctx=nctx,
            excluded_files=excluded_files,
            excluded_doc_classes=excluded_doc_classes,
            include_zero_chunk=include_zero_chunk,
            vector_enabled=vector_enabled,
            keyword_enabled=keyword_enabled,
            generation_mode=generation_mode,
            prefer_index_chunk_params=True,
            retrieval_degraded=bool(retrieval_result.retrieval_degraded),
            vector_error=str(retrieval_result.vector_error or ""),
        )
        diag_payload["retrieval_degraded"] = bool(retrieval_result.retrieval_degraded)
        diag_payload["requested_retrieval_mode"] = str(
            getattr(retrieval_result, "requested_retrieval_mode", retrieval_mode) or retrieval_mode
        )

        if not nodes_for_answer:
            answer = (
                "**根据已知材料无法回答。**\n\n"
                "本轮未检索到任何文档片段，因此没有把上下文发给大模型。"
            )
        elif retrieval_only:
            answer = eval_judge.ANSWER_RETRIEVAL_ONLY
        else:
            parts: list[str] = []
            try:
                for token in eng.stream_answer(
                    cfg, qtext, system_prompt, nodes_for_answer, llm_model=lm, llm_num_ctx=nctx
                ):
                    parts.append(token)
                answer = "".join(parts).strip() or "（模型未返回正文）"
            except Exception as exc:
                answer = f"生成失败：{type(exc).__name__}: {exc}"
                diag_payload["generation_error"] = answer
        qa_id = store.insert_qa(
            question=qtext,
            answer=answer,
            sources=sources,
            params=params,
            diagnostics=diag_payload,
            session_id=sid,
        )
        result_rows.append(
            [
                qtext,
                "命中" if nodes_for_answer else "未命中",
                len(retrieval_result.merged_nodes),
                len(nodes_for_answer),
                str(sources[0].get("file_name") or "—") if sources else "—",
                qa_id,
            ]
        )
        lines.append(
            f"    Hybrid candidates {len(retrieval_result.merged_nodes)} "
            f"(vector {len(retrieval_result.vector_nodes)} / keyword {len(retrieval_result.keyword_nodes)}), "
            f"context {len(nodes_for_answer)}, QA#{qa_id}."
        )

    if not result_rows:
        result_rows = [["（暂无结果）", "—", 0, 0, "—", "—"]]
    return "\n".join(lines), result_rows, gr.update(value=_sessions_table_value())

def _eval_dataset_choices() -> tuple[list[str], str | None]:
    rows = store.list_eval_datasets()
    choices = [f"#{int(r['id'])} | {str(r['name'])} | {int(r.get('case_count') or 0)}题" for r in rows]
    value = choices[0] if choices else None
    return choices, value


def _parse_eval_dataset_id(choice: str | None) -> int | None:
    if not choice:
        return None
    try:
        return int(str(choice).split("|", 1)[0].strip().lstrip("#"))
    except Exception:
        return None


def _split_eval_list_field(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    text = str(raw).strip()
    if not text:
        return []
    text = text.replace("\n", "|").replace("，", ",").replace("；", ";")
    out: list[str] = []
    for seg in text.replace(";", "|").replace(",", "|").split("|"):
        s = seg.strip()
        if s:
            out.append(s)
    return out


def _parse_eval_bool(raw: Any) -> bool:
    s = str(raw or "").strip().lower()
    return s in ("1", "true", "yes", "y", "是")


def _pick_eval_field(row: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in row:
            return row.get(name)
    lowered = {str(k).strip().lower(): v for k, v in row.items()}
    for name in names:
        key = str(name).strip().lower()
        if key in lowered:
            return lowered[key]
    return None


def _parse_eval_cases_from_file(file_obj: Any) -> list[dict[str, Any]]:
    if not file_obj:
        return []
    # Path.name 只有文件名会丢目录；优先用完整路径
    if isinstance(file_obj, Path):
        path = file_obj
    else:
        raw = getattr(file_obj, "name", file_obj)
        path = Path(str(raw))
        # Gradio 临时文件通常是绝对路径；若仅有文件名则再试原对象字符串
        if not path.is_file() and not path.is_absolute():
            alt = Path(str(file_obj))
            if alt.is_file():
                path = alt
    if not path.is_file():
        return []
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            rows = payload.get("cases") or []
        elif isinstance(payload, list):
            rows = payload
        else:
            rows = []
    elif suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    elif suffix in (".xlsx", ".xls"):
        import pandas as pd

        xls = pd.ExcelFile(path)
        rows = []
        for sheet in xls.sheet_names:
            df = pd.read_excel(path, sheet_name=sheet)
            if df is None or df.empty or len(df.columns) == 0:
                continue
            rows.extend(df.fillna("").to_dict(orient="records"))
    else:
        raise ValueError("仅支持 .json / .csv / .xlsx / .xls")

    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        question = str(
            _pick_eval_field(
                row,
                "question",
                "问题",
                "问题 (Question)",
            )
            or ""
        ).strip()
        if not question:
            continue
        qtype = str(_pick_eval_field(row, "问题类型", "question_type", "type") or "").strip()
        target_chunk = str(
            _pick_eval_field(
                row,
                "应命中片段",
                "应命中片段 (Target_Chunk_Content)",
                "target_chunk_content",
            )
            or ""
        ).strip()
        note = str(_pick_eval_field(row, "note", "备注") or "").strip()
        out.append(
            {
                "question": question,
                "expected_answer": str(
                    _pick_eval_field(
                        row,
                        "expected_answer",
                        "标准答案",
                        "标准答案 (Golden_Answer)",
                        "golden_answer",
                    )
                    or ""
                ).strip(),
                "expected_file_names": _split_eval_list_field(
                    _pick_eval_field(
                        row,
                        "expected_file_names",
                        "expected_files",
                        "应命中文档",
                        "应命中文档 (Target_Document)",
                        "target_document",
                    )
                ),
                "expected_answer_keywords": _split_eval_list_field(
                    _pick_eval_field(
                        row,
                        "expected_answer_keywords",
                        "answer_keywords",
                    )
                ),
                "expected_chunk_content": target_chunk,
                "allow_abstain": _parse_eval_bool(
                    _pick_eval_field(
                        row,
                        "allow_abstain",
                        "是否允许拒答",
                        "是否允许拒答 (Is_Reject)",
                        "is_reject",
                    )
                ),
                "tags": ([qtype] if qtype else []) + _split_eval_list_field(_pick_eval_field(row, "tags", "标签")),
                "note": note,
            }
        )
    return out


def _eval_cases_preview_rows(cases: list[dict[str, Any]]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for idx, case in enumerate(cases[:200], start=1):
        rows.append(
            [
                idx,
                case.get("question") or "",
                case.get("expected_answer") or "",
                " | ".join(case.get("expected_file_names") or []),
                case.get("expected_chunk_content") or "",
                " | ".join(case.get("expected_answer_keywords") or []),
                "是" if bool(case.get("allow_abstain")) else "否",
                " | ".join(case.get("tags") or []),
            ]
        )
    if not rows:
        rows = [[0, "（暂无样本）", "", "", "", "", ""]]
    return rows


def _eval_dataset_summary_markdown(dataset_id: int | None) -> str:
    if dataset_id is None:
        return "未选择评测集。"
    dataset = store.get_eval_dataset(dataset_id)
    if dataset is None:
        return "未找到评测集。"
    cases = store.fetch_eval_cases(dataset_id)
    quality = dataset_quality_summary(cases, resolve_expected_file_details=_resolve_expected_eval_file_details)
    issue_counts = quality.get("issue_counts") or {}
    top_issues = ", ".join(
        f"{k}:{v}" for k, v in sorted(issue_counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))[:8]
    ) or "none"
    return (
        f"评测集：`{dataset.get('name')}`\n\n"
        f"样本数：`{len(cases)}`；期望文件：`{quality.get('file_expected_count')}`；"
        f"期望片段：`{quality.get('chunk_expected_count')}`；答案关键词：`{quality.get('keyword_expected_count')}`；"
        f"拒答样本：`{quality.get('abstain_count')}`。\n\n"
        f"Dataset quality issues: `{top_issues}`."
    )


def _normalize_eval_text(text: Any) -> str:
    s = str(text or "").strip().lower()
    return "".join(ch for ch in s if not ch.isspace())


def _normalize_eval_filename_key(text: Any) -> str:
    s = _normalize_eval_text(Path(str(text or "")).stem)
    return "".join(ch for ch in s if ch.isalnum() or ("\u4e00" <= ch <= "\u9fff"))


def _eval_filename_alias_keys(text: Any) -> set[str]:
    raw = str(text or "").strip()
    if not raw:
        return set()
    stem = Path(raw).stem
    variants = {stem}
    cleaned = re.sub(r"^[A-Za-z]{2,}\d+\+?", "", stem, flags=re.IGNORECASE)
    cleaned = re.sub(r"^[A-Za-z]+\d+\+?", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^[\W_]+", "", cleaned)
    variants.add(cleaned)
    for suffix in ("-高校精品", "高校精品", "-精品", "精品"):
        if cleaned.endswith(suffix):
            variants.add(cleaned[: -len(suffix)])
    return {k for v in variants if v and (k := _normalize_eval_filename_key(v))}


def _eval_filename_text_variants(text: Any) -> set[str]:
    raw = str(text or "").strip()
    if not raw:
        return set()
    variants = {raw}
    variants.update(x.strip() for x in re.findall(r"《([^》]+)》", raw) if x.strip())
    for part in re.split(r"\s*(?:[/／|,，;；、]|\bor\b|或)\s*", raw, flags=re.IGNORECASE):
        part = str(part or "").strip()
        cleaned = part.strip("《》\"'“”‘’()（）[]【】")
        if part:
            variants.add(part)
        if cleaned:
            variants.add(cleaned)
    return {x for x in variants if x}


def _eval_file_alias_path() -> Path:
    return cfg.data_dir / "eval_file_aliases.json"


def _load_eval_file_alias_entries() -> list[dict[str, str]]:
    path = _eval_file_alias_path()
    payload = _load_json_or_backup(path)
    if payload is None:
        return []
    raw_entries = payload.get("aliases") if isinstance(payload, dict) else payload
    out: list[dict[str, str]] = []
    if isinstance(raw_entries, dict):
        raw_entries = [{"alias": k, "target_file": v} for k, v in raw_entries.items()]
    for item in raw_entries or []:
        if not isinstance(item, dict):
            continue
        alias = str(item.get("alias") or item.get("raw") or "").strip()
        target = Path(str(item.get("target_file") or item.get("target") or "")).name.strip()
        if alias and target:
            out.append({"alias": alias, "target_file": target})
    return out


def _save_eval_file_alias_entries(entries: list[dict[str, str]]) -> None:
    path = _eval_file_alias_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    dedup: dict[str, dict[str, str]] = {}
    for item in entries:
        alias = str(item.get("alias") or "").strip()
        target = Path(str(item.get("target_file") or "")).name.strip()
        if alias and target:
            dedup[_normalize_eval_text(alias)] = {"alias": alias, "target_file": target}
    payload = {"aliases": sorted(dedup.values(), key=lambda x: _normalize_eval_text(x["alias"]))}
    _atomic_write_json(path, payload)


def _eval_alias_lookup_keys(text: Any) -> set[str]:
    raw = str(text or "").strip()
    if not raw:
        return set()
    keys = {raw.lower(), _normalize_eval_text(raw), _normalize_eval_filename_key(raw)}
    keys.update(_candidate_eval_filename_keys(raw))
    return {k for k in keys if k}


def _longest_common_substring_len(a: str, b: str) -> int:
    if not a or not b:
        return 0
    if len(a) > len(b):
        a, b = b, a
    prev = [0] * (len(a) + 1)
    best = 0
    for ch_b in b:
        cur = [0] * (len(a) + 1)
        for i, ch_a in enumerate(a, start=1):
            if ch_a == ch_b:
                cur[i] = prev[i - 1] + 1
                if cur[i] > best:
                    best = cur[i]
        prev = cur
    return best


def _eval_file_alias_key_map(disk_files: list[str]) -> dict[str, set[str]]:
    disk_by_lower = {Path(name).name.lower(): Path(name).name for name in disk_files if str(name or "").strip()}
    out: dict[str, set[str]] = {}
    for item in _load_eval_file_alias_entries():
        target = Path(str(item.get("target_file") or "")).name
        target_lower = target.lower()
        if target_lower not in disk_by_lower:
            continue
        for key in _eval_alias_lookup_keys(item.get("alias")):
            out.setdefault(key, set()).add(target_lower)
    return out


def _uploaded_eval_file_names() -> list[str]:
    return [str(x.get("name") or "").strip() for x in uploaded_files_snapshot(cfg) if str(x.get("name") or "").strip()]


def _candidate_eval_filename_keys(name: str) -> set[str]:
    raw = str(name or "").strip()
    if not raw:
        return set()
    keys: set[str] = set()
    for variant in _eval_filename_text_variants(raw):
        base = Path(variant).name
        stem = Path(base).stem
        keys.update(
            {
                str(base).strip().lower(),
                _normalize_eval_text(base),
                _normalize_eval_filename_key(base),
                _normalize_eval_text(stem),
                _normalize_eval_filename_key(stem),
            }
        )
        keys.update(_eval_filename_alias_keys(base))
    return {k for k in keys if k}


def _suggest_expected_eval_file_candidates(raw: str, disk_files: list[str]) -> list[dict[str, Any]]:
    expected = str(raw or "").strip()
    expected_keys = _candidate_eval_filename_keys(expected)
    expected_variants = _eval_filename_text_variants(expected)
    expected_norms = {_normalize_eval_filename_key(x) for x in expected_variants}
    expected_stems = {_normalize_eval_filename_key(Path(x).stem) for x in expected_variants}
    expected_norms = {x for x in expected_norms if x}
    expected_stems = {x for x in expected_stems if x}
    suggestions: list[dict[str, Any]] = []

    for file_name in disk_files:
        candidate_keys = _candidate_eval_filename_keys(file_name)
        candidate_norm = _normalize_eval_filename_key(file_name)
        candidate_stem = _normalize_eval_filename_key(Path(file_name).stem)
        reasons: list[str] = []
        score = 0.0

        if expected_keys & candidate_keys:
            score = max(score, 1.0)
            reasons.append("别名/规范化键一致")
        for expected_norm in expected_norms:
            if expected_norm and candidate_norm:
                if expected_norm in candidate_norm or candidate_norm in expected_norm:
                    score = max(score, 0.92)
                    reasons.append("文件名包含关系")
                common_len = _longest_common_substring_len(expected_norm, candidate_norm)
                if common_len >= 6:
                    score = max(score, min(0.88, 0.58 + common_len * 0.025))
                    reasons.append("长公共片段")
                score = max(score, difflib.SequenceMatcher(None, expected_norm, candidate_norm).ratio() * 0.86)
        for expected_stem in expected_stems:
            if expected_stem and candidate_stem:
                if expected_stem in candidate_stem or candidate_stem in expected_stem:
                    score = max(score, 0.9)
                    reasons.append("主文件名包含关系")
                common_len = _longest_common_substring_len(expected_stem, candidate_stem)
                if common_len >= 6:
                    score = max(score, min(0.88, 0.58 + common_len * 0.025))
                    reasons.append("主文件名长公共片段")
                score = max(score, difflib.SequenceMatcher(None, expected_stem, candidate_stem).ratio() * 0.88)

        if score >= 0.58:
            suggestions.append(
                {
                    "file": file_name,
                    "score": round(float(score), 3),
                    "reason": "、".join(dict.fromkeys(reasons)) or "名称相似",
                }
            )

    suggestions.sort(key=lambda x: (-float(x.get("score") or 0), str(x.get("file") or "")))
    return suggestions[:3]


def _format_eval_file_suggestions(suggestions: list[dict[str, Any]]) -> str:
    if not suggestions:
        return "-"
    parts = []
    for item in suggestions[:3]:
        file_name = str(item.get("file") or "")
        score = float(item.get("score") or 0)
        reason = str(item.get("reason") or "名称相似")
        parts.append(f"{file_name} ({score:.2f}, {reason})")
    return " | ".join(parts) or "-"


def _resolve_expected_eval_file_details(expected_values: list[Any]) -> dict[str, Any]:
    raw_values = [str(x).strip() for x in (expected_values or []) if str(x).strip()]
    details: list[dict[str, Any]] = []
    if not raw_values:
        return {
            "raw_values": [],
            "resolved_files": [],
            "unresolved_files": [],
            "ambiguous_files": [],
            "suggestions": [],
            "has_unresolved": False,
            "has_ambiguous": False,
        }
    disk_files = _uploaded_eval_file_names()
    disk_set = {name.lower() for name in disk_files}
    disk_key_map: dict[str, set[str]] = {}
    alias_key_map = _eval_file_alias_key_map(disk_files)

    def _add_key(key: str, target: str) -> None:
        if not key:
            return
        disk_key_map.setdefault(key, set()).add(target)

    for name in disk_files:
        lowered = name.lower()
        _add_key(lowered, lowered)
        for key in _candidate_eval_filename_keys(name):
            _add_key(key, lowered)

    resolved_set: set[str] = set()
    unresolved: list[str] = []
    ambiguous: list[dict[str, Any]] = []
    suggestions: list[dict[str, Any]] = []
    for raw in raw_values:
        lowered = raw.lower()
        matched: set[str] = set()
        if lowered in disk_set:
            matched.add(lowered)
        else:
            for key in _eval_alias_lookup_keys(raw):
                matched.update(alias_key_map.get(key) or set())
            for key in _candidate_eval_filename_keys(raw):
                matched.update(disk_key_map.get(key) or set())
        matched_sorted = sorted(matched)
        if len(matched_sorted) == 1:
            resolved_set.update(matched_sorted)
            details.append({"raw": raw, "matches": matched_sorted, "status": "resolved"})
        elif len(matched_sorted) > 1:
            ambiguous.append({"raw": raw, "matches": matched_sorted})
            details.append({"raw": raw, "matches": matched_sorted, "status": "ambiguous"})
        else:
            raw_suggestions = _suggest_expected_eval_file_candidates(raw, disk_files)
            unresolved.append(raw)
            suggestions.append({"raw": raw, "candidates": raw_suggestions})
            details.append({"raw": raw, "matches": [], "status": "unresolved", "suggestions": raw_suggestions})
    return {
        "raw_values": raw_values,
        "resolved_files": sorted(resolved_set),
        "unresolved_files": unresolved,
        "ambiguous_files": ambiguous,
        "suggestions": suggestions,
        "has_unresolved": bool(unresolved),
        "has_ambiguous": bool(ambiguous),
        "details": details,
    }


def _resolve_expected_eval_files(expected_values: list[Any]) -> set[str]:
    return set(_resolve_expected_eval_file_details(expected_values).get("resolved_files") or [])


def _eval_filename_matches_expected(actual_name: str, expected_names: set[str], expected_raw_values: list[Any] | None = None) -> bool:
    actual = str(actual_name or "").strip()
    if not actual:
        return False
    actual_lower = actual.lower()
    if actual_lower in expected_names:
        return True
    actual_keys = _candidate_eval_filename_keys(actual)
    actual_stem_key = _normalize_eval_filename_key(Path(actual).stem)

    for expected in list(expected_names) + [str(x).strip().lower() for x in (expected_raw_values or []) if str(x).strip()]:
        if not expected:
            continue
        if actual_lower == expected:
            return True
        if _filename_equiv(actual, expected):
            return True
        expected_keys = _candidate_eval_filename_keys(expected)
        if actual_keys & expected_keys:
            return True
        expected_stem_key = _normalize_eval_filename_key(Path(expected).stem)
        if actual_stem_key and expected_stem_key:
            if len(actual_stem_key) >= 4 and actual_stem_key in expected_stem_key:
                return True
            if len(expected_stem_key) >= 4 and expected_stem_key in actual_stem_key:
                return True
    return False


def _file_list_hits_expected(file_names: list[str], expected_names: set[str], expected_raw_values: list[Any] | None = None) -> bool | None:
    if not expected_names and not any(str(x).strip() for x in (expected_raw_values or [])):
        return None
    return any(_eval_filename_matches_expected(name, expected_names, expected_raw_values) for name in file_names)


def _is_abstain_answer(answer: str) -> bool:

    return eval_judge.is_abstain_answer(answer)





def _text_match_loose(expect_text: str, actual_text: str) -> bool:

    return eval_judge.text_match_loose(expect_text, actual_text)





def _chunk_match_metrics(expect_text: str, actual_text: str) -> dict[str, Any]:

    return eval_judge.chunk_match_metrics(expect_text, actual_text)





def _chunk_match_diagnostics(

    expected_chunk: str,

    candidates: list[dict[str, Any]],

    sources: list[dict[str, Any]],

) -> dict[str, Any]:

    return eval_judge.chunk_match_diagnostics(expected_chunk, candidates, sources)



    s = str(answer or "")
    return ("无法回答" in s) or ("未检索到任何文档片段" in s)


def _evaluate_case_result(
    case: dict[str, Any],
    vector_diag: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    answer: str,
    *,
    final_ranked_sources: list[dict[str, Any]] | None = None,
    use_rerank: bool = False,
    context_truncated: bool = False,
    generation_mode: str | None = None,
    no_context: bool | None = None,
) -> dict[str, Any]:
    expected_raw_files = [str(x).strip() for x in (case.get("expected_file_names") or []) if str(x).strip()]
    expected_info = _resolve_expected_eval_file_details(expected_raw_files)
    return eval_judge.judge_case(
        case,
        vector_diag,
        sources,
        answer,
        final_ranked_sources=final_ranked_sources,
        use_rerank=use_rerank,
        context_truncated=context_truncated,
        expected_file_info=expected_info,
        generation_mode=generation_mode,
        no_context=no_context,
    )


def _eval_run_summary_rows() -> list[list[Any]]:
    return _eval_run_summary_rows_for_dataset(None)


def _empty_eval_result_rows() -> list[list[Any]]:
    return [[0, "（暂无结果）", "", "", "", "", "", "", "", "", ""]]


def _empty_eval_summary_rows() -> list[list[Any]]:
    return [["样本数", 0]]


def _eval_case_bucket(case: dict[str, Any]) -> str:
    """file_eval=可解析期望文件；unresolved_file_eval=有标注但未解析；non_file_eval=无文件标注。"""
    expected_raw = [str(x).strip() for x in (case.get("expected_file_names") or []) if str(x).strip()]
    if not expected_raw:
        return "non_file_eval"
    expected_files = _resolve_expected_eval_files(expected_raw)
    if expected_files:
        return "file_eval"
    return "unresolved_file_eval"


def _summary_rows_from_summary(summary: dict[str, Any] | None) -> list[list[Any]]:
    summary = summary or {}
    rows = [
        ["Samples", int(summary.get("case_count") or 0)],
        ["File-evaluable cases", int(summary.get("file_eval_case_count") or 0)],
        ["Unresolved expected-file cases", int(summary.get("unresolved_file_case_count") or 0)],
        ["Labeled file cases", int(summary.get("labeled_file_case_count") or 0)],
        ["Non-file cases", int(summary.get("non_file_eval_case_count") or 0)],
        ["Candidate hit rate", summary.get("candidate_hit_rate") or "-"],
        ["Candidate hit rate (all labeled)", summary.get("candidate_hit_rate_all_labeled") or "-"],
        ["Context hit rate", summary.get("context_hit_rate") or "-"],
        ["Context hit rate (all labeled)", summary.get("context_hit_rate_all_labeled") or "-"],
        ["Chunk hit rate", summary.get("chunk_hit_rate") or "-"],
        ["Answer hit rate", summary.get("answer_hit_rate") or "-"],
        ["Abstain accuracy", summary.get("abstain_accuracy") or "-"],
        ["OK count", int(summary.get("ok_count") or 0)],
    ]
    if summary.get("generation_mode"):
        rows.append(["Generation mode", summary.get("generation_mode")])
    if summary.get("query_anchoring_enabled") is not None:
        rows.append(["Query anchoring", "on" if summary.get("query_anchoring_enabled") else "off"])
    if summary.get("retrieval_degraded"):
        rows.append(["Retrieval degraded", "yes"])
    if summary.get("chunk_params_mismatch"):
        rows.append(["Chunk params mismatch", "yes (used index values)"])
    if summary.get("run_fingerprint"):
        rows.append(["Run fingerprint", summary.get("run_fingerprint")])
    issue_counts = ((summary.get("dataset_quality") or {}).get("issue_counts") or {})
    if issue_counts:
        rows.append([
            "Dataset quality issues",
            ", ".join(f"{k}:{v}" for k, v in sorted(issue_counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))[:6]),
        ])
    attr_counts = summary.get("attribution_counts") or {}
    if attr_counts:
        rows.append([
            "Top attribution",
            ", ".join(f"{k}:{v}" for k, v in sorted(attr_counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))[:6]),
        ])
    return rows


def _eval_run_summary_rows_for_dataset(dataset_id: int | None) -> list[list[Any]]:
    rows = store.list_eval_runs(limit=100)
    if dataset_id is not None:
        rows = [r for r in rows if int(r.get("dataset_id") or 0) == int(dataset_id)]
    out: list[list[Any]] = []
    for r in rows[:20]:
        summary = r.get("summary") or {}
        out.append(
            [
                int(r["id"]),
                str(r.get("dataset_name") or ""),
                str(r.get("name") or ""),
                str(r.get("created_at") or ""),
                int(summary.get("case_count") or 0),
                int(summary.get("file_eval_case_count") or 0),
                int(summary.get("non_file_eval_case_count") or 0),
                summary.get("candidate_hit_rate") or "—",
                summary.get("context_hit_rate") or "—",
                summary.get("chunk_hit_rate") or "—",
                summary.get("answer_hit_rate") or "—",
                summary.get("abstain_accuracy") or "—",
            ]
        )
    if not out:
        out = [[0, "（暂无运行）", "", "", 0, 0, 0, "—", "—", "—", "—", "—"]]
    return out


def _empty_eval_param_rows() -> list[list[Any]]:
    return [["参数", "值"], ["（暂无运行）", "—"]]


def _empty_eval_error_rows() -> list[list[Any]]:
    return [["错误类型", "数量", "占比"], ["（暂无结果）", 0, "—"]]


def _empty_eval_tag_rows() -> list[list[Any]]:
    return [["标签", "样本数", "文档命中率", "片段命中率", "答案命中率（参考）", "OK 数"], ["（暂无结果）", 0, "—", "—", "—", 0]]


def _empty_eval_file_rows() -> list[list[Any]]:
    return [["File", "Cases", "Candidate hit", "Context hit", "Chunk hit", "Answer hit", "OK", "Errors"], ["(no results)", 0, "-", "-", "-", "-", 0, "-"]]


def _empty_eval_chunk_diag_rows() -> list[list[Any]]:
    return [["（暂无片段诊断）", "", "", "", "", "", "", "", "", ""]]


def _empty_eval_funnel_rows() -> list[list[Any]]:
    return [["（暂无检索漏斗）", "", "", "", "", "", "", ""]]


def _format_eval_rate_from_bools(values: list[bool | None]) -> str:
    filtered = [bool(v) for v in values if v is not None]
    if not filtered:
        return "—"
    return f"{(sum(1 for v in filtered if v) / len(filtered)) * 100:.1f}%"


def _params_rows_from_run(run: dict[str, Any] | None) -> list[list[Any]]:
    if not run:
        return _empty_eval_param_rows()
    params = run.get("params") or {}
    ordered_keys = [
        ("run_fingerprint", "Run fingerprint"),
        ("generation_mode", "Generation mode"),
        ("retrieval_mode", "Retrieval mode"),
        ("vector_enabled", "Vector enabled"),
        ("keyword_enabled", "Keyword enabled"),
        ("llm_model", "LLM model"),
        ("embed_model", "Embedding model"),
        ("rerank_model", "Rerank model"),
        ("chunk_mode", "Chunk mode"),
        ("chunk_size", "Chunk size"),
        ("chunk_overlap", "Chunk overlap"),
        ("top_n", "Top-N"),
        ("top_k", "Top-K"),
        ("use_rerank", "Use rerank"),
        ("query_anchoring_enabled", "Query anchoring"),
        ("llm_num_ctx", "LLM num_ctx"),
        ("dataset_quality", "Dataset quality"),
        ("index_snapshot", "Index snapshot"),
    ]
    rows: list[list[Any]] = []
    for key, label in ordered_keys:
        val = params.get(key)
        if isinstance(val, (dict, list)):
            val = json.dumps(val, ensure_ascii=False)
        elif val in (None, ""):
            val = "-"
        rows.append([label, val])
    return rows or _empty_eval_param_rows()


def _eval_error_rows_for_run(run_id: int) -> list[list[Any]]:
    results = store.fetch_eval_case_results(run_id)
    if not results:
        return _empty_eval_error_rows()
    total = len(results)
    counts: dict[str, int] = {}
    for item in results:
        key = str(item.get("error_type") or "UNKNOWN")
        counts[key] = counts.get(key, 0) + 1
    rows = []
    for key, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        rows.append([key, count, f"{(count / total) * 100:.1f}%"])
    rows.extend(attribution_rows(results))
    return rows or _empty_eval_error_rows()


def _eval_dataset_quality_rows(dataset_id: int | None) -> list[list[Any]]:
    if dataset_id is None:
        return dataset_quality_rows(None)
    cases = store.fetch_eval_cases(dataset_id)
    summary = dataset_quality_summary(cases, resolve_expected_file_details=_resolve_expected_eval_file_details)
    return dataset_quality_rows(summary)


def _eval_failure_detail_rows(run_id: int | None) -> list[list[Any]]:
    if run_id is None:
        return [["（暂无失败样本）", "", "", "", "", "", "", 0]]
    rows: list[list[Any]] = []
    for item in store.fetch_eval_case_results(run_id):
        err = str(item.get("error_type") or "")
        if err == "OK":
            continue
        diag = item.get("diagnostics") or {}
        eval_diag = diag.get("eval_case_diagnostics") or {}
        expected = eval_diag.get("expected_file_resolution") or {}
        unresolved_files = list(expected.get("unresolved_files") or [])
        suggestion_rows = list(expected.get("suggestions") or [])
        has_suggestion_candidates = any(list(x.get("candidates") or []) for x in suggestion_rows if isinstance(x, dict))
        fallback_resolved_files: list[str] = []
        if unresolved_files and not has_suggestion_candidates:
            fallback_resolution = _resolve_expected_eval_file_details(unresolved_files)
            fallback_resolved_files = list(fallback_resolution.get("resolved_files") or [])
            suggestion_rows = list(fallback_resolution.get("suggestions") or [])
        suggestion_text = " ; ".join(
            f"{str(x.get('raw') or '')}: {_format_eval_file_suggestions(list(x.get('candidates') or []))}"
            for x in suggestion_rows
        )
        if not suggestion_text and fallback_resolved_files:
            suggestion_text = "可解析为: " + " | ".join(str(x) for x in fallback_resolved_files)
        sources = item.get("sources") or []
        rows.append(
            [
                int(item.get("case_id") or 0),
                str(item.get("question") or ""),
                err,
                str(diag.get("retrieval_attribution") or attribution_code(diagnostics=diag, error_type=err)),
                " | ".join(str(x) for x in unresolved_files) or "-",
                suggestion_text or "-",
                str((sources[0] or {}).get("file_name") or "-") if sources else "-",
                int(item.get("qa_id") or 0),
            ]
        )
    return rows or [["（暂无失败样本）", "", "", "", "", "", "", 0]]


def _first_eval_suggestion_file(suggestions_text: str) -> str:
    first = str(suggestions_text or "").split("|", 1)[0].strip()
    if not first or first == "-":
        return ""
    return first.split("(", 1)[0].strip()


def _eval_file_alias_rows(dataset_id: int | None) -> list[list[Any]]:
    rows: list[list[Any]] = []
    disk_files = _uploaded_eval_file_names()
    if dataset_id is not None:
        counts: dict[str, int] = {}
        status_by_raw: dict[str, str] = {}
        target_by_raw: dict[str, str] = {}
        suggestions_by_raw: dict[str, str] = {}
        for case in store.fetch_eval_cases(dataset_id):
            for raw in [str(x).strip() for x in (case.get("expected_file_names") or []) if str(x).strip()]:
                info = _resolve_expected_eval_file_details([raw])
                counts[raw] = counts.get(raw, 0) + 1
                if info.get("resolved_files"):
                    status_by_raw[raw] = "resolved"
                    target_by_raw[raw] = " | ".join(str(x) for x in info.get("resolved_files") or [])
                    suggestions_by_raw[raw] = "-"
                elif info.get("has_ambiguous"):
                    status_by_raw[raw] = "ambiguous"
                    matches = []
                    for item in info.get("ambiguous_files") or []:
                        matches.extend(str(x) for x in item.get("matches") or [])
                    target_by_raw[raw] = " | ".join(sorted(set(matches))) or "-"
                    suggestions_by_raw[raw] = "-"
                else:
                    status_by_raw[raw] = "unresolved"
                    target_by_raw[raw] = "-"
                    cand_rows = list((info.get("suggestions") or [{}])[0].get("candidates") or [])
                    suggestions_by_raw[raw] = _format_eval_file_suggestions(cand_rows)
        for raw, count in sorted(counts.items(), key=lambda kv: (status_by_raw.get(kv[0]) == "resolved", -kv[1], kv[0])):
            rows.append([
                raw,
                int(count),
                status_by_raw.get(raw) or "unknown",
                target_by_raw.get(raw) or "-",
                suggestions_by_raw.get(raw) or "-",
                "dataset",
            ])

    seen_aliases = {_normalize_eval_text(str(r[0])) for r in rows}
    disk_lower = {Path(x).name.lower() for x in disk_files}
    for item in _load_eval_file_alias_entries():
        alias = str(item.get("alias") or "").strip()
        target = Path(str(item.get("target_file") or "")).name.strip()
        if not alias:
            continue
        status = "saved" if target.lower() in disk_lower else "target_missing"
        if _normalize_eval_text(alias) in seen_aliases:
            continue
        rows.append([alias, "-", status, target or "-", "-", "alias"])
    return rows or [["（暂无 alias 治理项）", 0, "-", "-", "-", "-"]]


def _eval_alias_target_dropdown_update(value: str | None = None):
    choices = _uploaded_eval_file_names()
    clean_value = Path(str(value or "")).name.strip()
    if clean_value not in choices:
        clean_value = choices[0] if choices else None
    return gr.update(choices=choices, value=clean_value)


def do_eval_alias_row_select(evt: gr.SelectData):
    if not getattr(evt, "selected", False):
        return "", gr.update()
    row_value = getattr(evt, "row_value", None)
    if not isinstance(row_value, (list, tuple)) or not row_value:
        return "", gr.update()
    raw = str(row_value[0] or "").strip()
    target = str(row_value[3] or "").strip()
    if not target or target == "-":
        target = _first_eval_suggestion_file(str(row_value[4] or ""))
    return raw, _eval_alias_target_dropdown_update(target)


def do_save_eval_file_alias(dataset_choice: str | None, alias_raw: str, target_file: str | None):
    alias = str(alias_raw or "").strip()
    target = Path(str(target_file or "")).name.strip()
    dataset_id = _parse_eval_dataset_id(dataset_choice)
    disk_files = _uploaded_eval_file_names()
    if not alias:
        status = "请先填写或从表格选择 expected raw。"
    elif target not in disk_files:
        status = "请选择一个当前 uploads 中存在的目标文件。"
    else:
        entries = [x for x in _load_eval_file_alias_entries() if _normalize_eval_text(x.get("alias")) != _normalize_eval_text(alias)]
        entries.append({"alias": alias, "target_file": target})
        _save_eval_file_alias_entries(entries)
        status = f"已保存 alias：{alias} -> {target}"
    return (
        status,
        _eval_dataset_quality_rows(dataset_id),
        _eval_failure_detail_rows(_latest_eval_run_id(dataset_id)),
        _eval_file_alias_rows(dataset_id),
        _eval_alias_target_dropdown_update(target),
    )


def do_delete_eval_file_alias(dataset_choice: str | None, alias_raw: str):
    alias = str(alias_raw or "").strip()
    dataset_id = _parse_eval_dataset_id(dataset_choice)
    before = _load_eval_file_alias_entries()
    after = [x for x in before if _normalize_eval_text(x.get("alias")) != _normalize_eval_text(alias)]
    if alias and len(after) != len(before):
        _save_eval_file_alias_entries(after)
        status = f"已删除 alias：{alias}"
    else:
        status = "未找到可删除的 alias。"
    return (
        status,
        _eval_dataset_quality_rows(dataset_id),
        _eval_failure_detail_rows(_latest_eval_run_id(dataset_id)),
        _eval_file_alias_rows(dataset_id),
        _eval_alias_target_dropdown_update(),
    )


def _eval_file_bucket_rows_for_run(run: dict[str, Any] | None) -> list[list[Any]]:
    if not run:
        return [["评测桶", "样本数", "候选命中率", "文档命中率", "片段命中率", "拒答正确率"], ["（暂无结果）", 0, "—", "—", "—", "—"]]
    run_id = int(run.get("id") or 0)
    dataset_id = int(run.get("dataset_id") or 0)
    results = store.fetch_eval_case_results(run_id)
    if not results or dataset_id <= 0:
        return [["评测桶", "样本数", "候选命中率", "文档命中率", "片段命中率", "拒答正确率"], ["（暂无结果）", 0, "—", "—", "—", "—"]]
    case_map = {int(x["id"]): x for x in store.fetch_eval_cases(dataset_id)}
    bucket_map: dict[str, list[dict[str, Any]]] = {"file_eval": [], "non_file_eval": []}
    for item in results:
        case = case_map.get(int(item.get("case_id") or 0), {})
        bucket_map.setdefault(_eval_case_bucket(case), []).append(item)

    def _rate(items: list[dict[str, Any]], field: str) -> str:
        vals = [bool(x[field]) for x in items if x.get(field) is not None]
        if not vals:
            return "—"
        return f"{(sum(1 for v in vals if v) / len(vals)) * 100:.1f}%"

    labels = {
        "file_eval": "文件命中可评估",
        "non_file_eval": "非文件命中题",
    }
    rows: list[list[Any]] = []
    for key in ("file_eval", "non_file_eval"):
        items = bucket_map.get(key) or []
        rows.append([
            labels[key],
            len(items),
            _rate(items, "candidate_hit"),
            _rate(items, "context_hit"),
            _rate(items, "chunk_hit"),
            _rate(items, "abstain_correct"),
        ])
    return rows


def _eval_tag_rows_for_run(run: dict[str, Any] | None) -> list[list[Any]]:
    if not run:
        return _empty_eval_tag_rows()
    run_id = int(run.get("id") or 0)
    dataset_id = int(run.get("dataset_id") or 0)
    results = store.fetch_eval_case_results(run_id)
    if not results or dataset_id <= 0:
        return _empty_eval_tag_rows()
    case_map = {int(x["id"]): x for x in store.fetch_eval_cases(dataset_id)}
    bucket: dict[str, list[dict[str, Any]]] = {}
    for item in results:
        case = case_map.get(int(item.get("case_id") or 0), {})
        tags = list(case.get("tags") or []) or ["（未标注）"]
        for tag in tags:
            key = str(tag or "").strip() or "（未标注）"
            bucket.setdefault(key, []).append(item)
    rows: list[list[Any]] = []
    for tag, items in sorted(bucket.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        rows.append(
            [
                tag,
                len(items),
                _format_eval_rate_from_bools([x.get("context_hit") for x in items]),
                _format_eval_rate_from_bools([x.get("chunk_hit") for x in items]),
                _format_eval_rate_from_bools([x.get("answer_hit") for x in items]),
                sum(1 for x in items if str(x.get("error_type") or "") == "OK"),
            ]
        )
    return rows or _empty_eval_tag_rows()


def _eval_file_rows_for_run(run: dict[str, Any] | None) -> list[list[Any]]:
    if not run:
        return _empty_eval_file_rows()
    run_id = int(run.get("id") or 0)
    dataset_id = int(run.get("dataset_id") or 0)
    results = store.fetch_eval_case_results(run_id)
    if not results or dataset_id <= 0:
        return _empty_eval_file_rows()
    case_map = {int(x["id"]): x for x in store.fetch_eval_cases(dataset_id)}
    bucket: dict[str, list[dict[str, Any]]] = {}
    for item in results:
        case = case_map.get(int(item.get("case_id") or 0), {})
        names = list(case.get("expected_file_names") or [])
        if not names:
            sources = item.get("sources") or []
            names = [str((sources[0] or {}).get("file_name") or "(no expected file)")] if sources else ["(no expected file)"]
        for name in names:
            bucket.setdefault(str(name or "(blank)"), []).append(item)
    rows: list[list[Any]] = []
    for name, items in sorted(bucket.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        err_counts: dict[str, int] = {}
        for item in items:
            err = str(item.get("error_type") or "UNKNOWN")
            if err != "OK":
                err_counts[err] = err_counts.get(err, 0) + 1
        err_s = ", ".join(f"{k}:{v}" for k, v in sorted(err_counts.items(), key=lambda kv: (-kv[1], kv[0]))) or "-"
        rows.append([
            name,
            len(items),
            _format_eval_rate_from_bools([x.get("candidate_hit") for x in items]),
            _format_eval_rate_from_bools([x.get("context_hit") for x in items]),
            _format_eval_rate_from_bools([x.get("chunk_hit") for x in items]),
            _format_eval_rate_from_bools([x.get("answer_hit") for x in items]),
            sum(1 for x in items if str(x.get("error_type") or "") == "OK"),
            err_s,
        ])
    return rows or _empty_eval_file_rows()


def _eval_result_rows_for_run(run_id: int) -> list[list[Any]]:
    results = store.fetch_eval_case_results(run_id)
    rows: list[list[Any]] = []
    for idx, item in enumerate(results, start=1):
        sources = item.get("sources") or []
        diag = item.get("diagnostics") or {}
        attribution = str(
            diag.get("retrieval_attribution") or attribution_code(diagnostics=diag, error_type=item.get("error_type"))
        )
        rows.append(
            [
                idx,
                str(item.get("question") or ""),
                "命中" if item.get("candidate_hit") else ("—" if item.get("candidate_hit") is None else "未命中"),
                "命中" if item.get("context_hit") else ("—" if item.get("context_hit") is None else "未命中"),
                "命中" if item.get("chunk_hit") else ("—" if item.get("chunk_hit") is None else "未命中"),
                "命中" if item.get("answer_hit") else ("—" if item.get("answer_hit") is None else "未命中"),
                "正确" if item.get("abstain_correct") else "错误",
                str(item.get("error_type") or ""),
                attribution,
                str((sources[0] or {}).get("file_name") or "—") if sources else "—",
                item.get("qa_id") or "",
            ]
        )
    return rows or _empty_eval_result_rows()


def _eval_chunk_diag_rows_for_run(run_id: int | None) -> list[list[Any]]:
    if run_id is None:
        return _empty_eval_chunk_diag_rows()
    run = _eval_run_by_id(run_id)
    dataset_id = int((run or {}).get("dataset_id") or 0)
    case_map = {int(x["id"]): x for x in store.fetch_eval_cases(dataset_id)} if dataset_id > 0 else {}
    rows: list[list[Any]] = []
    for item in store.fetch_eval_case_results(run_id):
        diag = item.get("diagnostics") or {}
        eval_diag = diag.get("eval_case_diagnostics") or {}
        chunk_diag = eval_diag.get("chunk_diagnostics") or {}
        if not chunk_diag.get("has_expected_chunk"):
            case = case_map.get(int(item.get("case_id") or 0), {})
            chunk_diag = _chunk_match_diagnostics(
                str(case.get("expected_chunk_content") or ""),
                list(diag.get("merged_candidates") or []),
                list(diag.get("final_sources") or item.get("sources") or []),
            )
        if not chunk_diag.get("has_expected_chunk"):
            continue
        best = dict(chunk_diag.get("best") or {})
        rows.append(
            [
                int(item.get("case_id") or 0),
                str(item.get("error_type") or ""),
                "命中" if item.get("chunk_hit") else "未命中",
                str(best.get("stage") or "-"),
                int(best.get("rank") or 0),
                str(best.get("file_name") or "-"),
                f"{float(best.get('similarity') or 0):.3f}",
                f"{float(best.get('coverage') or 0) * 100:.1f}%",
                int(best.get("common_chars") or 0),
                str(best.get("preview") or ""),
            ]
        )
    rows.sort(key=lambda r: (r[2] == "命中", -int(r[8] or 0), int(r[0] or 0)))
    return rows or _empty_eval_chunk_diag_rows()


def _stage_file_names(items: list[dict[str, Any]]) -> list[str]:
    return [str(x.get("file_name") or "").strip() for x in items if str(x.get("file_name") or "").strip()]


def _format_top_files(items: list[dict[str, Any]], limit: int = 3) -> str:
    names: list[str] = []
    for item in items[:limit]:
        name = str(item.get("file_name") or "").strip()
        if name:
            names.append(name)
    return " | ".join(names) or "-"


def _eval_funnel_rows_for_run(run_id: int | None) -> list[list[Any]]:
    if run_id is None:
        return _empty_eval_funnel_rows()
    rows: list[list[Any]] = []
    for item in store.fetch_eval_case_results(run_id):
        diag = item.get("diagnostics") or {}
        eval_diag = diag.get("eval_case_diagnostics") or {}
        expected = eval_diag.get("expected_file_resolution") or {}
        expected_files = set(expected.get("resolved_files") or [])
        expected_raw = list(expected.get("raw_values") or [])
        stages = [
            ("vector", list(diag.get("vector_candidates") or [])),
            ("keyword", list(diag.get("keyword_candidates") or [])),
            ("merged", list(diag.get("merged_candidates") or [])),
            ("final_ranked", list(diag.get("final_ranked_contexts") or [])),
            ("context", list(diag.get("final_sources") or item.get("sources") or [])),
        ]
        for stage, items in stages:
            hit = _file_list_hits_expected(_stage_file_names(items), expected_files, expected_raw)
            rows.append(
                [
                    int(item.get("case_id") or 0),
                    stage,
                    len(items),
                    "命中" if hit else ("—" if hit is None else "未命中"),
                    " | ".join(str(x) for x in expected.get("unresolved_files") or []) or "-",
                    _format_top_files(items),
                    str(item.get("error_type") or ""),
                    int(item.get("qa_id") or 0),
                ]
            )
    return rows or _empty_eval_funnel_rows()


def _eval_run_dashboard_outputs(dataset_id: int | None, run_id: int | None = None):
    recent_rows = _eval_run_summary_rows_for_dataset(dataset_id)
    if run_id is None:
        return (
            _empty_eval_summary_rows(),
            recent_rows,
            _empty_eval_result_rows(),
            _empty_eval_param_rows(),
            _empty_eval_error_rows(),
            _empty_eval_tag_rows(),
            _empty_eval_file_rows(),
            _empty_eval_chunk_diag_rows(),
            _empty_eval_funnel_rows(),
        )
    run_rows = store.list_eval_runs(limit=200)
    run = next((r for r in run_rows if int(r.get("id") or 0) == int(run_id)), None)
    summary_rows = _summary_rows_from_summary((run or {}).get("summary") or {})
    result_rows = _eval_result_rows_for_run(run_id)
    param_rows = _params_rows_from_run(run)
    error_rows = _eval_error_rows_for_run(run_id)
    tag_rows = _eval_tag_rows_for_run(run)
    file_rows = _eval_file_rows_for_run(run)
    chunk_diag_rows = _eval_chunk_diag_rows_for_run(run_id)
    funnel_rows = _eval_funnel_rows_for_run(run_id)
    return summary_rows, recent_rows, result_rows, param_rows, error_rows, tag_rows, file_rows, chunk_diag_rows, funnel_rows


def _finalize_eval_run_summary(run_id: int) -> list[list[Any]]:
    results = store.fetch_eval_case_results(run_id)
    run_rows = store.list_eval_runs(limit=200)
    run = next((r for r in run_rows if int(r.get("id") or 0) == int(run_id)), None)
    dataset_id = int((run or {}).get("dataset_id") or 0)
    case_map = {int(x["id"]): x for x in store.fetch_eval_cases(dataset_id)} if dataset_id > 0 else {}
    buckets = {
        int(x.get("case_id") or 0): _eval_case_bucket(case_map.get(int(x.get("case_id") or 0), {}))
        for x in results
    }
    file_eval_results = [x for x in results if buckets.get(int(x.get("case_id") or 0)) == "file_eval"]
    unresolved_results = [x for x in results if buckets.get(int(x.get("case_id") or 0)) == "unresolved_file_eval"]
    labeled_results = file_eval_results + unresolved_results
    non_file_eval_results = [x for x in results if buckets.get(int(x.get("case_id") or 0)) == "non_file_eval"]
    params = (run or {}).get("params") or {}
    retrieval_only = str(params.get("generation_mode") or "").strip().lower() in {"retrieval_only", "retrieval-only"}

    def _rate(field: str, items: list[dict[str, Any]] | None = None) -> str:
        pool = items if items is not None else results
        vals = [bool(x[field]) for x in pool if x.get(field) is not None]
        if not vals:
            return "—"
        return f"{(sum(1 for v in vals if v) / len(vals)) * 100:.1f}%"

    def _rate_all_labeled(field: str) -> str:
        """可解析样本按原字段计；未解析期望文件计为未命中（暴露进分母）。"""
        if not labeled_results:
            return "—"
        hits = 0
        for item in labeled_results:
            bucket = buckets.get(int(item.get("case_id") or 0))
            if bucket == "unresolved_file_eval":
                continue
            if item.get(field) is True:
                hits += 1
        return f"{(hits / len(labeled_results)) * 100:.1f}%"

    summary = {
        "case_count": len(results),
        "file_eval_case_count": len(file_eval_results),
        "unresolved_file_case_count": len(unresolved_results),
        "labeled_file_case_count": len(labeled_results),
        "non_file_eval_case_count": len(non_file_eval_results),
        "candidate_hit_rate": _rate("candidate_hit", file_eval_results),
        "candidate_hit_rate_all_labeled": _rate_all_labeled("candidate_hit"),
        "context_hit_rate": _rate("context_hit", file_eval_results),
        "context_hit_rate_all_labeled": _rate_all_labeled("context_hit"),
        "chunk_hit_rate": _rate("chunk_hit", file_eval_results),
        "answer_hit_rate": "—" if retrieval_only else _rate("answer_hit", results),
        "abstain_accuracy": "—" if retrieval_only else _rate("abstain_correct", results),
        "ok_count": sum(1 for x in results if str(x.get("error_type") or "") == "OK"),
        "generation_mode": params.get("generation_mode"),
        "query_anchoring_enabled": params.get("query_anchoring_enabled"),
        "retrieval_degraded": bool(params.get("retrieval_degraded")),
        "chunk_params_mismatch": bool(params.get("chunk_params_mismatch")),
    }
    attr_counts: dict[str, int] = {}
    for item in results:
        code = attribution_code(diagnostics=item.get("diagnostics"), error_type=item.get("error_type"))
        attr_counts[code] = attr_counts.get(code, 0) + 1
    summary["attribution_counts"] = attr_counts
    summary["run_fingerprint"] = params.get("run_fingerprint")
    summary["dataset_quality"] = params.get("dataset_quality") or {}
    store.finish_eval_run(run_id, summary)
    return _summary_rows_from_summary(summary)


def _eval_run_id_from_select(evt: gr.SelectData) -> int | None:
    if not getattr(evt, "selected", False):
        return None
    row_value = getattr(evt, "row_value", None)
    if isinstance(row_value, (list, tuple)) and row_value:
        try:
            rid = int(str(row_value[0]).strip())
            return rid if rid > 0 else None
        except (TypeError, ValueError):
            return None
    return None


def do_eval_dataset_import(file_obj: Any, dataset_name: str, description: str):
    try:
        cases = _parse_eval_cases_from_file(file_obj)
    except Exception as e:
        msg = f"导入失败：{type(e).__name__}: {e}"
        if "openpyxl" in str(e).lower():
            msg += "。当前环境缺少 Excel 依赖，请在项目 .venv 中重新安装 requirements.txt。"
        choices, value = _eval_dataset_choices()
        return (
            msg,
            gr.update(choices=choices, value=value),
            [[0, "（暂无样本）", "", "", "", "", "", ""]],
            "未导入评测集。",
            _eval_run_summary_rows_for_dataset(None),
            _empty_eval_result_rows(),
            _empty_eval_summary_rows(),
            _empty_eval_param_rows(),
            _empty_eval_error_rows(),
            _empty_eval_tag_rows(),
            _empty_eval_file_rows(),
            _empty_eval_chunk_diag_rows(),
            _empty_eval_funnel_rows(),
            _eval_dataset_quality_rows(None),
            _eval_failure_detail_rows(None),
            _eval_file_alias_rows(None),
            _eval_alias_target_dropdown_update(),
        )
    if not cases:
        choices, value = _eval_dataset_choices()
        return (
            "导入失败：文件里没有有效样本，至少需要 question 字段。",
            gr.update(choices=choices, value=value),
            [[0, "（暂无样本）", "", "", "", "", "", ""]],
            "未导入评测集。",
            _eval_run_summary_rows_for_dataset(None),
            _empty_eval_result_rows(),
            _empty_eval_summary_rows(),
            _empty_eval_param_rows(),
            _empty_eval_error_rows(),
            _empty_eval_tag_rows(),
            _empty_eval_file_rows(),
            _empty_eval_chunk_diag_rows(),
            _empty_eval_funnel_rows(),
            _eval_dataset_quality_rows(None),
            _eval_failure_detail_rows(None),
            _eval_file_alias_rows(None),
            _eval_alias_target_dropdown_update(),
        )
    name = str(dataset_name or "").strip()
    if not name:
        name = Path(getattr(file_obj, "name", file_obj)).stem
    dataset_id = store.replace_eval_dataset(name, cases, description=str(description or "").strip() or None)
    imported_dataset = store.get_eval_dataset(dataset_id) or {}
    imported_name = str(imported_dataset.get("name") or name)
    choices, _ = _eval_dataset_choices()
    choice_value = next((x for x in choices if x.startswith(f"#{dataset_id} ")), None)
    summary_rows, recent_rows, result_rows, param_rows, error_rows, tag_rows, file_rows, chunk_diag_rows, funnel_rows = _eval_run_dashboard_outputs(dataset_id, None)
    return (
        f"已导入评测集：#{dataset_id} {imported_name}，共 {len(cases)} 题。",
        gr.update(choices=choices, value=choice_value),
        _eval_cases_preview_rows(store.fetch_eval_cases(dataset_id)),
        _eval_dataset_summary_markdown(dataset_id),
        recent_rows,
        result_rows,
        summary_rows,
        param_rows,
        error_rows,
        tag_rows,
        file_rows,
        chunk_diag_rows,
        funnel_rows,
        _eval_dataset_quality_rows(dataset_id),
        _eval_failure_detail_rows(_latest_eval_run_id(dataset_id)),
        _eval_file_alias_rows(dataset_id),
        _eval_alias_target_dropdown_update(),
    )


def do_eval_dataset_select(dataset_choice: str | None):
    dataset_id = _parse_eval_dataset_id(dataset_choice)
    if dataset_id is None:
        return (
            "未选择评测集。",
            [[0, "（暂无样本）", "", "", "", "", "", ""]],
            _eval_run_summary_rows_for_dataset(None),
            _empty_eval_result_rows(),
            _empty_eval_summary_rows(),
            _empty_eval_param_rows(),
            _empty_eval_error_rows(),
            _empty_eval_tag_rows(),
            _empty_eval_file_rows(),
            _empty_eval_chunk_diag_rows(),
            _empty_eval_funnel_rows(),
            _eval_dataset_quality_rows(None),
            _eval_failure_detail_rows(None),
            _eval_file_alias_rows(None),
            _eval_alias_target_dropdown_update(),
        )
    summary_rows, recent_rows, result_rows, param_rows, error_rows, tag_rows, file_rows, chunk_diag_rows, funnel_rows = _eval_run_dashboard_outputs(dataset_id, None)
    return (
        _eval_dataset_summary_markdown(dataset_id),
        _eval_cases_preview_rows(store.fetch_eval_cases(dataset_id)),
        recent_rows,
        result_rows,
        summary_rows,
        param_rows,
        error_rows,
        tag_rows,
        file_rows,
        chunk_diag_rows,
        funnel_rows,
        _eval_dataset_quality_rows(dataset_id),
        _eval_failure_detail_rows(_latest_eval_run_id(dataset_id)),
        _eval_file_alias_rows(dataset_id),
        _eval_alias_target_dropdown_update(),
    )


def do_eval_run_select(dataset_choice: str | None, evt: gr.SelectData):
    dataset_id = _parse_eval_dataset_id(dataset_choice)
    run_id = _eval_run_id_from_select(evt)
    if run_id is None:
        summary_rows, recent_rows, result_rows, param_rows, error_rows, tag_rows, file_rows, chunk_diag_rows, funnel_rows = _eval_run_dashboard_outputs(dataset_id, None)
        return "未选中实验运行。", result_rows, summary_rows, recent_rows, param_rows, error_rows, tag_rows, file_rows, chunk_diag_rows, funnel_rows
    summary_rows, recent_rows, result_rows, param_rows, error_rows, tag_rows, file_rows, chunk_diag_rows, funnel_rows = _eval_run_dashboard_outputs(dataset_id, run_id)
    return f"已载入 RUN#{run_id} 的实验结果。", result_rows, summary_rows, recent_rows, param_rows, error_rows, tag_rows, file_rows, chunk_diag_rows, funnel_rows


def do_eval_dashboard_refresh(dataset_choice: str | None):
    dataset_id = _parse_eval_dataset_id(dataset_choice)
    summary_rows, recent_rows, result_rows, param_rows, error_rows, tag_rows, file_rows, chunk_diag_rows, funnel_rows = _eval_run_dashboard_outputs(dataset_id, None)
    if dataset_id is None:
        return "未选择评测集。", recent_rows, result_rows, summary_rows, param_rows, error_rows, tag_rows, file_rows, chunk_diag_rows, funnel_rows
    return f"已刷新评测集 #{dataset_id} 的实验面板。", recent_rows, result_rows, summary_rows, param_rows, error_rows, tag_rows, file_rows, chunk_diag_rows, funnel_rows


def do_eval_governance_refresh(dataset_choice: str | None):
    dataset_id = _parse_eval_dataset_id(dataset_choice)
    return (
        _eval_dataset_quality_rows(dataset_id),
        _eval_failure_detail_rows(_latest_eval_run_id(dataset_id)),
        _eval_file_alias_rows(dataset_id),
        _eval_alias_target_dropdown_update(),
    )


def _summary_rows_to_map(rows: list[list[Any]]) -> dict[str, Any]:
    return {str(r[0]): r[1] for r in rows or [] if isinstance(r, (list, tuple)) and len(r) >= 2}


def do_run_retrieval_param_grid(dataset_choice: str | None):
    dataset_id = _parse_eval_dataset_id(dataset_choice)
    if dataset_id is None:
        return "请先选择评测集。", [["（暂无网格结果）", "", "", "", "", "", "", "", ""]], _eval_run_summary_rows_for_dataset(None), _eval_run_compare_rows(), _eval_run_diff_rows(), _eval_baseline_compare_rows(None)
    specs = [
        {"mode": "keyword", "top_n": 5, "top_k": 1, "rerank": False},
        {"mode": "keyword", "top_n": 8, "top_k": 3, "rerank": False},
        {"mode": "vector", "top_n": 5, "top_k": 2, "rerank": False},
        {"mode": "hybrid", "top_n": 5, "top_k": 2, "rerank": False},
        {"mode": "hybrid", "top_n": 8, "top_k": 3, "rerank": False},
        {"mode": "hybrid", "top_n": 8, "top_k": 3, "rerank": True},
    ]
    rows: list[list[Any]] = []
    logs: list[str] = [f"开始 retrieval-only 参数网格，共 {len(specs)} 组。"]
    for spec in specs:
        mode = str(spec["mode"])
        tn = int(spec["top_n"])
        tk = int(spec["top_k"])
        rr = bool(spec["rerank"])
        # 参数网格只变检索 knobs；切片取自索引；anchoring 关闭（真实召回）
        out = do_run_eval_dataset(
            dataset_choice,
            f"grid retrieval {mode} topN={tn} topK={tk} rerank={rr} {time.strftime('%Y-%m-%d %H:%M:%S')}",
            str(cfg.prompt.get("system_default", "")).strip(),
            tn,
            tk,
            rr,
            None,
            None,
            None,
            str(cfg.ollama.get("llm_model") or ""),
            2048,
            str(cfg.ollama.get("embed_model") or ""),
            "retrieval_only",
            mode,
            False,
        )
        try:
            status, _result_rows, summary_rows, _recent_rows, _param_rows, _error_rows, _tag_rows, _file_rows, _chunk_rows, _funnel_rows = out
            summary = _summary_rows_to_map(summary_rows)
            run = _latest_eval_run(dataset_id)
            rows.append([
                int((run or {}).get("id") or 0),
                mode,
                "yes" if rr else "no",
                tn,
                tk,
                summary.get("Candidate hit rate") or "-",
                summary.get("Context hit rate") or "-",
                summary.get("Chunk hit rate") or "-",
                summary.get("Candidate hit rate (all labeled)") or "-",
            ])
            logs.append(str(status).splitlines()[0] if str(status).strip() else f"topN={tn}, topK={tk} done")
        except Exception as exc:
            rows.append([0, mode, "yes" if rr else "no", tn, tk, "-", "-", "-", f"ERROR: {type(exc).__name__}: {exc}"])
            logs.append(f"topN={tn}, topK={tk} 失败：{type(exc).__name__}: {exc}")
    if not rows:
        rows = [["（暂无网格结果）", "", "", "", "", "", "", "", ""]]
    if rows and str(rows[0][0]).isdigit():
        rows.sort(
            key=lambda r: (
                _pct_text_to_float(r[6]) or -1.0,
                _pct_text_to_float(r[7]) or -1.0,
                _pct_text_to_float(r[5]) or -1.0,
            ),
            reverse=True,
        )
    return "\n".join(logs), rows, _eval_run_summary_rows_for_dataset(dataset_id), _eval_run_compare_rows(), _eval_run_diff_rows(), _eval_baseline_compare_rows(dataset_id)


def do_run_eval_dataset(
    dataset_choice: str | None,
    run_name: str,
    system_prompt: str,
    top_n: int,
    top_k: int,
    use_rerank: bool,
    chunk_size: int | None,
    chunk_overlap: int | None,
    chunk_mode: str | None,
    llm_model: str,
    llm_num_ctx: float,
    embed_model: str,
    generation_mode: str = "llm",
    retrieval_mode: str = "hybrid",
    query_anchoring: bool = False,
):
    dataset_id = _parse_eval_dataset_id(dataset_choice)
    if dataset_id is None:
        return (
            "请先选择评测集。",
            _empty_eval_result_rows(),
            _empty_eval_summary_rows(),
            _eval_run_summary_rows_for_dataset(None),
            _empty_eval_param_rows(),
            _empty_eval_error_rows(),
            _empty_eval_tag_rows(),
            _empty_eval_file_rows(),
            _empty_eval_chunk_diag_rows(),
            _empty_eval_funnel_rows(),
        )
    eng = _lazy_engine()
    cases = store.fetch_eval_cases(dataset_id)
    if not cases:
        return (
            "当前评测集没有样本。",
            _empty_eval_result_rows(),
            _empty_eval_summary_rows(),
            _eval_run_summary_rows_for_dataset(dataset_id),
            _empty_eval_param_rows(),
            _empty_eval_error_rows(),
            _empty_eval_tag_rows(),
            _empty_eval_file_rows(),
            _empty_eval_chunk_diag_rows(),
            _empty_eval_funnel_rows(),
        )
    em = (embed_model or "").strip() or None
    generation_mode = normalize_generation_mode(generation_mode, llm_model)
    retrieval_only = generation_mode == "retrieval_only"
    lm = effective_llm_model(llm_model, generation_mode)
    retrieval_mode = normalize_retrieval_mode(retrieval_mode)
    vector_enabled, keyword_enabled = retrieval_flags(retrieval_mode)
    cm = (str(chunk_mode).strip().lower() if chunk_mode else None)
    query_anchoring_enabled = bool(query_anchoring)
    try:
        nctx = int(round(float(llm_num_ctx)))
    except (TypeError, ValueError):
        nctx = 16384
    nctx = max(2048, min(nctx, 262144))
    top_n = max(1, int(top_n))
    top_k = max(1, int(top_k))
    try:
        chunk_size_i = int(chunk_size) if chunk_size is not None else None
    except (TypeError, ValueError):
        chunk_size_i = None
    try:
        chunk_overlap_i = int(chunk_overlap) if chunk_overlap is not None else None
    except (TypeError, ValueError):
        chunk_overlap_i = None

    index = eng.get_index(cfg, embed_model=em)
    if index is None:
        return (
            "当前没有可用索引，请先构建知识库索引。",
            _empty_eval_result_rows(),
            _empty_eval_summary_rows(),
            _eval_run_summary_rows_for_dataset(dataset_id),
            _empty_eval_param_rows(),
            _empty_eval_error_rows(),
            _empty_eval_tag_rows(),
            _empty_eval_file_rows(),
            _empty_eval_chunk_diag_rows(),
            _empty_eval_funnel_rows(),
        )
    exclusion_info = build_excluded_file_set(cfg, index=index, prefer_index=True)
    excluded_files = list(exclusion_info.get("excluded_files") or [])
    excluded_doc_classes = list(exclusion_info.get("excluded_doc_classes") or [])
    include_zero_chunk = bool(exclusion_info.get("include_zero_chunk"))

    params = eng.build_params_snapshot(
        cfg,
        chunk_size_i,
        chunk_overlap_i,
        top_n,
        top_k,
        bool(use_rerank),
        chunk_mode=cm,
        llm_model=lm,
        embed_model=em,
        llm_num_ctx=nctx,
        excluded_files=excluded_files,
        excluded_doc_classes=excluded_doc_classes,
        include_zero_chunk=include_zero_chunk,
        query_anchoring_enabled=query_anchoring_enabled,
        query_anchoring_source="expected_file_names" if query_anchoring_enabled else None,
        vector_enabled=vector_enabled,
        keyword_enabled=keyword_enabled,
        generation_mode=generation_mode,
        prefer_index_chunk_params=True,
    )
    params["dataset_quality"] = dataset_quality_summary(cases, resolve_expected_file_details=_resolve_expected_eval_file_details)
    index_state = _build_index_status_snapshot()
    index_block_msg = _index_state_blocking_message(index_state)
    if index_block_msg:
        return (
            index_block_msg,
            _empty_eval_result_rows(),
            _empty_eval_summary_rows(),
            _eval_run_summary_rows_for_dataset(dataset_id),
            _empty_eval_param_rows(),
            _empty_eval_error_rows(),
            _empty_eval_tag_rows(),
            _empty_eval_file_rows(),
            _empty_eval_chunk_diag_rows(),
            _empty_eval_funnel_rows(),
        )
    title = str(run_name or "").strip() or f"评测运行 {time.strftime('%Y-%m-%d %H:%M:%S')}"
    run_id = store.create_eval_run(dataset_id, title, params)
    run_error = ""
    run_vector_degraded = False
    log_lines = [f"已创建评测运行：RUN#{run_id}，共 {len(cases)} 题。"]
    log_lines.append(
        f"当前排除文件 {len(excluded_files)} 个，分类={('/'.join(excluded_doc_classes) or '—')}，含 zero-chunk={'是' if include_zero_chunk else '否'}。"
    )
    if params.get("chunk_params_warning"):
        log_lines.append(str(params["chunk_params_warning"]))
    else:
        log_lines.append(
            f"切片参数以索引为准：{params.get('chunk_mode')}/{params.get('chunk_size')}/{params.get('chunk_overlap')}。"
        )
    if query_anchoring_enabled:
        log_lines.append("Query anchoring=ON：基于 expected_file_names 增强检索问题（非真实召回口径，勿作 baseline）。")
    else:
        log_lines.append("Query anchoring=OFF：使用原始问题检索（真实召回口径）。")
    log_lines.append(f"Generation mode={generation_mode}; retrieval mode={retrieval_mode}.")
    if retrieval_only:
        log_lines.append("当前运行为 retrieval-only：跳过 LLM；answer/abstain 指标不计分。")
    result_rows: list[list[Any]] = []

    try:
        for idx, case in enumerate(cases, start=1):
            qtext = str(case.get("question") or "").strip()
            if query_anchoring_enabled:
                retrieval_query, query_anchors = eng.build_anchored_eval_query(
                    qtext, case.get("expected_file_names") or []
                )
            else:
                retrieval_query, query_anchors = qtext, []
            log_lines.append(f"[{idx}/{len(cases)}] {qtext}")
            retrieval_result = eng.hybrid_retrieve(
                cfg,
                index,
                retrieval_query,
                top_n,
                top_k,
                bool(use_rerank),
                excluded_files,
                vector_enabled=vector_enabled,
                keyword_enabled=keyword_enabled,
            )
            if retrieval_result.retrieval_degraded:
                run_vector_degraded = True
            vector_diag = eng.nodes_to_source_dicts(retrieval_result.vector_nodes, "vector")
            keyword_diag = eng.nodes_to_source_dicts(retrieval_result.keyword_nodes, "keyword")
            merged_diag = eng.nodes_to_source_dicts(retrieval_result.merged_nodes, "hybrid")
            excluded_candidate_count = retrieval_result.excluded_candidate_count
            nodes, kind = retrieval_result.final_nodes, retrieval_result.score_kind
            final_ranked_sources = eng.nodes_to_source_dicts(nodes, kind)
            nodes_for_answer, context_truncated = eng.select_nodes_for_answer(
                cfg,
                qtext,
                nodes,
                llm_model=lm,
                llm_num_ctx=nctx,
            )
            sources = eng.nodes_to_source_dicts(nodes_for_answer, kind)
            no_context = not bool(nodes_for_answer)
            diag_payload = _retrieval_diag_payload(
                query=retrieval_query,
                raw_query=qtext,
                query_anchors=query_anchors,
                vector_candidates=vector_diag,
                keyword_candidates=keyword_diag,
                merged_candidates=merged_diag,
                final_sources=sources,
                top_n=top_n,
                top_k=top_k,
                use_rerank=bool(use_rerank),
                score_kind=kind,
                excluded_files=excluded_files,
                excluded_doc_classes=excluded_doc_classes,
                include_zero_chunk=include_zero_chunk,
                excluded_candidate_count=excluded_candidate_count,
                retrieval_mode=retrieval_mode,
            )
            diag_payload["final_ranked_contexts"] = final_ranked_sources
            diag_payload["context_selected_count"] = len(nodes_for_answer)
            diag_payload["context_truncated"] = bool(context_truncated)
            diag_payload["vector_error"] = str(retrieval_result.vector_error or "")
            diag_payload["retrieval_degraded"] = bool(retrieval_result.retrieval_degraded)
            diag_payload["requested_retrieval_mode"] = str(
                getattr(retrieval_result, "requested_retrieval_mode", retrieval_mode) or retrieval_mode
            )
            diag_payload["query_anchoring_enabled"] = query_anchoring_enabled
            diag_payload["no_context"] = no_context
            diag_payload["index_state"] = dict(index_state)
            diag_payload["index_state_summary"] = _index_state_summary_text(index_state)
            if no_context:
                # 哨兵答案：不计为模型拒答，避免空检索被标成 OVER_ABSTAIN
                answer = eval_judge.ANSWER_NO_CONTEXT
            elif retrieval_only:
                answer = eval_judge.ANSWER_RETRIEVAL_ONLY
            else:
                parts: list[str] = []
                for token in eng.stream_answer(
                    cfg, qtext, system_prompt, nodes_for_answer, llm_model=lm, llm_num_ctx=nctx
                ):
                    parts.append(token)
                answer = "".join(parts).strip() or "（模型未返回正文）"
            judge = _evaluate_case_result(
                case,
                merged_diag,
                sources,
                answer,
                final_ranked_sources=final_ranked_sources,
                use_rerank=bool(use_rerank),
                context_truncated=bool(context_truncated),
                generation_mode=generation_mode,
                no_context=no_context,
            )
            diag_payload["eval_case_diagnostics"] = {
                "issue_codes": list(judge.get("issue_codes") or []),
                "expected_file_resolution": dict(judge.get("expected_file_resolution") or {}),
                "matched_candidate_files": list(judge.get("matched_candidate_files") or []),
                "chunk_diagnostics": dict(judge.get("chunk_diagnostics") or {}),
                "filtered_out": bool(judge.get("filtered_out")),
                "final_ranked_hit": judge.get("final_ranked_hit"),
                "topk_or_rerank_drop": bool(judge.get("topk_or_rerank_drop")),
                "context_budget_drop": bool(judge.get("context_budget_drop")),
                "target_file_hit_but_chunk_miss": bool(judge.get("target_file_hit_but_chunk_miss")),
                "candidate_hit": judge.get("candidate_hit"),
                "context_hit": judge.get("context_hit"),
                "chunk_hit": judge.get("chunk_hit"),
                "answer_hit": judge.get("answer_hit"),
                "abstain_expected": bool(judge.get("abstain_expected")),
                "abstain_actual": bool(judge.get("abstain_actual")),
                "abstain_correct": judge.get("abstain_correct"),
                "abstain_with_target_context": bool(judge.get("abstain_with_target_context")),
                "error_type": str(judge.get("error_type") or ""),
                "retrieval_only": bool(judge.get("retrieval_only")),
                "no_context": bool(judge.get("no_context")),
            }
            diag_payload["retrieval_attribution"] = attribution_code(
                diagnostics=diag_payload,
                judge=judge,
                error_type=str(judge.get("error_type") or ""),
            )
            qa_id = store.insert_qa(
                question=qtext,
                answer=answer,
                sources=sources,
                params=params,
                diagnostics=diag_payload,
                session_id=None,
            )
            store.save_eval_case_result(
                run_id=run_id,
                case_id=int(case["id"]),
                question=qtext,
                answer=answer,
                sources=sources,
                diagnostics=diag_payload,
                candidate_hit=judge["candidate_hit"],
                context_hit=judge["context_hit"],
                chunk_hit=judge["chunk_hit"],
                answer_hit=judge["answer_hit"],
                abstain_expected=judge["abstain_expected"],
                abstain_actual=judge["abstain_actual"],
                abstain_correct=judge["abstain_correct"],
                error_type=str(judge["error_type"] or ""),
                qa_id=qa_id,
            )
            attribution = str(diag_payload.get("retrieval_attribution") or "")
            abstain_cell = (
                "—"
                if judge.get("abstain_correct") is None
                else ("正确" if judge["abstain_correct"] else "错误")
            )
            result_rows.append(
                [
                    idx,
                    qtext,
                    "命中" if judge["candidate_hit"] else ("—" if judge["candidate_hit"] is None else "未命中"),
                    "命中" if judge["context_hit"] else ("—" if judge["context_hit"] is None else "未命中"),
                    "命中" if judge["chunk_hit"] else ("—" if judge["chunk_hit"] is None else "未命中"),
                    "命中" if judge["answer_hit"] else ("—" if judge["answer_hit"] is None else "未命中"),
                    abstain_cell,
                    str(judge["error_type"] or ""),
                    attribution,
                    str(sources[0].get("file_name") or "—") if sources else "—",
                    qa_id,
                ]
            )
            degrade_note = f", degraded={kind}" if retrieval_result.retrieval_degraded else ""
            log_lines.append(
                f"    candidates {len(retrieval_result.merged_nodes)} "
                f"(vector {len(retrieval_result.vector_nodes)} / keyword {len(retrieval_result.keyword_nodes)}), "
                f"excluded {excluded_candidate_count}, context {len(nodes_for_answer)}, "
                f"verdict={judge['error_type']}{degrade_note}, QA#{qa_id}"
            )
    except Exception:
        run_error = traceback.format_exc()
        log_lines.append("评测运行异常中断，已保存已完成样本并写入当前汇总。")
        log_lines.append(run_error)

    if run_vector_degraded:
        try:
            params = store.patch_eval_run_params(
                run_id,
                {
                    "retrieval_degraded": True,
                    "vector_error": "one_or_more_cases_vector_stage_failed",
                },
            )
            log_lines.append("警告：向量检索曾失败，本 RUN 已标记 retrieval_degraded（score_kind=keyword_fallback/vector_error）。")
        except Exception:
            params["retrieval_degraded"] = True
            log_lines.append("警告：向量检索曾失败，但回写 RUN 参数失败；汇总仍会尽量标记降级。")

    if not result_rows:
        result_rows = _empty_eval_result_rows()
    summary_rows = _finalize_eval_run_summary(run_id)
    if run_error:
        summary_rows.append(["运行状态", "异常中断（部分结果已保存）"])
    try:
        compare_path = write_eval_case_compare(store, run_id, cfg.data_dir / "exports")
        log_lines.append(f"已导出逐题对比表（原题+评测结果）：{compare_path}")
        summary_rows.append(["逐题对比表", str(compare_path)])
    except Exception as export_exc:
        log_lines.append(f"逐题对比表导出失败：{type(export_exc).__name__}: {export_exc}")
    recent_rows = _eval_run_summary_rows_for_dataset(dataset_id)
    run_rows = store.list_eval_runs(limit=200)
    run = next((r for r in run_rows if int(r.get("id") or 0) == int(run_id)), None)
    param_rows = _params_rows_from_run(run)
    if run_error:
        param_rows.append(["run_error", run_error])
    error_rows = _eval_error_rows_for_run(run_id)
    tag_rows = _eval_tag_rows_for_run(run)
    file_rows = _eval_file_rows_for_run(run)
    chunk_diag_rows = _eval_chunk_diag_rows_for_run(run_id)
    funnel_rows = _eval_funnel_rows_for_run(run_id)
    return "\n".join(log_lines), result_rows, summary_rows, recent_rows, param_rows, error_rows, tag_rows, file_rows, chunk_diag_rows, funnel_rows


def build_ui():
    c = cfg.chunking
    r = cfg.retrieval
    ingest = cfg.ingest

    def _image_enrichment_from_prefs() -> bool:
        p = prefs_mod.load_prefs(cfg.data_dir)
        if "image_enrichment" in p:
            return bool(p["image_enrichment"])
        return bool(ingest.get("image_enrichment", False))

    _rag_css = f"""
    /* 避免右侧悬浮控件贴窗口边缘被裁切 */
    .gradio-container {{
        padding-left: 0.35rem !important;
        padding-right: 0.75rem !important;
        box-sizing: border-box !important;
    }}
    .rag-workbench-header {{
        margin-bottom: 0.1rem !important;
    }}
    .rag-workbench-header h1 {{
        font-size: 1.35rem;
        font-weight: 650;
        letter-spacing: -0.02em;
        margin: 0 !important;
    }}
    .rag-header-title-row {{
        display: flex;
        align-items: center;
        gap: 0.45rem;
        flex-wrap: nowrap;
        margin-bottom: 0;
    }}
    .rag-header-title-row h1 {{
        margin: 0 !important;
        flex-shrink: 0;
    }}
    /* 主 Tab：与内容区节奏统一 */
    .rag-main-tabs {{
        margin-top: 0.15rem !important;
    }}
    .rag-main-tabs > div:first-child {{
        margin-bottom: 0.3rem !important;
    }}
    /* 知识库三列：列间距与浅色分区，避免挤成一团 */
    .rag-kb-main-row {{
        gap: 0.65rem !important;
        align-items: flex-start !important;
        min-height: {RAG_KB_MAIN_MIN_HEIGHT_PX}px;
    }}
    .rag-kb-col-left,
    .rag-kb-col-center,
    .rag-kb-col-right {{
        padding: 0.1rem 0.2rem 0.25rem 0.2rem;
        border-radius: 8px;
        border: 1px solid #eef2f7;
        background: #fafbfc;
        box-sizing: border-box;
    }}
    /* 三列首行标题视觉对齐（左/中/右首块） */
    .rag-kb-col-left .rag-kb-section-head,
    .rag-kb-col-center .rag-kb-center-md,
    .rag-kb-col-right .rag-embed-model-stack {{
        margin-top: 0 !important;
    }}
    .rag-kb-col-right .rag-embed-model-row {{
        align-items: center !important;
    }}
    /* 「!」浮层：checkbox+label；.rag-tip-pop 为 fixed，top/left 由 launch(head) 内联脚本写在感叹号右侧 */
    .rag-tip-anchor {{
        position: relative;
        display: inline-flex;
        align-items: center;
        flex-shrink: 0;
    }}
    .rag-tip-cb {{
        position: absolute !important;
        opacity: 0 !important;
        width: 0 !important;
        height: 0 !important;
        margin: 0 !important;
        pointer-events: none !important;
    }}
    .rag-tip-trigger {{
        display: inline-flex !important;
        align-items: center;
        justify-content: center;
        border-radius: 50%;
        border: 1px solid #94a3b8;
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
        color: #475569;
        font-weight: 800;
        font-family: inherit;
        cursor: pointer;
        flex-shrink: 0;
        box-sizing: border-box;
        line-height: 1;
        user-select: none;
        padding: 0;
        margin: 0;
        appearance: none;
        -webkit-appearance: none;
    }}
    .rag-tip-trigger--page {{
        width: 1.35rem;
        height: 1.35rem;
        font-size: 0.8rem;
    }}
    .rag-tip-trigger--h3 {{
        width: 1.12rem;
        height: 1.12rem;
        font-size: 0.68rem;
    }}
    .rag-tip-trigger--h5 {{
        width: 1rem;
        height: 1rem;
        font-size: 0.62rem;
    }}
    .rag-tip-pop {{
        display: none;
        position: fixed;
        margin: 0;
        padding: 0;
        border: none;
        background: transparent;
        min-width: 240px;
        max-width: min(420px, calc(100vw - 16px));
        width: max-content;
        max-height: min(80vh, 520px);
        overflow-x: hidden;
        overflow-y: auto;
        box-sizing: border-box;
        box-shadow: 0 12px 40px rgba(15, 23, 42, 0.14);
        border-radius: 10px;
        text-align: left;
        z-index: 2147483000;
        left: 0;
        top: 0;
    }}
    .rag-tip-anchor:has(.rag-tip-cb:checked) .rag-tip-pop {{
        display: block;
    }}
    .rag-tip-pop--wide {{
        min-width: min(400px, calc(100vw - 16px));
        max-width: min(520px, calc(100vw - 16px));
    }}
    .rag-tip-block--in-pop {{
        margin: 0 !important;
        border-radius: 10px;
        border-left-width: 3px;
        flex: none !important;
    }}
    .rag-kb-section-head {{
        display: flex;
        align-items: center;
        gap: 0.35rem;
        flex-wrap: nowrap;
        margin: 0 0 0.35rem 0;
    }}
    .rag-kb-section-head h3 {{
        margin: 0 !important;
        font-size: 1.15rem !important;
        font-weight: 600 !important;
        line-height: 1.35 !important;
    }}
    .rag-kb-section-head h5 {{
        margin: 0 !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        line-height: 1.35 !important;
    }}
    .rag-kb-embed-head {{
        display: flex;
        align-items: center;
        gap: 0.35rem;
        flex-wrap: wrap;
        margin: 0 !important;
        padding: 0 !important;
    }}
    .rag-kb-col-right .rag-embed-model-stack {{
        display: flex;
        flex-direction: column;
        gap: 0.35rem;
        width: 100%;
        min-width: 0;
    }}
    .rag-kb-col-right .rag-embed-model-stack .rag-kb-embed-head-wrap {{
        flex: 0 0 auto !important;
        min-height: 0 !important;
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
        width: 100%;
    }}
    .rag-kb-col-right .rag-embed-model-stack .rag-kb-embed-head {{
        flex-wrap: nowrap !important;
    }}
    .rag-kb-embed-title-col {{
        min-width: 0 !important;
        flex: 0 1 auto !important;
        align-self: center !important;
    }}
    .rag-kb-embed-title-col .rag-kb-embed-head {{
        margin: 0 !important;
        padding: 0 !important;
        flex-wrap: nowrap !important;
    }}
    .rag-kb-embed-title-col .rag-kb-embed-label {{
        font-size: 0.82rem !important;
        white-space: nowrap;
    }}
    .rag-kb-embed-label {{
        font-size: 0.875rem;
        font-weight: 500;
        color: #374151;
        line-height: 1.35;
    }}
    /* 全站小提示：灰底条，与页头说明一致 */
    .rag-tip-block {{
        font-size: 0.88rem;
        line-height: 1.5;
        color: #475569;
        padding: 0.4rem 0.65rem;
        margin: 0 0 0.5rem 0;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        border-left: 3px solid #94a3b8;
        box-sizing: border-box;
    }}
    .rag-tip-block p {{
        margin: 0.35em 0;
        line-height: 1.55;
    }}
    .rag-tip-block p:first-child {{ margin-top: 0; }}
    .rag-tip-block p:last-child {{ margin-bottom: 0; }}
    .rag-tip-block--header {{
        flex: 1;
        min-width: 0;
        margin-bottom: 0;
    }}
    .rag-tip-block--tight {{
        padding: 0.32rem 0.55rem;
        margin-bottom: 0.4rem;
        font-size: 0.82rem;
    }}
    .rag-workbench-tip {{ opacity: 0.9; font-size: 0.92em; margin-top: 0.45em !important; }}
    /* 对话输入行：避免 secondary 按钮默认 lg 高度高于单行输入框 */
    .rag-chat-input-row {{
        align-items: center !important;
    }}
    .rag-chat-input-row > div {{
        align-self: center !important;
    }}
    .rag-chat-input-row button {{
        min-height: 2.25rem !important;
        height: 2.25rem !important;
        max-height: 2.25rem !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
        box-sizing: border-box !important;
    }}
    /* 对话页三列：与中间「Chatbot + 引用 + 输入」总高对齐 */
    .rag-chat-main-row {{
        align-items: stretch !important;
    }}
    .rag-chat-col-left {{
        display: flex;
        flex-direction: column;
        min-height: 0;
        padding-top: 0 !important;
    }}
    /* 人工评估：约占页面宽度 3/4，居中 */
    .rag-eval-wrap {{
        max-width: 75%;
        width: 100%;
        margin-left: auto;
        margin-right: auto;
        margin-top: 0.35rem;
        box-sizing: border-box;
    }}
    /* 历史会话表可视区域与中间 Chatbot 同高 */
    .rag-chat-col-left .rag-session-table {{
        min-height: {RAG_CHATBOT_HEIGHT_PX}px;
        max-height: {RAG_CHATBOT_HEIGHT_PX}px;
    }}
    .rag-session-table .ag-icon-lock,
    .rag-session-table .ag-header-cell .ag-header-cell-menu-button {{
        display: none !important;
    }}
    .rag-session-table .ag-row-selected,
    .rag-session-table .ag-row-focus,
    .rag-session-table .ag-row:has(.ag-cell-focus) {{
        background-color: rgba(21, 101, 192, 0.12) !important;
    }}
    .rag-session-table .ag-row-selected .ag-cell,
    .rag-session-table .ag-row-focus .ag-cell,
    .rag-session-table .ag-row:has(.ag-cell-focus) .ag-cell {{
        background-color: transparent !important;
    }}
    .rag-session-table .ag-cell-focus {{
        border: 1px solid transparent !important;
        outline: none !important;
    }}
    .rag-session-table .ag-cell {{
        line-height: 1.35 !important;
        padding-top: 6px !important;
        padding-bottom: 6px !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        white-space: nowrap !important;
    }}
    .rag-session-table .ag-header-cell-text {{
        white-space: normal !important;
        line-height: 1.2 !important;
        font-size: 0.82rem !important;
    }}
    /* 历史会话标题行：左侧标题+!，右侧「新建会话」主按钮 */
    .rag-chat-session-toolbar {{
        align-items: center !important;
        width: 100% !important;
        flex-wrap: nowrap !important;
    }}
    .rag-chat-session-toolbar .rag-chat-session-head-col {{
        min-width: 0 !important;
        flex: 1 1 auto !important;
    }}
    .rag-chat-new-session-btn {{
        min-height: 2.25rem !important;
        height: 2.25rem !important;
        max-height: 2.25rem !important;
        flex-shrink: 0 !important;
        padding-left: 0.75rem !important;
        padding-right: 0.75rem !important;
        font-size: 0.875rem !important;
        box-sizing: border-box !important;
    }}
    .rag-chat-col-middle {{
        display: flex;
        flex-direction: column;
        min-width: 0;
        min-height: 0;
        gap: 0.3rem;
        padding-right: 0.25rem;
        box-sizing: border-box;
    }}
    /* Chatbot：为纵向滚动条预留间隙，避免复制键与滚条重叠 */
    .rag-chatbot-panel,
    .rag-chatbot-panel .wrap,
    .rag-chatbot-panel [class*="scroll"] {{
        scrollbar-gutter: stable;
    }}
    .rag-chatbot-panel {{
        padding-right: 14px !important;
        box-sizing: border-box !important;
    }}
    .rag-chat-col-middle .rag-chatbot-panel .message-wrap,
    .rag-chat-col-middle .rag-chatbot-panel [class*="message-row"] {{
        padding-right: 6px !important;
    }}
    .rag-chat-col-middle .rag-chatbot-panel .user,
    .rag-chat-col-middle .rag-chatbot-panel [class*="user"] {{
        margin-right: 4px !important;
        padding-right: 8px !important;
    }}
    .rag-chat-col-middle .rag-chatbot-panel button {{
        flex-shrink: 0 !important;
        margin-right: 4px !important;
    }}
    /* 引用区限高，避免中间列无限拉高导致右侧显得「过长」 */
    .rag-sources-panel {{
        max-height: {RAG_SOURCES_PANEL_MAX_PX}px;
        overflow-y: auto;
        min-height: 3rem;
    }}
    /* 右侧设置区不超出中间输入条下沿：总高与中间列一致，内部滚动 */
    .rag-chat-col-right {{
        display: flex;
        flex-direction: column;
        min-height: 0;
        max-height: {RAG_CHAT_MIDDLE_STACK_PX}px;
        overflow-y: auto;
        overflow-x: hidden;
        overscroll-behavior: contain;
        gap: 0.3rem;
        padding-right: 0.15rem;
    }}
    .rag-chat-col-right .rag-chat-right-md,
    .rag-chat-col-right .rag-chat-right-md * {{
        margin-top: 0 !important;
        margin-bottom: 0.25rem !important;
    }}
    .rag-chat-col-right .rag-system-prompt-box textarea {{
        max-height: 12rem !important;
    }}
    .rag-chat-col-right .rag-llm-num-ctx {{
        flex: 0 0 auto;
    }}
    /* 右侧「生成与检索」：下拉/滑块等标签不要用主题大号加粗蓝字，与整页协调 */
    .rag-chat-col-right label.block-label,
    .rag-chat-col-right label.block-label > span,
    .rag-chat-col-right .label-wrap label,
    .rag-chat-col-right .label-wrap .label-text,
    .rag-chat-col-right .wrap > label {{
        font-size: 0.8125rem !important;
        font-weight: 400 !important;
        color: #4b5563 !important;
        line-height: 1.35 !important;
    }}
    .rag-chat-col-right .rag-chat-right-md h1,
    .rag-chat-col-right .rag-chat-right-md h2,
    .rag-chat-col-right .rag-chat-right-md h3,
    .rag-chat-col-right .rag-chat-right-md h4,
    .rag-chat-col-right .rag-chat-right-md h5,
    .rag-chat-col-right .rag-chat-right-md h6 {{
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        color: #374151 !important;
        line-height: 1.35 !important;
    }}
    dialog.rag-src-dialog::backdrop {{
        background: rgba(15, 23, 42, 0.35);
    }}
    /* 知识库 2:5:3 三列：等高拉伸；右侧过长时列内滚动 */
    .rag-kb-col-left,
    .rag-kb-col-center,
    .rag-kb-col-right {{
        display: flex;
        flex-direction: column;
        min-height: 0;
        gap: 0.2rem;
    }}
    .rag-kb-col-center .rag-kb-center-md,
    .rag-kb-col-center .rag-kb-center-md * {{
        margin-top: 0 !important;
        margin-bottom: 0.2rem !important;
    }}
    .rag-kb-col-left .rag-kb-files-table {{
        max-height: {RAG_KB_FILES_TABLE_MAX_PX}px;
        min-height: 180px;
    }}
    .rag-kb-chunk-preview {{
        max-height: 360px;
        overflow-y: auto;
        overflow-x: hidden;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 10px 12px;
        background: #fafbfc;
        box-sizing: border-box;
    }}
    .rag-kb-col-left .rag-kb-build-log,
    #rag_kb_build_log.rag-kb-build-log {{
        flex: 0 0 auto !important;
        min-height: 0 !important;
        overflow: hidden !important;
        max-height: calc({RAG_KB_BUILD_LOG_TEXTAREA_PX}px + 4.5rem) !important;
    }}
    .rag-kb-col-left .rag-kb-build-log .wrap,
    .rag-kb-col-left .rag-kb-build-log .form,
    #rag_kb_build_log .wrap,
    #rag_kb_build_log .form,
    #rag_kb_build_log .container {{
        overflow: hidden !important;
        min-height: 0 !important;
        max-height: none !important;
    }}
    /* 构建日志：Gradio 5 Textbox 在 lines≠max_lines 时会 JS 按内容改 height；须 lines==max_lines 关闭该行为 */
    .rag-kb-col-left .rag-kb-build-log textarea,
    #rag_kb_build_log textarea,
    #rag_kb_build_log textarea[data-testid="textbox"] {{
        height: {RAG_KB_BUILD_LOG_TEXTAREA_PX}px !important;
        min-height: {RAG_KB_BUILD_LOG_TEXTAREA_PX}px !important;
        max-height: {RAG_KB_BUILD_LOG_TEXTAREA_PX}px !important;
        overflow-y: auto !important;
        resize: none !important;
        box-sizing: border-box !important;
    }}
    /* 构建日志标题行：长文案 + 复制键同一行，避免复制键被 overflow 裁掉 */
    .rag-kb-col-left .rag-kb-build-log .panel-header,
    .rag-kb-col-left .rag-kb-build-log .label-wrap,
    .rag-kb-col-left .rag-kb-build-log label.block-label {{
        display: flex !important;
        align-items: center !important;
        gap: 0.35rem !important;
        flex-wrap: nowrap !important;
        overflow: visible !important;
        padding-right: 0.15rem !important;
        box-sizing: border-box !important;
    }}
    .rag-kb-col-left .rag-kb-build-log .copy-button,
    .rag-kb-col-left .rag-kb-build-log button.copy-code-button,
    .rag-kb-col-left .rag-kb-build-log label.block-label button {{
        flex-shrink: 0 !important;
        align-self: center !important;
    }}
    .rag-kb-col-left .rag-kb-build-log label.block-label > span,
    .rag-kb-col-left .rag-kb-build-log .label-text {{
        flex: 1 1 auto !important;
        min-width: 0 !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }}
    /* 构建日志：仅「构建日志」四字，与正文协调，去掉主题标签高亮底 */
    .rag-kb-col-left .rag-kb-build-log label.block-label,
    .rag-kb-col-left .rag-kb-build-log label.block-label > span {{
        font-size: 0.8125rem !important;
        font-weight: 400 !important;
        color: #4b5563 !important;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        border-radius: 0 !important;
    }}
    /* 中间列：上传区（压缩空白，仍保证可点选） */
    .rag-kb-upload-zone {{
        flex: 0 0 auto !important;
        min-height: 120px !important;
        width: 100% !important;
        overflow: visible !important;
    }}
    .rag-kb-col-center .rag-kb-file-upload,
    .rag-kb-upload-zone .rag-kb-file-upload {{
        min-height: 110px !important;
        flex: 0 0 auto !important;
    }}
    #rag_kb_file_upload {{
        min-height: 120px !important;
    }}
    .rag-kb-notice textarea {{
        max-height: 4.25rem !important;
        min-height: 2.5rem !important;
        line-height: 1.35 !important;
    }}
    .rag-kb-upload-zone .file-preview,
    .rag-kb-upload-zone [class*="upload"] {{
        min-height: 88px !important;
    }}
    .rag-kb-col-center .rag-kb-center-actions {{
        margin-top: auto;
        padding-top: 0.2rem;
    }}
    .rag-kb-col-right {{
        max-height: min(100%, {RAG_KB_MAIN_MIN_HEIGHT_PX}px);
        overflow-y: auto;
        overscroll-behavior: contain;
        gap: 0.2rem !important;
    }}
    /* 切分策略 / Chunk 参数：与嵌入说明一致，弱化主题大号加粗蓝标签 */
    .rag-kb-col-right label.block-label,
    .rag-kb-col-right label.block-label > span,
    .rag-kb-col-right .label-wrap label,
    .rag-kb-col-right .label-wrap .label-text,
    .rag-kb-col-right .wrap > label {{
        font-size: 0.8125rem !important;
        font-weight: 400 !important;
        color: #4b5563 !important;
        line-height: 1.35 !important;
    }}
    /* Radio 选项：缩小字号、常规字重；选中态改为浅底深字，避免大块饱和蓝底 */
    .rag-kb-col-right label:has(input[type="radio"]) {{
        font-size: 0.8125rem !important;
        font-weight: 400 !important;
        color: #374151 !important;
    }}
    .rag-kb-col-right label:has(input[type="radio"]:checked) {{
        font-weight: 500 !important;
        color: #1e3a8a !important;
        background: #eff6ff !important;
        border-color: #bfdbfe !important;
    }}
    /* 嵌入模型行：下拉与刷新按钮垂直居中对齐、同高 */
    .rag-embed-model-row {{
        align-items: center !important;
    }}
    .rag-embed-model-row > div {{
        align-self: center !important;
    }}
    .rag-embed-model-row .rag-embed-refresh-btn {{
        min-height: 2.5rem !important;
        height: 2.5rem !important;
        max-height: 2.5rem !important;
        padding: 0 0.65rem !important;
        font-size: 0.8125rem !important;
        line-height: 1.2 !important;
        box-sizing: border-box !important;
    }}
    /* 文档表：interactive + static_columns 以便整行选中；隐藏静态列锁标 */
    .rag-kb-files-table .ag-icon-lock,
    .rag-kb-files-table .ag-header-cell .ag-header-cell-menu-button {{
        display: none !important;
    }}
    .rag-kb-files-table .ag-row-selected,
    .rag-kb-files-table .ag-row-focus,
    .rag-kb-files-table .ag-row:has(.ag-cell-focus) {{
        background-color: rgba(21, 101, 192, 0.12) !important;
    }}
    .rag-kb-files-table .ag-row-selected .ag-cell,
    .rag-kb-files-table .ag-row-focus .ag-cell,
    .rag-kb-files-table .ag-row:has(.ag-cell-focus) .ag-cell {{
        background-color: transparent !important;
    }}
    .rag-kb-files-table .ag-cell-focus {{
        border: 1px solid transparent !important;
        outline: none !important;
    }}
    /* 文档表：单行省略 + 表体横向滚动，避免窄列里文字竖着堆 */
    .rag-kb-col-left .rag-kb-files-table .ag-body-viewport,
    .rag-kb-col-left .rag-kb-files-table .ag-center-cols-viewport {{
        overflow-x: auto !important;
    }}
    .rag-kb-files-table .ag-center-cols-container {{
        min-width: 520px !important;
    }}
    .rag-kb-files-table .ag-cell {{
        line-height: 1.35 !important;
        padding-top: 8px !important;
        padding-bottom: 8px !important;
        align-items: center !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        white-space: nowrap !important;
    }}
    .rag-kb-files-table .ag-header-cell-text,
    .rag-kb-files-table .ag-header-cell-label {{
        white-space: normal !important;
        line-height: 1.15 !important;
        font-size: 0.8rem !important;
    }}
    /* 避免单元格内弹出层/展开改变行高：去掉截断点击展开（已取消 max_chars） */
    .rag-kb-files-table .ag-cell-value,
    .rag-kb-files-table .ag-popup {{
        max-height: none !important;
    }}
    .rag-kb-col-left .rag-kb-build-log textarea {{
        min-height: 160px;
    }}
    """

    _theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate",
    )

    def _persist_llm_choice(name: str):
        p = prefs_mod.load_prefs(cfg.data_dir)
        p["llm_model"] = name
        prefs_mod.save_prefs(cfg.data_dir, p)

    def _persist_llm_num_ctx(value: float):
        try:
            v = int(round(float(value)))
        except (TypeError, ValueError):
            v = 16384
        v = max(2048, min(v, 262144))
        p = prefs_mod.load_prefs(cfg.data_dir)
        p["llm_num_ctx"] = v
        prefs_mod.save_prefs(cfg.data_dir, p)

    def _persist_embed_choice(name: str):
        p = prefs_mod.load_prefs(cfg.data_dir)
        p["embed_model"] = name
        prefs_mod.save_prefs(cfg.data_dir, p)
        w = kb_embed_warning_markdown(name or "")
        return (
            kb_files_table_value(name or ""),
            gr.update(value=w, visible=bool(w)),
            gr.update(value=_KB_CHUNK_PREVIEW_EMPTY_HTML),
        )

    def _persist_image_enrichment(on: bool):
        p = prefs_mod.load_prefs(cfg.data_dir)
        p["image_enrichment"] = bool(on)
        prefs_mod.save_prefs(cfg.data_dir, p)

    def _persist_image_vision(name: str):
        p = prefs_mod.load_prefs(cfg.data_dir)
        p["image_vision_model"] = str(name or "").strip()
        prefs_mod.save_prefs(cfg.data_dir, p)

    def _persist_image_lang(text: str):
        p = prefs_mod.load_prefs(cfg.data_dir)
        p["image_tesseract_lang"] = str(text or "").strip()
        prefs_mod.save_prefs(cfg.data_dir, p)

    def _persist_image_ocr_skip(val: float):
        try:
            v = max(0, min(500, int(round(float(val)))))
        except (TypeError, ValueError):
            v = 20
        p = prefs_mod.load_prefs(cfg.data_dir)
        p["image_ocr_skip_vlm_min_chars"] = v
        prefs_mod.save_prefs(cfg.data_dir, p)

    def _persist_image_ocr_engine(val: str):
        p = prefs_mod.load_prefs(cfg.data_dir)
        s = str(val or "").strip().lower()
        p["image_ocr_engine"] = s if s in ("tesseract", "paddleocr") else "tesseract"
        prefs_mod.save_prefs(cfg.data_dir, p)

    def _refresh_models_action():
        llm_c, emb_c, cll, cem = refresh_model_choices()
        vm_cur, _, _, _ = _image_pipeline_prefs_merged()
        vis_c = _vision_model_choices_for_ui(vm_cur)
        return (
            gr.update(choices=llm_c, value=cll),
            gr.update(choices=emb_c, value=cem),
            gr.update(choices=vis_c, value=vm_cur),
        )

    def _after_build_refresh(embed_sel: str):
        return kb_files_table_value(embed_sel or "")

    def _on_app_load():
        llm_c, emb_c, cur_llm, cur_emb = refresh_model_choices()
        store.ensure_default_session()
        if not _all_session_labels():
            store.create_session("默认会话")
        sid = store.ensure_default_session()
        hist = session_history_for_chatbot(sid)
        kb = kb_files_table_value(cur_emb or "")
        warn = kb_embed_warning_markdown(cur_emb or "")
        vm_cur, lang_cur, skip_cur, ocr_cur = _image_pipeline_prefs_merged()
        vis_c = _vision_model_choices_for_ui(vm_cur)
        return (
            gr.update(choices=llm_c, value=cur_llm),
            gr.update(choices=emb_c, value=cur_emb),
            gr.update(value=_sessions_table_value()),
            hist,
            sid,
            kb,
            gr.update(value=warn, visible=bool(warn)),
            gr.update(value=initial_llm_num_ctx_for_ui()),
            gr.update(value=_image_enrichment_from_prefs()),
            gr.update(value=ocr_cur),
            gr.update(value=lang_cur),
            gr.update(choices=vis_c, value=vm_cur),
            gr.update(value=float(skip_cur)),
            gr.update(value=kb_chroma_dir_redirect_warning_markdown(),
                       visible=bool(kb_chroma_dir_redirect_warning_markdown())),
            gr.update(visible=bool(kb_chroma_dir_redirect_warning_markdown())),
        )

    def _on_session_table_select(evt: gr.SelectData):
        """点击列表行切换会话（与 list_sessions 行序一致）。"""
        if not evt.selected:
            return gr.update(), gr.update()
        idx = evt.index
        if isinstance(idx, (list, tuple)) and len(idx) >= 1:
            row_i = int(idx[0])
        else:
            row_i = int(idx)
        rv = evt.row_value
        if rv and isinstance(rv, list) and str(rv[0]).startswith("（暂无"):
            return gr.update(), gr.update()
        rows = store.list_sessions(limit=400)
        if row_i < 0 or row_i >= len(rows):
            return gr.update(), gr.update()
        sid = int(rows[row_i]["id"])
        return session_history_for_chatbot(sid), sid

    def _on_new_session():
        sid = store.create_session(None)
        return [], sid, gr.update(value=_sessions_table_value())

    def _on_kb_file_select(evt: gr.SelectData):
        """只存表格行号；预览/删除时用当前磁盘快照解析真实文件名，避免与向量库元数据字符串不一致。"""
        disk = uploaded_files_snapshot(cfg)
        if not getattr(evt, "selected", False) or not disk:
            return None
        idx = evt.index
        r0: int | None = None
        r1: int | None = None
        if isinstance(idx, (list, tuple)) and len(idx) >= 1:
            r0 = int(idx[0])
            r1 = int(idx[1]) if len(idx) >= 2 else None
        else:
            r0 = int(idx)
        if r0 is not None and 0 <= r0 < len(disk):
            return r0
        if r1 is not None and 0 <= r1 < len(disk):
            return r1
        rv = getattr(evt, "row_value", None)
        if rv is not None:
            cells = list(rv) if isinstance(rv, (list, tuple)) else [rv]
            if cells:
                s0 = str(cells[0]).strip()
                if s0.startswith("（暂无"):
                    return None
                guess = _canonical_upload_filename(s0, disk)
                if guess:
                    for i, f in enumerate(disk):
                        if f["name"] == guess:
                            return i
        return None

    with gr.Blocks(
        title="RAG 验证工作台",
        theme=_theme,
        css=_rag_css,
        head=_rag_tip_head_script(),
    ) as demo:
        gr.HTML(_RAG_HEADER_HTML)

        session_id = gr.State(value=store.ensure_default_session())

        # 须先于「对话」Tab 创建 chunk_* / chunk_mode / embed_dd，供事件绑定引用
        with gr.Tabs(selected="chat", elem_classes=["rag-main-tabs"]):
            with gr.Tab("知识库", id="kb"):
                llm_c0, emb_c0, _, cur_emb0 = refresh_model_choices()
                _vm0, _lang0, _skip0, _ocr0 = _image_pipeline_prefs_merged()
                _vision_c0 = _vision_model_choices_for_ui(_vm0)
                _kb_warn0 = kb_embed_warning_markdown(cur_emb0 or "")
                kb_file_sel = gr.State(value=None)
                with gr.Row(equal_height=False, elem_classes=["rag-kb-main-row"]):
                    with gr.Column(scale=3, min_width=300, elem_classes=["rag-kb-col-left"]):
                        gr.HTML(_KB_SECTION_UPLOADED_HTML)
                        kb_files_df = gr.Dataframe(
                            headers=["文件名", "大小", "上传时间", "向量时间", "索引状态"],
                            value=kb_files_table_value(cur_emb0 or ""),
                            label="已上传文档",
                            show_label=False,
                            interactive=True,
                            static_columns=[0, 1, 2, 3, 4],
                            col_count=(5, "fixed"),
                            max_height=RAG_KB_FILES_TABLE_MAX_PX,
                            wrap=False,
                            column_widths=["42%", "8%", "14%", "14%", "22%"],
                            type="array",
                            elem_classes=["rag-kb-files-table"],
                        )
                        with gr.Row():
                            btn_remove = gr.Button("移除选中文件", variant="secondary")
                            btn_preview_chunks = gr.Button("预览切片", variant="secondary")
                        kb_chunk_preview = gr.HTML(
                            value=_KB_CHUNK_PREVIEW_EMPTY_HTML,
                            elem_classes=["rag-kb-chunk-preview"],
                        )
                        build_msg = gr.Textbox(
                            label="构建日志",
                            # 与 max_lines 必须相等，否则 Gradio 前端会随内容把 textarea 撑高（覆盖 CSS）
                            lines=12,
                            max_lines=12,
                            autoscroll=True,
                            show_copy_button=True,
                            elem_id="rag_kb_build_log",
                            elem_classes=["rag-kb-build-log"],
                        )
                    with gr.Column(scale=4, min_width=280, elem_classes=["rag-kb-col-center"]):
                        gr.Markdown("##### 上传与构建", elem_classes=["rag-kb-center-md"])
                        with gr.Column(elem_classes=["rag-kb-upload-zone"]):
                            up = gr.File(
                                label="选择文件（可多选）",
                                file_count="multiple",
                                type="filepath",
                                height=140,
                                elem_classes=["rag-kb-file-upload"],
                                elem_id="rag_kb_file_upload",
                            )
                        with gr.Accordion("高级", open=False):
                            max_mb = gr.Number(
                                value=float(ingest.get("max_file_size_mb", 50)),
                                label="单文件大小上限 (MB)",
                            )
                            gr.Markdown("##### 图片检索", elem_classes=["rag-kb-center-md"])
                            kb_image_enrichment = gr.Checkbox(
                                value=_image_enrichment_from_prefs(),
                                label="是否开启图片检索",
                                info="勾选后，构建向量索引时会对 PDF/DOCX 内嵌图先 OCR；识别文字过少时再调用视觉模型补充。选项会保存到本地偏好。",
                                elem_classes=["rag-kb-image-enrichment"],
                            )
                            kb_image_ocr_engine = gr.Dropdown(
                                choices=[("Tesseract", "tesseract"), ("PaddleOCR", "paddleocr")],
                                value=_ocr0,
                                label="OCR 引擎",
                                info="PaddleOCR 需单独安装依赖（见项目内 requirements-paddleocr.txt）。「OCR 语言包」仅对 Tesseract 生效；Paddle 使用内置中英文模型。",
                                elem_classes=["rag-kb-image-pipeline-field"],
                            )
                            kb_image_tesseract_lang = gr.Textbox(
                                value=_lang0,
                                label="OCR 语言包（Tesseract）",
                                info="仅在使用 Tesseract 时作为 --lang；需本机已安装对应 traineddata。",
                                elem_classes=["rag-kb-image-pipeline-field"],
                            )
                            kb_image_vision_model = gr.Dropdown(
                                choices=_vision_c0,
                                value=_vm0,
                                label="视觉模型（Ollama）",
                                allow_custom_value=True,
                                elem_classes=["rag-kb-image-pipeline-field"],
                            )
                            kb_image_ocr_skip = gr.Number(
                                value=float(_skip0),
                                label="OCR 不少于该字符数则跳过视觉",
                                minimum=0,
                                maximum=500,
                                precision=0,
                                elem_classes=["rag-kb-image-pipeline-field"],
                            )
                        kb_notice = gr.Textbox(
                            label="上传提示",
                            lines=2,
                            max_lines=4,
                            elem_classes=["rag-kb-notice"],
                        )
                        with gr.Column(elem_classes=["rag-kb-center-actions"]):
                            with gr.Row():
                                btn_save = gr.Button("保存到上传目录", variant="secondary")
                                btn_build = gr.Button("构建向量索引", variant="primary")
                    with gr.Column(scale=3, min_width=260, elem_classes=["rag-kb-col-right"]):
                        with gr.Column(elem_classes=["rag-embed-model-stack"]):
                            gr.HTML(_KB_EMBED_HEAD_HTML, elem_classes=["rag-kb-embed-head-wrap"])
                            with gr.Row(equal_height=True, elem_classes=["rag-embed-model-row"]):
                                embed_dd = gr.Dropdown(
                                    choices=emb_c0,
                                    value=cur_emb0 or (emb_c0[0] if emb_c0 else ""),
                                    label="",
                                    show_label=False,
                                    interactive=len(emb_c0) > 1,
                                    allow_custom_value=True,
                                    scale=4,
                                    min_width=0,
                                )
                                btn_refresh_models = gr.Button(
                                    "刷新",
                                    variant="secondary",
                                    size="sm",
                                    scale=0,
                                    min_width=72,
                                    elem_classes=["rag-embed-refresh-btn"],
                                )
                        kb_embed_warn = gr.Markdown(
                            value=_kb_warn0,
                            visible=bool(_kb_warn0),
                            elem_classes=["rag-tip-block", "rag-tip-block--tight"],
                        )
                        with gr.Row(elem_classes=["rag-tip-block", "rag-tip-block--tight"]):
                            kb_chroma_redirect_warn = gr.Markdown(
                                value=kb_chroma_dir_redirect_warning_markdown(),
                                visible=bool(kb_chroma_dir_redirect_warning_markdown()),
                            )
                            kb_chroma_redirect_ack_btn = gr.Button(
                                "✓ 不再提示",
                                size="sm",
                                variant="secondary",
                                visible=bool(kb_chroma_dir_redirect_warning_markdown()),
                            )
                        kb_chroma_redirect_ack_btn.click(
                            do_ack_chroma_dir_redirect,
                            outputs=[kb_chroma_redirect_warn, kb_chroma_redirect_ack_btn],
                        )
                        chunk_mode = gr.Radio(
                            choices=[
                                ("按句切分（LlamaIndex Sentence）", "sentence"),
                                ("按 Token 切分", "token"),
                                ("段落优先（双换行再切）", "paragraph"),
                            ],
                            value=str(c.get("chunk_mode_default") or "sentence").strip().lower() or "sentence",
                            label="切分策略",
                        )
                        chunk_size = gr.Slider(
                            128,
                            2048,
                            value=int(c.get("chunk_size_default", 512)),
                            step=64,
                            label="Chunk 大小（sentence/paragraph 为 token 预算；token 模式同为 token）",
                        )
                        chunk_overlap = gr.Slider(
                            0,
                            512,
                            value=int(c.get("chunk_overlap_default", 64)),
                            step=32,
                            label="Chunk 重叠",
                        )
                        gr.Markdown(
                            "变更切分策略或大小后需**重新构建**。参数会写入问答记录。",
                            elem_classes=["rag-tip-block", "rag-tip-block--tight"],
                        )
                        with gr.Accordion("切片统计诊断", open=False):
                            btn_chunk_diag = gr.Button("刷新切片统计", variant="secondary", size="sm")
                            chunk_diag_summary = gr.HTML(value=_chunk_diag_summary_html({"summary": {}}))
                            chunk_diag_df = gr.Dataframe(
                                headers=["文件名", "入库状态", "块数", "平均字符", "最大字符", "空块", "图片提示块", "OCR 提示块", "视觉提示块", "文档分类", "分类依据", "文本页/总页", "可疑页占比%"],
                                value=[["（当前无切片统计）", "未知", 0, 0, 0, 0, 0, 0, 0, "—", "—", "0/0", 0]],
                                show_label=False,
                                interactive=False,
                                static_columns=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                                col_count=(13, "fixed"),
                                type="array",
                                wrap=False,
                                max_height=220,
                            )
                            gr.Markdown("##### Index Version Ops")
                            with gr.Row():
                                btn_index_ops = gr.Button("Refresh index diagnostics", variant="secondary", size="sm")
                                btn_index_cleanup_preview = gr.Button("Preview old-index cleanup", variant="secondary", size="sm")
                            index_ops_status = gr.Textbox(label="Cleanup preview", lines=2, max_lines=4)
                            index_ops_summary = gr.HTML(value=index_ops_summary_html(index_ops_diagnostics(cfg, store=store, keep_recent=3)))
                            index_ops_df = gr.Dataframe(
                                headers=["Directory", "Path", "Size"],
                                value=index_ops_rows(index_ops_diagnostics(cfg, store=store, keep_recent=3)),
                                show_label=False,
                                interactive=False,
                                static_columns=[0, 1, 2],
                                col_count=(3, "fixed"),
                                type="array",
                                wrap=False,
                                max_height=180,
                            )

                kb_files_df.select(_on_kb_file_select, outputs=kb_file_sel)
                btn_preview_chunks.click(
                    _kb_chunk_preview_html,
                    inputs=[kb_file_sel, embed_dd],
                    outputs=[kb_chunk_preview],
                )
                btn_chunk_diag.click(
                    do_chunk_diagnostics,
                    inputs=[embed_dd],
                    outputs=[chunk_diag_summary, chunk_diag_df],
                )
                btn_index_ops.click(
                    do_index_ops_diagnostics,
                    outputs=[index_ops_summary, index_ops_df],
                )
                btn_index_cleanup_preview.click(
                    do_index_cleanup_dry_run,
                    outputs=[index_ops_status, index_ops_summary, index_ops_df],
                )

                with gr.Row():
                    btn_index_cleanup_apply = gr.Button("\u771f\u6b63\u6267\u884c\u6e05\u7406\uff08\u4e0d\u53ef\u6062\u590d\uff09", variant="stop", size="sm")
                    cb_index_cleanup_confirm = gr.Checkbox(
                        label="\u6211\u5df2\u786e\u8ba4\u8981\u6e05\u7406\u4ee5\u4e0a\u76ee\u5f55\uff08\u4e0d\u53ef\u6062\u590d\uff09",
                        value=False,
                    )
                btn_index_cleanup_apply.click(
                    do_index_cleanup_apply,
                    [cb_index_cleanup_confirm],
                    [index_ops_status, index_ops_summary, index_ops_df],
                )

                btn_save.click(
                    do_save_uploads,
                    [up, max_mb, embed_dd],
                    [kb_notice, kb_files_df],
                )
                kb_image_enrichment.change(_persist_image_enrichment, kb_image_enrichment, None)
                kb_image_ocr_engine.change(_persist_image_ocr_engine, kb_image_ocr_engine, None)
                kb_image_vision_model.change(_persist_image_vision, kb_image_vision_model, None)
                kb_image_tesseract_lang.change(_persist_image_lang, kb_image_tesseract_lang, None)
                kb_image_ocr_skip.change(_persist_image_ocr_skip, kb_image_ocr_skip, None)
                b_chain = btn_build.click(
                    do_build_index,
                    [
                        chunk_size,
                        chunk_overlap,
                        chunk_mode,
                        embed_dd,
                        kb_image_enrichment,
                        kb_image_vision_model,
                        kb_image_tesseract_lang,
                        kb_image_ocr_skip,
                        kb_image_ocr_engine,
                    ],
                    build_msg,
                )
                b_chain.then(_after_build_refresh, embed_dd, [kb_files_df])
                btn_remove.click(
                    remove_kb_file,
                    [kb_file_sel, embed_dd],
                    [kb_notice, kb_files_df, kb_file_sel],
                )
                embed_dd.change(
                    _persist_embed_choice,
                    embed_dd,
                    [kb_files_df, kb_embed_warn, kb_chunk_preview],
                )

            with gr.Tab("对话", id="chat"):
                llm_c1, _, cur_llm1, _ = refresh_model_choices()
                with gr.Row(equal_height=True, elem_classes=["rag-chat-main-row"]):
                    with gr.Column(scale=1, min_width=300, elem_classes=["rag-chat-col-left"]):
                        with gr.Row(equal_height=True, elem_classes=["rag-chat-session-toolbar"]):
                            with gr.Column(scale=1, min_width=0, elem_classes=["rag-chat-session-head-col"]):
                                gr.HTML(_CHAT_SESSION_HEAD_HTML)
                            btn_new_session = gr.Button(
                                "新建会话",
                                variant="primary",
                                size="sm",
                                scale=0,
                                min_width=100,
                                elem_classes=["rag-chat-new-session-btn"],
                            )
                        session_table = gr.Dataframe(
                            headers=["会话", "更新时间"],
                            value=_sessions_table_value(),
                            label="历史会话",
                            show_label=False,
                            interactive=True,
                            static_columns=[0, 1],
                            col_count=(2, "fixed"),
                            max_height=RAG_CHATBOT_HEIGHT_PX,
                            wrap=False,
                            column_widths=["58%", "42%"],
                            type="array",
                            elem_classes=["rag-session-table"],
                        )
                        gr.HTML(_CHAT_EXPORT_HEAD_HTML)
                        fmt_export = gr.Radio(
                            choices=["json", "csv"],
                            value="json",
                            label="格式",
                        )
                        btn_export = gr.Button("导出 qa_log", variant="secondary", size="sm")
                        exp_path = gr.Textbox(
                            label="导出文件路径",
                            lines=2,
                            max_lines=4,
                        )
                    with gr.Column(scale=4, min_width=400, elem_classes=["rag-chat-col-middle"]):
                        chatbot = gr.Chatbot(
                            label="当前会话",
                            height=RAG_CHATBOT_HEIGHT_PX,
                            type="messages",
                            show_copy_button=True,
                            elem_classes=["rag-chatbot-panel"],
                            placeholder=(
                                "<div style=\"text-align:center;padding:1.2em;color:#6b7280;\">"
                                "选择或新建会话后提问。请先完成知识库索引构建。"
                                "</div>"
                            ),
                        )
                        with gr.Row(equal_height=True, elem_classes=["rag-chat-input-row"]):
                            msg = gr.Textbox(
                                placeholder="输入问题，Enter 发送",
                                show_label=False,
                                container=False,
                                scale=5,
                                lines=1,
                                max_lines=1,
                            )
                            send = gr.Button(
                                "发送",
                                variant="primary",
                                size="sm",
                                scale=0,
                                min_width=88,
                            )
                            btn_clear = gr.Button(
                                "清空本页",
                                variant="secondary",
                                size="sm",
                                scale=0,
                                min_width=96,
                            )
                        with gr.Column(elem_classes=["rag-sources-panel"], scale=1, min_width=0):
                            sources = gr.HTML(
                                label="引用片段与得分",
                                value=_SOURCES_EMPTY_HTML,
                            )
                            retrieval_diag = gr.HTML(
                                label="检索诊断",
                                value=_DIAG_EMPTY_HTML,
                            )
                    with gr.Column(scale=2, min_width=280, elem_classes=["rag-chat-col-right"]):
                        gr.Markdown("##### 生成与检索", elem_classes=["rag-chat-right-md"])
                        with gr.Accordion("系统提示词", open=True):
                            system_prompt = gr.Textbox(
                                value=str(cfg.prompt.get("system_default", "")).strip(),
                                lines=6,
                                label="System prompt",
                                show_label=False,
                                elem_classes=["rag-system-prompt-box"],
                            )
                        llm_dd = gr.Dropdown(
                            choices=llm_c1,
                            value=cur_llm1 or (llm_c1[0] if llm_c1 else ""),
                            label="对话模型 LLM（Ollama）",
                            interactive=len(llm_c1) > 1,
                            allow_custom_value=True,
                        )
                        # 用 Number 而非大块 Slider，避免右侧栏限高时把「检索参数」挤出视区
                        llm_num_ctx = gr.Number(
                            label="LLM 上下文窗口（num_ctx，Ollama）",
                            value=float(initial_llm_num_ctx_for_ui()),
                            minimum=1,
                            maximum=262144,
                            precision=0,
                            elem_classes=["rag-llm-num-ctx"],
                        )
                        llm_dd.change(_persist_llm_choice, llm_dd)
                        llm_num_ctx.change(_persist_llm_num_ctx, llm_num_ctx)
                        with gr.Accordion("检索参数", open=True):
                            top_n = gr.Slider(
                                1,
                                100,
                                value=int(r.get("top_n_default", 20)),
                                step=1,
                                label="向量初筛 Top-N",
                            )
                            top_k = gr.Slider(
                                1,
                                20,
                                value=int(r.get("top_k_default", 3)),
                                step=1,
                                label="送入上下文的 Top-K（≤ N）",
                            )
                            use_rerank = gr.Checkbox(
                                value=bool(cfg.rerank.get("enabled_default", False)),
                                label="启用 Cross-Encoder 重排（需 sentence-transformers，首次可能下载模型）",
                            )

                session_table.select(
                    _on_session_table_select,
                    outputs=[chatbot, session_id],
                )
                btn_new_session.click(_on_new_session, outputs=[chatbot, session_id, session_table])
                btn_export.click(do_export, [fmt_export], exp_path)

                with gr.Column(elem_classes=["rag-eval-wrap"]):
                    with gr.Accordion("人工评估（最近一次回答）", open=False):
                        with gr.Row():
                            rating = gr.Slider(1, 5, value=3, step=1, label="评分 1–5")
                            note = gr.Textbox(
                                label="备注",
                                placeholder="可选：错因、期望行为等",
                                scale=3,
                            )
                            btn_rate = gr.Button("保存评分", variant="secondary", scale=0)
                        rate_status = gr.Textbox(label="状态", lines=1, max_lines=3)

            with gr.Tab("评测", id="eval"):
                eval_llm_choices, eval_emb_choices, eval_cur_llm, eval_cur_emb = refresh_model_choices()
                eval_choices, eval_value = _eval_dataset_choices()
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5, min_width=420):
                        gr.Markdown("##### 评测集")
                        with gr.Row():
                            eval_file = gr.File(
                                label="导入评测集（json/csv/xlsx/xls）",
                                file_count="single",
                                file_types=[".json", ".csv", ".xlsx", ".xls"],
                                type="filepath",
                            )
                            eval_dataset_name = gr.Textbox(
                                label="评测集名称",
                                placeholder="可选：默认取文件名",
                            )
                            eval_dataset_desc = gr.Textbox(
                                label="说明",
                                placeholder="可选：评测集用途或范围",
                            )
                            btn_eval_import = gr.Button("导入评测集", variant="secondary")
                        eval_import_status = gr.Textbox(label="导入状态", lines=2, max_lines=4)
                        eval_dataset_dd = gr.Dropdown(
                            choices=eval_choices,
                            value=eval_value,
                            label="当前评测集",
                            allow_custom_value=False,
                        )
                        eval_dataset_summary = gr.Markdown(
                            value=_eval_dataset_summary_markdown(_parse_eval_dataset_id(eval_value))
                        )
                        eval_cases_df = gr.Dataframe(
                            headers=["#", "问题", "标准答案", "期望文件", "应命中片段", "答案关键词", "允许拒答", "标签"],
                            value=_eval_cases_preview_rows(
                                store.fetch_eval_cases(_parse_eval_dataset_id(eval_value))
                                if _parse_eval_dataset_id(eval_value) is not None
                                else []
                            ),
                            show_label=False,
                            interactive=False,
                            static_columns=[0, 1, 2, 3, 4, 5, 6, 7],
                            col_count=(8, "fixed"),
                            type="array",
                            wrap=False,
                            max_height=260,
                        )
                    with gr.Column(scale=4, min_width=360):
                        gr.Markdown("##### 实验运行")
                        eval_run_name = gr.Textbox(
                            label="运行名称",
                            placeholder="可选：默认自动生成",
                        )
                        eval_system_prompt = gr.Textbox(
                            value=str(cfg.prompt.get("system_default", "")).strip(),
                            lines=5,
                            label="System prompt",
                        )
                        with gr.Row():
                            eval_llm_dd = gr.Dropdown(
                                choices=eval_llm_choices,
                                value=eval_cur_llm or (eval_llm_choices[0] if eval_llm_choices else ""),
                                label="对话模型",
                                allow_custom_value=True,
                            )
                            eval_embed_dd = gr.Dropdown(
                                choices=eval_emb_choices,
                                value=eval_cur_emb or (eval_emb_choices[0] if eval_emb_choices else ""),
                                label="嵌入模型",
                                allow_custom_value=True,
                            )
                        eval_llm_num_ctx = gr.Number(
                            label="LLM 上下文窗口（num_ctx）",
                            value=float(initial_llm_num_ctx_for_ui()),
                            minimum=1,
                            maximum=262144,
                            precision=0,
                        )
                        with gr.Row():
                            eval_generation_mode = gr.Radio(
                                choices=[("Full LLM", "llm"), ("Retrieval-only", "retrieval_only")],
                                value="llm",
                                label="Generation mode",
                            )
                            eval_retrieval_mode = gr.Radio(
                                choices=[("Hybrid", "hybrid"), ("Vector", "vector"), ("Keyword", "keyword")],
                                value="hybrid",
                                label="Retrieval mode",
                            )
                        eval_query_anchoring = gr.Checkbox(
                            value=False,
                            label="Query anchoring（把 expected 文件名拼进检索问题；默认关闭，真实召回请保持关闭）",
                        )
                        with gr.Row():
                            eval_top_n = gr.Slider(
                                1,
                                100,
                                value=int(r.get("top_n_default", 20)),
                                step=1,
                                label="Top-N",
                            )
                            eval_top_k = gr.Slider(
                                1,
                                20,
                                value=int(r.get("top_k_default", 3)),
                                step=1,
                                label="Top-K",
                            )
                        with gr.Row():
                            eval_chunk_size = gr.Slider(
                                128,
                                2048,
                                value=int(c.get("chunk_size_default", 512)),
                                step=64,
                                label="Chunk",
                            )
                            eval_chunk_overlap = gr.Slider(
                                0,
                                512,
                                value=int(c.get("chunk_overlap_default", 64)),
                                step=32,
                                label="Overlap",
                            )
                        eval_chunk_mode = gr.Radio(
                            choices=[
                                ("按句切分", "sentence"),
                                ("按 Token 切分", "token"),
                                ("段落优先", "paragraph"),
                            ],
                            value=str(c.get("chunk_mode_default") or "sentence").strip().lower() or "sentence",
                            label="切分策略（仅对照；RUN 指纹以当前索引 manifest 为准，改切片需重建）",
                        )
                        eval_use_rerank = gr.Checkbox(
                            value=bool(cfg.rerank.get("enabled_default", False)),
                            label="启用 Cross-Encoder 重排",
                        )
                        btn_eval_run = gr.Button("执行评测", variant="primary")
                        with gr.Accordion("运行日志", open=False):
                            eval_run_status = gr.Textbox(label="运行日志", lines=8, max_lines=14)
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5, min_width=520):
                        with gr.Row(equal_height=True):
                            gr.Markdown("##### 实验面板")
                            btn_eval_dashboard_refresh = gr.Button("刷新实验面板", variant="secondary", size="sm")
                        with gr.Tabs():
                            with gr.Tab("总览"):
                                with gr.Row(equal_height=False):
                                    with gr.Column(scale=2, min_width=220):
                                        eval_run_summary_df = gr.Dataframe(
                                            headers=["指标", "值"],
                                            value=[["样本数", 0]],
                                            show_label=False,
                                            interactive=False,
                                            static_columns=[0, 1],
                                            col_count=(2, "fixed"),
                                            type="array",
                                            wrap=False,
                                            max_height=220,
                                        )
                                    with gr.Column(scale=3, min_width=280):
                                        eval_run_params_df = gr.Dataframe(
                                            headers=["参数", "值"],
                                            value=_empty_eval_param_rows(),
                                            show_label=False,
                                            interactive=False,
                                            static_columns=[0, 1],
                                            col_count=(2, "fixed"),
                                            type="array",
                                            wrap=False,
                                            max_height=220,
                                        )
                                eval_recent_runs_df = gr.Dataframe(
                                    headers=["RUN ID", "评测集", "运行名称", "创建时间", "样本数", "文件命中可评估题", "非文件命中题", "候选命中率", "文档命中率", "片段命中率", "答案命中率（参考）", "拒答正确率（参考）"],
                                    value=_eval_run_summary_rows(),
                                    show_label=False,
                                    interactive=False,
                                    static_columns=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                                    col_count=(12, "fixed"),
                                    type="array",
                                    wrap=False,
                                    max_height=240,
                                )
                            with gr.Tab("结果明细"):
                                eval_run_result_df = gr.Dataframe(
                                    headers=["#", "问题", "候选命中", "文档命中", "片段命中", "答案命中（参考）", "拒答判定（参考）", "错误类型", "归因", "首个来源", "QA ID"],
                                    value=_empty_eval_result_rows(),
                                    show_label=False,
                                    interactive=False,
                                    static_columns=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                    col_count=(11, "fixed"),
                                    type="array",
                                    wrap=False,
                                    max_height=520,
                                )
                            with gr.Tab("诊断分析"):
                                with gr.Row(equal_height=False):
                                    with gr.Column(scale=2, min_width=220):
                                        eval_error_df = gr.Dataframe(
                                            headers=["错误类型", "数量", "占比"],
                                            value=_empty_eval_error_rows(),
                                            show_label=False,
                                            interactive=False,
                                            static_columns=[0, 1, 2],
                                            col_count=(3, "fixed"),
                                            type="array",
                                            wrap=False,
                                            max_height=240,
                                        )
                                    with gr.Column(scale=3, min_width=320):
                                        eval_tag_df = gr.Dataframe(
                                            headers=["标签", "样本数", "文档命中率", "片段命中率", "答案命中率（参考）", "OK 数"],
                                            value=_empty_eval_tag_rows(),
                                            show_label=False,
                                            interactive=False,
                                            static_columns=[0, 1, 2, 3, 4, 5],
                                            col_count=(6, "fixed"),
                                            type="array",
                                            wrap=False,
                                            max_height=220,
                                        )
                                        eval_file_df = gr.Dataframe(
                                            headers=["File", "Cases", "Candidate hit", "Context hit", "Chunk hit", "Answer hit", "OK", "Errors"],
                                            value=_empty_eval_file_rows(),
                                            show_label=False,
                                            interactive=False,
                                            static_columns=[0, 1, 2, 3, 4, 5, 6, 7],
                                            col_count=(8, "fixed"),
                                            type="array",
                                            wrap=False,
                                            max_height=260,
                                        )
                                eval_chunk_diag_df = gr.Dataframe(
                                    headers=["Case ID", "错误类型", "片段命中", "最佳阶段", "Rank", "文件", "相似度", "覆盖率", "公共字符", "片段预览"],
                                    value=_empty_eval_chunk_diag_rows(),
                                    show_label=False,
                                    interactive=False,
                                    static_columns=list(range(10)),
                                    col_count=(10, "fixed"),
                                    type="array",
                                    wrap=False,
                                    max_height=260,
                                )
                                eval_funnel_df = gr.Dataframe(
                                    headers=["Case ID", "阶段", "候选数", "目标文件命中", "未解析文件", "Top files", "错误类型", "QA ID"],
                                    value=_empty_eval_funnel_rows(),
                                    show_label=False,
                                    interactive=False,
                                    static_columns=list(range(8)),
                                    col_count=(8, "fixed"),
                                    type="array",
                                    wrap=False,
                                    max_height=300,
                                )
                                btn_eval_governance = gr.Button("刷新治理/失败详情", variant="secondary", size="sm")
                                with gr.Row(equal_height=False):
                                    eval_quality_df = gr.Dataframe(
                                        headers=["质量项", "数量", "占比"],
                                        value=_eval_dataset_quality_rows(_parse_eval_dataset_id(eval_value)),
                                        show_label=False,
                                        interactive=False,
                                        static_columns=[0, 1, 2],
                                        col_count=(3, "fixed"),
                                        type="array",
                                        wrap=False,
                                        max_height=260,
                                    )
                                    eval_failure_df = gr.Dataframe(
                                        headers=["Case ID", "问题", "错误类型", "归因", "未解析文件", "建议文件", "首个来源", "QA ID"],
                                        value=_eval_failure_detail_rows(_latest_eval_run_id(_parse_eval_dataset_id(eval_value))),
                                        show_label=False,
                                        interactive=False,
                                        static_columns=[0, 1, 2, 3, 4, 5, 6, 7],
                                        col_count=(8, "fixed"),
                                        type="array",
                                        wrap=False,
                                        max_height=320,
                                    )
                                with gr.Accordion("Expected file alias 治理", open=True):
                                    eval_alias_status = gr.Textbox(label="Alias 操作状态", value="", interactive=False, lines=1)
                                    eval_alias_df = gr.Dataframe(
                                        headers=["Expected raw", "Cases", "Status", "Target file", "Suggestions", "Source"],
                                        value=_eval_file_alias_rows(_parse_eval_dataset_id(eval_value)),
                                        show_label=False,
                                        interactive=False,
                                        static_columns=[0, 1, 2, 3, 4, 5],
                                        col_count=(6, "fixed"),
                                        type="array",
                                        wrap=False,
                                        max_height=260,
                                    )
                                    with gr.Row():
                                        eval_alias_raw = gr.Textbox(
                                            label="Expected raw",
                                            placeholder="例如：《投资学》 / 《公司理财》",
                                            scale=2,
                                        )
                                        eval_alias_target = gr.Dropdown(
                                            label="Target file",
                                            choices=_uploaded_eval_file_names(),
                                            value=(_uploaded_eval_file_names() or [None])[0],
                                            interactive=True,
                                            scale=2,
                                        )
                                    with gr.Row():
                                        btn_eval_alias_save = gr.Button("保存 alias", variant="primary", size="sm")
                                        btn_eval_alias_delete = gr.Button("删除 alias", variant="secondary", size="sm")
                    with gr.Column(scale=2, min_width=320):
                        gr.Markdown("##### 实验工具")
                        with gr.Accordion("批量问题回放", open=False):
                            batch_title = gr.Textbox(
                                label="回放会话标题",
                                placeholder="可选：默认自动生成批量回放时间",
                            )
                            batch_questions = gr.Textbox(
                                label="问题列表（每行一个）",
                                lines=6,
                                max_lines=12,
                                placeholder="例如\n本项目的目标是什么？\n图片增强支持哪些方式？",
                            )
                            btn_batch_replay = gr.Button("执行批量回放", variant="primary")
                            batch_status = gr.Textbox(label="回放日志", lines=6, max_lines=12)
                            batch_result_df = gr.Dataframe(
                                headers=["问题", "检索结果", "候选数", "送入上下文", "首个来源文件", "QA ID"],
                                value=[["（暂无结果）", "—", 0, 0, "—", "—"]],
                                show_label=False,
                                interactive=False,
                                static_columns=[0, 1, 2, 3, 4, 5],
                                col_count=(6, "fixed"),
                                type="array",
                                wrap=False,
                                max_height=220,
                            )
                        with gr.Accordion("实验对比汇总", open=True):
                            btn_compare = gr.Button("刷新实验对比", variant="secondary", size="sm")
                            compare_df = gr.Dataframe(
                                headers=[
                                    "嵌入模型",
                                    "LLM",
                                    "切分策略",
                                    "Chunk",
                                    "Overlap",
                                    "重排",
                                    "Top-N",
                                    "Top-K",
                                    "问答数",
                                    "已评分数",
                                    "平均分",
                                    "拒答/未命中数",
                                    "会话数",
                                ],
                                value=[["（暂无问答记录）", "", "", 0, 0, "否", 0, 0, 0, 0, "—", 0, 0]],
                                show_label=False,
                                interactive=False,
                                static_columns=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                                col_count=(13, "fixed"),
                                type="array",
                                wrap=False,
                                max_height=260,
                            )
                            eval_compare_df = gr.Dataframe(
                                headers=[
                                    "RUN ID",
                                    "评测集",
                                    "运行名称",
                                    "Fingerprint",
                                    "检索模式",
                                    "向量",
                                    "LLM",
                                    "Embedding",
                                    "Top-N",
                                    "Top-K",
                                    "候选命中率",
                                    "上下文命中率",
                                    "片段命中率",
                                    "答案命中率",
                                    "OK",
                                    "Top attribution",
                                ],
                                value=_eval_run_compare_rows(),
                                show_label=False,
                                interactive=False,
                                static_columns=list(range(16)),
                                col_count=(16, "fixed"),
                                type="array",
                                wrap=False,
                                max_height=260,
                            )
                            eval_diff_df = gr.Dataframe(
                                headers=["对比项", "当前RUN", "上一RUN", "变化", "说明"],
                                value=_eval_run_diff_rows(),
                                show_label=False,
                                interactive=False,
                                static_columns=[0, 1, 2, 3, 4],
                                col_count=(5, "fixed"),
                                type="array",
                                wrap=False,
                                max_height=320,
                            )
                            baseline_status = gr.Textbox(label="Baseline 状态", value="", interactive=False, lines=1)
                            btn_set_baseline = gr.Button("将当前评测集最新 RUN 设为 Baseline", variant="secondary", size="sm")
                            cb_set_baseline_force = gr.Checkbox(
                                label="强制覆盖（绕过数据集阻塞项）",
                                value=False,
                            )
                            baseline_compare_df = gr.Dataframe(
                                headers=["对比项", "最新RUN", "Baseline", "变化", "说明"],
                                value=_eval_baseline_compare_rows(_parse_eval_dataset_id(eval_value)),
                                show_label=False,
                                interactive=False,
                                static_columns=[0, 1, 2, 3, 4],
                                col_count=(5, "fixed"),
                                type="array",
                                wrap=False,
                                max_height=260,
                            )
                            btn_param_grid = gr.Button("运行 retrieval-only 参数网格", variant="secondary", size="sm")
                            gr.Markdown(
                                "*排序口径：上下文命中率 → 片段命中率 → 候选命中率，降序。选参数组后顶部出现的就是当前数据集上表现最佳的取回口径。*",
                                visible=True,
                            )
                            grid_status = gr.Textbox(label="参数网格日志", lines=3, max_lines=8, interactive=False)
                            grid_df = gr.Dataframe(
                                headers=["RUN ID", "Mode", "Rerank", "Top-N", "Top-K", "候选命中率", "上下文命中率", "片段命中率", "全标注候选命中率"],
                                value=[["（暂无网格结果）", "", "", "", "", "", "", "", ""]],
                                show_label=False,
                                interactive=False,
                                static_columns=list(range(9)),
                                col_count=(9, "fixed"),
                                type="array",
                                wrap=False,
                                max_height=220,
                            )
                        with gr.Accordion("平台运维", open=False):
                            btn_ollama_health = gr.Button("检查 Ollama 健康", variant="secondary", size="sm")
                            ollama_health_df = gr.Dataframe(
                                headers=["项目", "值"],
                                value=ollama_health_rows({}),
                                show_label=False,
                                interactive=False,
                                static_columns=[0, 1],
                                col_count=(2, "fixed"),
                                type="array",
                                wrap=False,
                                max_height=180,
                            )
                            btn_export_eval_report = gr.Button("导出最新评测报告 JSON", variant="secondary", size="sm")
                            btn_export_eval_compare = gr.Button(
                                "导出逐题对比表 (原题+结果 xlsx)",
                                variant="secondary",
                                size="sm",
                            )
                            eval_report_file = gr.File(label="评测报告 / 对比表", interactive=False)
                            btn_regression_eval = gr.Button("运行当前评测集检索回归", variant="secondary", size="sm")

                _chat_inputs = [
                    msg,
                    chatbot,
                    system_prompt,
                    top_n,
                    top_k,
                    use_rerank,
                    chunk_size,
                    chunk_overlap,
                    chunk_mode,
                    llm_dd,
                    llm_num_ctx,
                    embed_dd,
                    session_id,
                ]
                send.click(do_chat_stream, _chat_inputs, [chatbot, sources, retrieval_diag, msg])
                msg.submit(do_chat_stream, _chat_inputs, [chatbot, sources, retrieval_diag, msg])

                def _clear_chat():
                    return [], _SOURCES_EMPTY_HTML, _DIAG_EMPTY_HTML, ""

                btn_clear.click(_clear_chat, outputs=[chatbot, sources, retrieval_diag, msg])

                btn_rate.click(do_rate_last, [rating, note, session_id], rate_status)
                btn_batch_replay.click(
                    do_batch_replay,
                    [
                        batch_questions,
                        batch_title,
                        eval_system_prompt,
                        eval_top_n,
                        eval_top_k,
                        eval_use_rerank,
                        eval_chunk_size,
                        eval_chunk_overlap,
                        eval_chunk_mode,
                        eval_llm_dd,
                        eval_llm_num_ctx,
                        eval_embed_dd,
                        eval_generation_mode,
                        eval_retrieval_mode,
                    ],
                    [batch_status, batch_result_df, session_table],
                )
                btn_compare.click(do_experiment_compare, outputs=[compare_df, eval_compare_df, eval_diff_df, baseline_compare_df])
                btn_set_baseline.click(
                    do_set_latest_eval_baseline,
                    [eval_dataset_dd, cb_set_baseline_force],
                    [baseline_status, baseline_compare_df],
                )
                btn_param_grid.click(
                    do_run_retrieval_param_grid,
                    [eval_dataset_dd],
                    [grid_status, grid_df, eval_recent_runs_df, eval_compare_df, eval_diff_df, baseline_compare_df],
                )
                btn_eval_governance.click(
                    do_eval_governance_refresh,
                    [eval_dataset_dd],
                    [eval_quality_df, eval_failure_df, eval_alias_df, eval_alias_target],
                )
                eval_alias_df.select(
                    do_eval_alias_row_select,
                    outputs=[eval_alias_raw, eval_alias_target],
                )
                btn_eval_alias_save.click(
                    do_save_eval_file_alias,
                    [eval_dataset_dd, eval_alias_raw, eval_alias_target],
                    [eval_alias_status, eval_quality_df, eval_failure_df, eval_alias_df, eval_alias_target],
                )
                btn_eval_alias_delete.click(
                    do_delete_eval_file_alias,
                    [eval_dataset_dd, eval_alias_raw],
                    [eval_alias_status, eval_quality_df, eval_failure_df, eval_alias_df, eval_alias_target],
                )
                btn_ollama_health.click(do_ollama_health_check, outputs=[ollama_health_df])
                btn_export_eval_report.click(do_export_latest_eval_report, [eval_dataset_dd], [eval_report_file])
                btn_export_eval_compare.click(
                    do_export_latest_eval_case_compare,
                    [eval_dataset_dd],
                    [eval_report_file],
                )
                btn_regression_eval.click(
                    do_run_retrieval_regression,
                    [eval_dataset_dd],
                    [
                        eval_run_status,
                        eval_run_result_df,
                        eval_run_summary_df,
                        eval_recent_runs_df,
                        eval_run_params_df,
                        eval_error_df,
                        eval_tag_df,
                        eval_file_df,
                        eval_chunk_diag_df,
                        eval_funnel_df,
                    ],
                )
                btn_eval_import.click(
                    do_eval_dataset_import,
                    [eval_file, eval_dataset_name, eval_dataset_desc],
                    [
                        eval_import_status,
                        eval_dataset_dd,
                        eval_cases_df,
                        eval_dataset_summary,
                        eval_recent_runs_df,
                        eval_run_result_df,
                        eval_run_summary_df,
                        eval_run_params_df,
                        eval_error_df,
                        eval_tag_df,
                        eval_file_df,
                        eval_chunk_diag_df,
                        eval_funnel_df,
                        eval_quality_df,
                        eval_failure_df,
                        eval_alias_df,
                        eval_alias_target,
                    ],
                )
                eval_dataset_dd.change(
                    do_eval_dataset_select,
                    [eval_dataset_dd],
                    [
                        eval_dataset_summary,
                        eval_cases_df,
                        eval_recent_runs_df,
                        eval_run_result_df,
                        eval_run_summary_df,
                        eval_run_params_df,
                        eval_error_df,
                        eval_tag_df,
                        eval_file_df,
                        eval_chunk_diag_df,
                        eval_funnel_df,
                        eval_quality_df,
                        eval_failure_df,
                        eval_alias_df,
                        eval_alias_target,
                    ],
                )
                btn_eval_run.click(
                    do_run_eval_dataset,
                    [
                        eval_dataset_dd,
                        eval_run_name,
                        eval_system_prompt,
                        eval_top_n,
                        eval_top_k,
                        eval_use_rerank,
                        eval_chunk_size,
                        eval_chunk_overlap,
                        eval_chunk_mode,
                        eval_llm_dd,
                        eval_llm_num_ctx,
                        eval_embed_dd,
                        eval_generation_mode,
                        eval_retrieval_mode,
                        eval_query_anchoring,
                    ],
                    [
                        eval_run_status,
                        eval_run_result_df,
                        eval_run_summary_df,
                        eval_recent_runs_df,
                        eval_run_params_df,
                        eval_error_df,
                        eval_tag_df,
                        eval_file_df,
                        eval_chunk_diag_df,
                        eval_funnel_df,
                    ],
                )
                btn_eval_dashboard_refresh.click(
                    do_eval_dashboard_refresh,
                    [eval_dataset_dd],
                    [
                        eval_run_status,
                        eval_recent_runs_df,
                        eval_run_result_df,
                        eval_run_summary_df,
                        eval_run_params_df,
                        eval_error_df,
                        eval_tag_df,
                        eval_file_df,
                        eval_chunk_diag_df,
                        eval_funnel_df,
                    ],
                )
                eval_recent_runs_df.select(
                    do_eval_run_select,
                    [eval_dataset_dd],
                    [
                        eval_run_status,
                        eval_run_result_df,
                        eval_run_summary_df,
                        eval_recent_runs_df,
                        eval_run_params_df,
                        eval_error_df,
                        eval_tag_df,
                        eval_file_df,
                        eval_chunk_diag_df,
                        eval_funnel_df,
                    ],
                )

        demo.load(
            _on_app_load,
            outputs=[
                llm_dd,
                embed_dd,
                session_table,
                chatbot,
                session_id,
                kb_files_df,
                kb_embed_warn,
                llm_num_ctx,
                kb_image_enrichment,
                kb_image_ocr_engine,
                kb_image_tesseract_lang,
                kb_image_vision_model,
                kb_image_ocr_skip,
                kb_chroma_redirect_warn,
                kb_chroma_redirect_ack_btn,
            ],
        )
        btn_refresh_models.click(
            _refresh_models_action,
            outputs=[llm_dd, embed_dd, kb_image_vision_model],
        )

    return demo


def _rag_tip_head_script() -> str:
    """把提示框定位脚本写入页面 <head>（gr.Blocks(head=...)，不是 launch）。"""
    p = ROOT / "rag_tip_popover.js"
    if not p.is_file():
        print(
            "[RAG-Lite] 警告: 未找到 "
            + str(p)
            + "，感叹号提示无法靠右定位。请把该文件与 main.py 放在同一目录。",
            flush=True,
        )
        return ""
    body = p.read_text(encoding="utf-8")
    return f'<script type="text/javascript">\n{body}\n</script>'


def main():
    ui = cfg.ui
    print("[RAG-Lite] 构建 Gradio 界面 …", flush=True)
    demo = build_ui()
    print(f"[RAG-Lite] 启用队列并绑定 {ui.get('server_name', '127.0.0.1')}:{ui.get('server_port', 7860)} …", flush=True)
    demo.queue()
    host = str(ui.get("server_name", "127.0.0.1"))
    port = int(ui.get("server_port", 7860))
    print(f"[RAG-Lite] 启动后请打开: http://{host}:{port}/ （关闭请 Ctrl+C）")
    demo.launch(
        server_name=host,
        server_port=port,
        share=bool(ui.get("share", False)),
        inbrowser=bool(ui.get("open_browser", True)),
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[RAG-Lite] 已停止。", flush=True)
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        print("\n[RAG-Lite] 运行失败，请查看上方报错。", flush=True)
        sys.exit(1)


