from __future__ import annotations

import html
import importlib.metadata
import importlib.util
import operator
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

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
# Gradio 6 起 Chatbot 默认仅接受 OpenAI messages；5.x 与 4.x 多为 [user, bot] 对话轮次
CHAT_USE_OPENAI_MESSAGES = _GRADIO_MAJOR >= 6
if _GRADIO_MAJOR >= 6:
    print(
        "[RAG-Lite] 提示: 检测到 Gradio 6+，导入会较慢。建议: pip install "
        f'{_GRADIO_CLIENT_PIN} {_GRADIO_VERSION_PIN} --force-reinstall',
        flush=True,
    )

print(
    f"[RAG-Lite] Gradio {gr.__version__} 已就绪，本步耗时 {time.perf_counter() - _t_gradio:.1f} 秒；"
    f" Chatbot 数据模式={'messages' if CHAT_USE_OPENAI_MESSAGES else 'tuples(5.x)'}",
    flush=True,
)

from rag_lite.config import load_config
from rag_lite.ingest import (
    _filename_equiv,
    chroma_collection_count,
    chroma_sample_distinct_filenames,
    uploaded_files_snapshot,
)
from rag_lite.ollama_util import list_ollama_models, merge_model_choices
from rag_lite import prefs as prefs_mod
from rag_lite.store import ExperimentStore

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
    return [[r["question"], r["answer"]] for r in rows]


def kb_embed_warning_markdown(embed_selected: str) -> str:
    manifest = store.get_index_manifest()
    em = manifest.get("embed_model") if manifest else None
    if not em or em == embed_selected:
        return ""
    return (
        "**提示**：磁盘索引由嵌入模型 `" + str(em) + "` 构建，当前选择为 `" + str(embed_selected) + "`。"
        "若检索报错或效果异常，请改回一致模型或重新构建索引。"
    )


def kb_files_table_value(embed_selected: str) -> list[list[str]]:
    """已上传文档表：文件名、大小、上传时间、向量时间、索引状态。"""
    manifest = store.get_index_manifest()
    disk = uploaded_files_snapshot(cfg)
    if not disk:
        return [["（暂无文件）", "—", "—", "—", "—"]]
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
            if prev and prev.get("mtime_ns") == f.get("mtime_ns") and int(prev.get("size", -1)) == sz:
                st = "✓ 已入库（与上次成功构建一致）"
                vec_t = _format_session_time(prev.get("indexed_at") or built_at_global)
            elif prev:
                st = "⚠ 文件已变更，需重建索引"
                vec_t = _format_session_time(prev.get("indexed_at") or built_at_global)
            else:
                st = "未索引（相对上次构建为新增）"
        out.append([name, sz_s, up_t, vec_t, st])
    return out


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


def _kb_chunk_preview_html(row_idx: Any) -> str:
    """知识库：按当前表格行对应的上传目录真实文件名展示切片（行号解析，避免 State 字符串与向量库不一致）。"""
    ing = _lazy_ingest()
    disk = uploaded_files_snapshot(cfg)
    fn = None
    ri = _coerce_kb_row_index(row_idx)
    if ri is not None and 0 <= ri < len(disk):
        fn = disk[ri]["name"]
    if not fn:
        return '<p style="color:#92400e;margin:0;">请先在表格中<strong>点击一行</strong>选中文件。</p>'
    chunks, total_found = ing.fetch_chunks_for_file(cfg, fn)
    if total_found == 0:
        cc = chroma_collection_count(cfg)
        chroma_path = html.escape(str(cfg.chroma_dir.resolve()))
        if cc == 0:
            return (
                f'<p style="color:#92400e;margin:0;">当前向量库为空（共 0 条，路径：<code>{chroma_path}</code>）。'
                "请先<strong>构建向量索引</strong>。</p>"
            )
        sample = chroma_sample_distinct_filenames(cfg, limit=24)
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
            f"<strong>{cc}</strong> 条记录）。请先在表格中<strong>重新点击该行</strong>再预览；"
            f"若仍失败，请<strong>重新构建向量索引</strong>，并核对下方文件名是否与「{html.escape(fn)}」一致。"
            f"{sample_html}</p>"
        )
    note = ""
    if total_found > len(chunks):
        note = f"（数据库中共 {total_found} 块，以下仅展示前 {len(chunks)} 块）"
    head = (
        f'<p style="margin:0 0 10px 0;color:#475569;font-size:0.9em;">'
        f"匹配到 <strong>{total_found}</strong> 个文本块{note}。</p>"
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


def _messages_to_tuple_turns(messages: list[dict]) -> list[list]:
    """Gradio 4.x Chatbot: [[user, bot], ...]"""
    pairs: list[list] = []
    i = 0
    while i < len(messages):
        m = messages[i]
        role = m.get("role")
        text = str(m.get("content", ""))
        if role == "user":
            if i + 1 < len(messages) and messages[i + 1].get("role") == "assistant":
                pairs.append([text, str(messages[i + 1].get("content", ""))])
                i += 2
            else:
                pairs.append([text, ""])
                i += 1
        elif role == "assistant":
            pairs.append(["", text])
            i += 1
        else:
            i += 1
    return pairs


def _chatbot_value(messages: list[dict]) -> list:
    if CHAT_USE_OPENAI_MESSAGES:
        return messages
    return _messages_to_tuple_turns(messages)


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
        eng.refresh_index_cache(cfg, embed_model=embed_model)


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
        yield _chatbot_value(history), _sources_panel_html([]), gr.update()
        return

    eng = _lazy_engine()
    em = (embed_model or "").strip() or None
    index = eng.get_index(cfg, embed_model=em)
    top_n = max(1, int(top_n))
    top_k = max(1, min(int(top_k), top_n))
    cm = (chunk_mode or "sentence").strip().lower()
    lm = (llm_model or "").strip() or None
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
        yield _chatbot_value(new_hist_err), _sources_panel_html([]), _out_msg()
        return

    new_hist = history + [
        {"role": "user", "content": qtext},
        {
            "role": "assistant",
            "content": f"**① 向量检索中…**（初筛 Top-{top_n}）",
        },
    ]
    yield _chatbot_value(new_hist), _sources_panel_html([]), _out_msg()

    try:
        prior_rows = store.fetch_qa_rows_for_session(sid)
        if len(prior_rows) == 0:
            store.update_session_meta(sid, title=qtext[:60])

        nodes_vec = eng.vector_retrieve(cfg, index, qtext, top_n)
        line1 = f"**① 向量检索完成**（候选 {len(nodes_vec)} 条 · 初筛 Top-{top_n}）"
        new_hist[-1]["content"] = line1
        yield _chatbot_value(new_hist), _sources_panel_html([]), _out_msg()

        if bool(use_rerank) and nodes_vec:
            new_hist[-1]["content"] = f"{line1}\n\n**② Cross-Encoder 重排中…**"
            yield _chatbot_value(new_hist), _sources_panel_html([]), _out_msg()

        nodes, kind = eng.apply_topk_rerank(cfg, qtext, nodes_vec, top_k, bool(use_rerank))
        sources = eng.nodes_to_source_dicts(nodes, kind)
        src_html = _sources_panel_html(sources)
        if not nodes:
            no_ctx = (
                "**根据已知材料无法回答。**\n\n"
                "本轮**未检索到任何文档片段**，因此**没有**把「已知上下文」发给大模型，回答不应来自你的上传文件。\n\n"
                "**请排查：**①「知识库」是否已成功构建索引；② 嵌入模型是否与构建时一致且已在 Ollama 就绪；"
                "③ 问题与文档主题是否相关。"
            )
            new_hist[-1]["content"] = no_ctx
            yield _chatbot_value(new_hist), src_html, _out_msg()
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
            )
            LAST_QA_ID = store.insert_qa(
                question=qtext,
                answer=no_ctx,
                sources=sources,
                params=params,
                session_id=sid,
            )
            yield _chatbot_value(new_hist), src_html, _out_msg()
            return

        kind_zh = "重排" if kind == "rerank" else "向量截断"
        if bool(use_rerank) and nodes_vec:
            progress_head = (
                f"{line1}\n\n"
                f"**② 重排完成**（{kind_zh}，送入上下文 {len(nodes)} 条）\n\n"
                f"**③ 正在生成回答…**"
            )
        else:
            progress_head = (
                f"{line1}\n\n"
                f"**③ 正在生成回答…**（未启用重排，送入上下文 {len(nodes)} 条）"
            )
        sep = "\n\n---\n\n"
        new_hist[-1]["content"] = progress_head + sep
        yield _chatbot_value(new_hist), src_html, _out_msg()
        partial = ""
        for token in eng.stream_answer(
            cfg, qtext, system_prompt, nodes, llm_model=lm, llm_num_ctx=nctx
        ):
            partial += token
            new_hist[-1]["content"] = progress_head + sep + partial
            yield _chatbot_value(new_hist), src_html, _out_msg()
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
        )
        # 入库保存完整气泡内容（含进度行 + 分隔线 + 正文），切换会话回来时仍能看到各步
        full_answer = (progress_head + sep + partial).strip()
        LAST_QA_ID = store.insert_qa(
            question=qtext,
            answer=full_answer,
            sources=sources,
            params=params,
            session_id=sid,
        )
        yield _chatbot_value(new_hist), src_html, _out_msg()
    except Exception:
        tb = traceback.format_exc()
        err = f"生成失败:\n```\n{tb}\n```"
        LAST_QA_ID = None
        new_hist[-1]["content"] = err
        yield _chatbot_value(new_hist), _sources_panel_html([]), _out_msg()


def do_rate_last(rating: float | None, note: str):
    global LAST_QA_ID
    if LAST_QA_ID is None:
        return "没有可打分的对话（先完成一次问答）。"
    r = int(rating) if rating is not None else None
    store.update_qa_rating(LAST_QA_ID, r, note or None)
    return f"已保存评分：id={LAST_QA_ID}"


def do_export(fmt: str):
    out_dir = cfg.data_dir / "exports"
    out_dir.mkdir(parents=True, exist_ok=True)
    if fmt == "json":
        dest = out_dir / "qa_export.json"
        store.export_json(dest)
    else:
        dest = out_dir / "qa_export.csv"
        store.export_csv(dest)
    return str(dest.resolve())


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
                        chunk_mode = gr.Radio(
                            choices=[
                                ("按句切分（LlamaIndex Sentence，默认）", "sentence"),
                                ("按 Token 切分", "token"),
                                ("段落优先（双换行再切）", "paragraph"),
                            ],
                            value="sentence",
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

                kb_files_df.select(_on_kb_file_select, outputs=kb_file_sel)
                btn_preview_chunks.click(
                    _kb_chunk_preview_html,
                    inputs=[kb_file_sel],
                    outputs=[kb_chunk_preview],
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
                            type="tuples",
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
                            minimum=2048,
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
                send.click(do_chat_stream, _chat_inputs, [chatbot, sources, msg])
                msg.submit(do_chat_stream, _chat_inputs, [chatbot, sources, msg])

                def _clear_chat():
                    return [], _SOURCES_EMPTY_HTML, ""

                btn_clear.click(_clear_chat, outputs=[chatbot, sources, msg])

                btn_rate.click(do_rate_last, [rating, note], rate_status)

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
