from __future__ import annotations

import html
import importlib.metadata
import importlib.util
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

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
from rag_lite.ingest import uploaded_files_snapshot
from rag_lite.ollama_util import list_ollama_models, merge_model_choices
from rag_lite import prefs as prefs_mod
from rag_lite.store import ExperimentStore

print("[RAG-Lite] 加载 config.yaml 与本地目录 …", flush=True)
cfg = load_config(ROOT)
store = ExperimentStore(cfg.sqlite_path)
print("[RAG-Lite] 配置与 SQLite 就绪。", flush=True)


def _ollama_base() -> str:
    return str(cfg.ollama.get("base_url", "http://127.0.0.1:11434"))


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


def _fmt_session_label(sid: int, title: str | None) -> str:
    t = (title or "未命名").strip() or "未命名"
    if len(t) > 42:
        t = t[:39] + "…"
    return f"#{sid} · {t}"


def _all_session_labels() -> list[str]:
    rows = store.list_sessions(limit=400)
    return [_fmt_session_label(int(r["id"]), r.get("title")) for r in rows]


def _session_id_from_label(label: str | None) -> int | None:
    if not label or not isinstance(label, str):
        return None
    m = re.match(r"^#(\d+)", label.strip())
    if not m:
        return None
    return int(m.group(1))


def session_history_for_chatbot(session_id: int) -> list:
    rows = store.fetch_qa_rows_for_session(session_id)
    return [[r["question"], r["answer"]] for r in rows]


def kb_docs_markdown(embed_selected: str) -> str:
    manifest = store.get_index_manifest()
    disk = uploaded_files_snapshot(cfg)
    lines = [
        "| 文件 | 大小 | 索引状态 |",
        "| --- | --- | --- |",
    ]
    if not disk:
        lines.append("| — | — | 上传目录暂无文档 |")
        return "\n".join(lines)
    emb_warn = ""
    if manifest and manifest.get("embed_model") and manifest["embed_model"] != embed_selected:
        emb_warn = (
            "\n\n> **提示**：磁盘索引由嵌入模型 `" + str(manifest["embed_model"]) + "` 构建，"
            "当前选择为 `" + str(embed_selected) + "`。"
            "检索将按当前选择的模型加载向量库；若报错或效果异常，请用与构建时一致的模型或重新构建索引。\n"
        )
    mf_files: dict[str, dict] = {}
    if manifest:
        for x in manifest.get("files") or []:
            if isinstance(x, dict) and x.get("name"):
                mf_files[str(x["name"])] = x
    for f in disk:
        name = f["name"]
        sz = int(f["size"])
        sz_s = f"{sz / 1024:.1f} KB" if sz < 1024 * 1024 else f"{sz / (1024 * 1024):.1f} MB"
        st = "未索引"
        if manifest:
            prev = mf_files.get(name)
            if prev and prev.get("mtime_ns") == f.get("mtime_ns") and int(prev.get("size", -1)) == sz:
                st = "✓ 已入库（与上次成功构建一致）"
            elif prev:
                st = "⚠ 文件已变更，需重建索引"
            else:
                st = "未索引（相对上次构建为新增）"
        lines.append(f"| `{name}` | {sz_s} | {st} |")
    return "\n".join(lines) + emb_warn


def list_upload_filenames() -> list[str]:
    return [f["name"] for f in uploaded_files_snapshot(cfg)]


def remove_kb_file(filename: str, embed_selected: str) -> tuple[str, str]:
    fn = (filename or "").strip()
    if not fn:
        return "请选择要移除的文件。", kb_docs_markdown(embed_selected)
    fn = Path(fn).name
    p = (cfg.uploads_dir / fn).resolve()
    uploads_resolved = cfg.uploads_dir.resolve()
    try:
        p.relative_to(uploads_resolved)
    except ValueError:
        return "路径非法。", kb_docs_markdown(embed_selected)
    if not p.is_file():
        return f"文件不存在：{fn}", kb_docs_markdown(embed_selected)
    try:
        p.unlink()
    except OSError as e:
        return f"删除失败：{e}", kb_docs_markdown(embed_selected)
    return (
        f"已从上传目录删除「{fn}」。若此前建过索引，请重新点击「构建向量索引」以同步向量库。",
        kb_docs_markdown(embed_selected),
    )


def label_for_session_id(sid: int) -> str | None:
    for r in store.list_sessions(limit=500):
        if int(r["id"]) == sid:
            return _fmt_session_label(sid, r.get("title"))
    return None

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
    '<p style="margin:0;color:#6b7280;font-size:0.92em;">'
    "回答生成后，此处展示检索到的文档片段与相似度 / 重排得分。"
    "</p>"
)


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


def _sources_panel_html(sources: list) -> str:
    """PRD：高亮展示引用片段与匹配分数；附带可折叠面板（FR-3.2）。"""
    intro = (
        '<p style="margin:0 0 8px 0;color:#444;font-size:0.95em;">'
        "以下为本次回答所<strong>引用</strong>的源文档片段；片段正文已用底色高亮，并标注检索得分。"
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
        kind_label = "Cross-Encoder 重排得分" if kind_key == "rerank" else "向量检索得分"
        score = s.get("score")
        score_s = f"{score:.4f}" if score is not None else "—"
        fname = html.escape(str(s.get("file_name", "")))
        chunk_raw = str(s.get("chunk", ""))
        chunk_esc = html.escape(chunk_raw).replace("\n", "<br />\n")
        badge_bg = _score_badge_class(kind_key)
        cards.append(
            f'<div style="border:1px solid #ddd;border-radius:8px;padding:12px 14px;'
            f'margin-bottom:10px;background:#fff;box-shadow:0 1px 2px rgba(0,0,0,0.04);">'
            f'<div style="display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin-bottom:10px;">'
            f'<span style="font-weight:600;color:#1565c0;">[{i}] {fname}</span>'
            f'<span style="background:{badge_bg};padding:3px 10px;border-radius:999px;'
            f'font-size:0.85em;border:1px solid rgba(0,0,0,0.06);">'
            f"{html.escape(kind_label)}：<strong>{html.escape(score_s)}</strong></span>"
            f"</div>"
            f'<div style="line-height:1.65;color:#222;">'
            f'<mark style="background:#fff59d;padding:2px 6px;border-radius:3px;'
            f'display:block;white-space:normal;">{chunk_esc}</mark>'
            f"</div></div>"
        )

    body = "".join(cards)
    return (
        intro
        + '<details open style="margin-top:4px;border:1px solid #e0e0e0;border-radius:8px;padding:8px 12px;background:#fafafa;">'
        + '<summary style="cursor:pointer;font-weight:600;color:#333;user-select:none;">'
        "参考来源（可折叠：文档名 · 高亮片段 · 向量/重排得分）"
        "</summary>"
        + '<div style="margin-top:12px;">'
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
    return msg, kb_docs_markdown(embed_selected), gr.update(choices=list_upload_filenames())


def do_build_index(chunk_size: int, overlap: int, chunk_mode: str, embed_model: str):
    """生成器：逐条刷新「状态」文本框，展示向量构建阶段与分批进度。"""
    ing = _lazy_ingest()
    eng = _lazy_engine()
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

    new_hist = history + [
        {"role": "user", "content": qtext},
        {"role": "assistant", "content": "正在检索与生成，请稍候…"},
    ]
    yield _chatbot_value(new_hist), _sources_panel_html([]), _out_msg()

    eng = _lazy_engine()
    em = (embed_model or "").strip() or None
    index = eng.get_index(cfg, embed_model=em)
    if index is None:
        err = "索引不存在：请先在「知识库」页上传文档并点击「构建向量索引」（或检查嵌入模型是否与构建时一致）。"
        new_hist[-1]["content"] = err
        yield _chatbot_value(new_hist), _sources_panel_html([]), _out_msg()
        return

    top_n = int(top_n)
    top_k = int(top_k)
    top_k = max(1, min(top_k, top_n))
    cm = (chunk_mode or "sentence").strip().lower()
    lm = (llm_model or "").strip() or None
    try:
        prior_rows = store.fetch_qa_rows_for_session(sid)
        if len(prior_rows) == 0:
            store.update_session_meta(sid, title=qtext[:60])

        nodes, kind = eng.retrieve(cfg, index, qtext, top_n, top_k, bool(use_rerank))
        sources = eng.nodes_to_source_dicts(nodes, kind)
        src_html = _sources_panel_html(sources)
        new_hist[-1]["content"] = ""
        yield _chatbot_value(new_hist), src_html, _out_msg()
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
        partial = ""
        for token in eng.stream_answer(cfg, qtext, system_prompt, nodes, llm_model=lm):
            partial += token
            new_hist[-1]["content"] = partial
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
        )
        LAST_QA_ID = store.insert_qa(
            question=qtext,
            answer=partial,
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

    _rag_css = """
    .rag-workbench-header h1 {
        font-size: 1.35rem;
        font-weight: 650;
        margin: 0 0 0.35em 0;
        letter-spacing: -0.02em;
    }
    .rag-workbench-header p { margin: 0.4em 0; line-height: 1.55; }
    .rag-workbench-tip { opacity: 0.88; font-size: 0.92em; margin-top: 0.5em !important; }
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

    def _persist_embed_choice(name: str):
        p = prefs_mod.load_prefs(cfg.data_dir)
        p["embed_model"] = name
        prefs_mod.save_prefs(cfg.data_dir, p)
        return kb_docs_markdown(name)

    def _refresh_models_action():
        llm_c, emb_c, cll, cem = refresh_model_choices()
        return gr.update(choices=llm_c, value=cll), gr.update(choices=emb_c, value=cem)

    def _after_build_refresh(embed_sel: str):
        return kb_docs_markdown(embed_sel), gr.update(choices=list_upload_filenames())

    def _on_app_load():
        llm_c, emb_c, cur_llm, cur_emb = refresh_model_choices()
        store.ensure_default_session()
        choices = _all_session_labels()
        if not choices:
            store.create_session("默认会话")
            choices = _all_session_labels()
        sid = store.ensure_default_session()
        lbl = label_for_session_id(sid)
        if not lbl and choices:
            lbl = choices[0]
        hist = session_history_for_chatbot(sid)
        kb = kb_docs_markdown(cur_emb)
        files = list_upload_filenames()
        return (
            gr.update(choices=llm_c, value=cur_llm),
            gr.update(choices=emb_c, value=cur_emb),
            gr.update(choices=choices, value=lbl),
            hist,
            sid,
            kb,
            gr.update(choices=files),
        )

    def _on_session_pick(label: str):
        sid = _session_id_from_label(label)
        if sid is None:
            sid = store.ensure_default_session()
        return session_history_for_chatbot(sid), sid

    def _on_new_session():
        sid = store.create_session(None)
        choices = _all_session_labels()
        lbl = _fmt_session_label(sid, None)
        return [], sid, gr.update(choices=choices, value=lbl)

    with gr.Blocks(title="RAG 验证工作台", theme=_theme, css=_rag_css) as demo:
        gr.Markdown(
            '<div class="rag-workbench-header">'
            "<h1>RAG 验证工作台</h1>"
            "<p>本地数据与模型，用于检索与生成效果对比。请先运行 "
            '<a href="https://ollama.com" target="_blank" rel="noopener noreferrer">Ollama</a>'
            " 并拉取所需模型。向量模型在<strong>知识库</strong>选择；对话模型在<strong>对话</strong>选择；"
            "二者可与 <code>config.yaml</code> 默认值不同，并会记住上次选择。</p>"
            '<p class="rag-workbench-tip">流程：上传文档 → 选择嵌入模型并构建索引 → 在对话中选择 LLM 提问；'
            "左侧可切换历史会话。导出仍为可选备份。</p>"
            "</div>"
        )

        session_id = gr.State(value=store.ensure_default_session())

        # 须先于「对话」Tab 创建 chunk_* / chunk_mode / embed_dd，供事件绑定引用
        with gr.Tabs(selected="chat"):
            with gr.Tab("知识库", id="kb"):
                gr.Markdown(
                    "### 文档与向量索引\n"
                    "上传文件到本地目录后构建向量库。下方表格显示每个文件是否已与**最近一次成功构建**一致。"
                )
                with gr.Row():
                    llm_c0, emb_c0, _, cur_emb0 = refresh_model_choices()
                    embed_dd = gr.Dropdown(
                        choices=emb_c0,
                        value=cur_emb0 or (emb_c0[0] if emb_c0 else ""),
                        label="嵌入模型（Ollama）",
                        interactive=len(emb_c0) > 1,
                        allow_custom_value=True,
                    )
                    btn_refresh_models = gr.Button("刷新 Ollama 模型列表", variant="secondary", scale=0)
                gr.Markdown(
                    '<p style="font-size:0.88em;opacity:0.9;margin:0 0 0.6em 0;">'
                    "若仅有一个候选，将自动固定为该模型。切换嵌入模型后请重新构建索引。"
                    "</p>"
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
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1):
                        up = gr.File(
                            label="选择文件（可多选）",
                            file_count="multiple",
                            type="filepath",
                        )
                        max_mb = gr.Number(
                            value=float(ingest.get("max_file_size_mb", 50)),
                            label="单文件大小上限 (MB)",
                        )
                        kb_notice = gr.Textbox(label="上传提示", lines=1, max_lines=2)
                        with gr.Row():
                            btn_save = gr.Button("保存到上传目录", variant="secondary")
                            btn_build = gr.Button("构建向量索引", variant="primary")
                    with gr.Column(scale=1):
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
                            '<p style="margin:0.75em 0 0 0;font-size:0.9em;opacity:0.9;">'
                            "变更切分策略或大小后需<strong>重新构建</strong>。参数会写入问答记录。"
                            "</p>"
                        )
                kb_doc_table = gr.Markdown(value=kb_docs_markdown(cur_emb0 or ""))
                with gr.Row():
                    file_to_remove = gr.Dropdown(
                        choices=list_upload_filenames(),
                        label="从上传目录移除文件",
                        allow_custom_value=False,
                    )
                    btn_remove = gr.Button("移除所选", variant="secondary")
                build_msg = gr.Textbox(
                    label="构建日志（进行中会刷新，请勿重复点击「构建」）",
                    lines=12,
                    max_lines=28,
                    autoscroll=True,
                    show_copy_button=True,
                )

                btn_save.click(
                    do_save_uploads,
                    [up, max_mb, embed_dd],
                    [kb_notice, kb_doc_table, file_to_remove],
                )
                b_chain = btn_build.click(
                    do_build_index,
                    [chunk_size, chunk_overlap, chunk_mode, embed_dd],
                    build_msg,
                )
                b_chain.then(_after_build_refresh, embed_dd, [kb_doc_table, file_to_remove])
                btn_remove.click(remove_kb_file, [file_to_remove, embed_dd], [kb_notice, kb_doc_table]).then(
                    lambda: gr.update(choices=list_upload_filenames()),
                    outputs=file_to_remove,
                )
                embed_dd.change(_persist_embed_choice, embed_dd, kb_doc_table)

            with gr.Tab("对话", id="chat"):
                gr.Markdown("### 问答与引用溯源")
                llm_c1, _, cur_llm1, _ = refresh_model_choices()
                llm_dd = gr.Dropdown(
                    choices=llm_c1,
                    value=cur_llm1 or (llm_c1[0] if llm_c1 else ""),
                    label="对话模型 LLM（Ollama）",
                    interactive=len(llm_c1) > 1,
                    allow_custom_value=True,
                )
                llm_dd.change(_persist_llm_choice, llm_dd)
                with gr.Row():
                    with gr.Column(scale=1, min_width=260):
                        gr.Markdown("##### 会话")
                        session_dd = gr.Dropdown(
                            choices=_all_session_labels(),
                            value=label_for_session_id(store.ensure_default_session()),
                            label="历史会话",
                            allow_custom_value=False,
                        )
                        btn_new_session = gr.Button("新建会话", variant="secondary")
                    with gr.Column(scale=4, min_width=400):
                        chatbot = gr.Chatbot(
                            label="当前会话",
                            height=400,
                            type="tuples",
                            show_copy_button=True,
                            placeholder=(
                                "<div style=\"text-align:center;padding:1.2em;color:#6b7280;\">"
                                "选择或新建会话后提问。请先完成知识库索引构建。"
                                "</div>"
                            ),
                        )
                        sources = gr.HTML(
                            label="引用片段与得分",
                            value=_SOURCES_EMPTY_HTML,
                        )
                        with gr.Row():
                            msg = gr.Textbox(
                                placeholder="输入问题，Enter 发送",
                                show_label=False,
                                container=False,
                                scale=5,
                                lines=1,
                            )
                            send = gr.Button("发送", variant="primary", scale=0, min_width=92)
                            btn_clear = gr.Button("清空本页", variant="secondary", scale=0, min_width=88)
                    with gr.Column(scale=2, min_width=280):
                        gr.Markdown("##### 生成与检索")
                        with gr.Accordion("系统提示词", open=True):
                            system_prompt = gr.Textbox(
                                value=str(cfg.prompt.get("system_default", "")).strip(),
                                lines=8,
                                label="System prompt",
                                show_label=False,
                            )
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

                session_dd.change(_on_session_pick, session_dd, [chatbot, session_id])
                btn_new_session.click(_on_new_session, outputs=[chatbot, session_id, session_dd])

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
                    embed_dd,
                    session_id,
                ]
                send.click(do_chat_stream, _chat_inputs, [chatbot, sources, msg])
                msg.submit(do_chat_stream, _chat_inputs, [chatbot, sources, msg])

                def _clear_chat():
                    return [], _SOURCES_EMPTY_HTML, ""

                btn_clear.click(_clear_chat, outputs=[chatbot, sources, msg])

                btn_rate.click(do_rate_last, [rating, note], rate_status)

            with gr.Tab("实验导出", id="export"):
                gr.Markdown(
                    "### 导出与备份\n"
                    "问答记录（含 `session_id`）保存在本地 SQLite。此处可导出全库备份；日常浏览与回顾请优先使用「对话」页的会话列表。"
                )
                with gr.Row():
                    fmt = gr.Radio(choices=["json", "csv"], value="json", label="格式")
                    btn_exp = gr.Button("导出 qa_log", variant="primary")
                exp_path = gr.Textbox(label="导出文件路径", lines=2)
                btn_exp.click(do_export, [fmt], exp_path)

        demo.load(
            _on_app_load,
            outputs=[llm_dd, embed_dd, session_dd, chatbot, session_id, kb_doc_table, file_to_remove],
        )
        btn_refresh_models.click(_refresh_models_action, outputs=[llm_dd, embed_dd])

    return demo


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
