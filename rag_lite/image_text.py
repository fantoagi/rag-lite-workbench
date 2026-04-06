"""PDF/DOCX embedded images: OCR first, then optional Ollama vision caption."""

from __future__ import annotations

import base64
import io
import json
import urllib.error
import urllib.request
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_lite.config import AppConfig


def resolve_tesseract_cmd(explicit: str | None) -> str | None:
    """Prefer yaml path; else PATH; else common Windows install dirs."""
    import shutil

    if explicit:
        p = Path(explicit.strip())
        if p.is_file():
            return str(p.resolve())
        return explicit.strip()
    w = shutil.which("tesseract")
    if w:
        return w
    if sys.platform == "win32":
        for cand in (
            Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
            Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
        ):
            if cand.is_file():
                return str(cand.resolve())
    return None

# Used only when config omits vision_prompt (ASCII-only source file).
_DEFAULT_VISION_PROMPT = (
    "Briefly describe the image in Chinese in 1-3 sentences for retrieval; "
    "include key visible text if any."
)


@dataclass
class ImageEnrichOptions:
    base_url: str
    vision_model: str
    vision_prompt: str
    request_timeout: float
    ocr_skip_vlm_min_chars: int
    max_images_per_file: int
    min_image_side_px: int
    max_image_side_px: int
    ocr_engine: str
    tesseract_lang: str
    tesseract_cmd: str | None


def merged_ingest_for_image(
    cfg: AppConfig,
    pipeline_overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    """Merge cfg.ingest with optional UI overrides for image pipeline keys."""
    out = dict(cfg.ingest)
    if not pipeline_overrides:
        return out
    for key in (
        "vision_model",
        "tesseract_lang",
        "ocr_skip_vlm_min_chars",
        "tesseract_cmd",
        "ocr_engine",
    ):
        if key not in pipeline_overrides:
            continue
        v = pipeline_overrides[key]
        if v is None:
            continue
        if isinstance(v, str) and not str(v).strip():
            continue
        if key == "ocr_skip_vlm_min_chars":
            try:
                out[key] = max(0, min(500, int(v)))
            except (TypeError, ValueError):
                pass
            continue
        if key == "ocr_engine":
            s = str(v).strip().lower()
            if s in ("tesseract", "paddleocr"):
                out[key] = s
            continue
        out[key] = v
    return out


def image_enrich_options_from_config(
    cfg: AppConfig,
    *,
    image_enrichment: bool | None = None,
    pipeline_overrides: dict[str, Any] | None = None,
) -> ImageEnrichOptions | None:
    ing_merged = merged_ingest_for_image(cfg, pipeline_overrides)
    if image_enrichment is not None:
        on = bool(image_enrichment)
    else:
        on = bool(ing_merged.get("image_enrichment"))
    if not on:
        return None
    o = cfg.ollama
    vm = str(ing_merged.get("vision_model") or "").strip()
    if not vm:
        vm = str(o.get("llm_model") or "llava")
    vp = ing_merged.get("vision_prompt")
    if vp is None or (isinstance(vp, str) and not str(vp).strip()):
        vp = cfg.ingest.get("vision_prompt")
    if vp is None or (isinstance(vp, str) and not str(vp).strip()):
        vp = _DEFAULT_VISION_PROMPT
    else:
        vp = str(vp)
    try:
        ocr_skip = int(ing_merged.get("ocr_skip_vlm_min_chars", 20))
    except (TypeError, ValueError):
        ocr_skip = 20
    ocr_skip = max(0, min(500, ocr_skip))
    raw_engine = str(ing_merged.get("ocr_engine", "tesseract")).strip().lower()
    ocr_engine = raw_engine if raw_engine in ("tesseract", "paddleocr") else "tesseract"
    return ImageEnrichOptions(
        base_url=str(o.get("base_url", "http://127.0.0.1:11434")).rstrip("/"),
        vision_model=vm,
        vision_prompt=vp,
        request_timeout=float(o.get("request_timeout", 600.0)),
        ocr_skip_vlm_min_chars=ocr_skip,
        max_images_per_file=max(1, int(ing_merged.get("max_images_per_file", 50))),
        min_image_side_px=max(1, int(ing_merged.get("min_image_side_px", 24))),
        max_image_side_px=max(64, int(ing_merged.get("max_image_side_px", 1600))),
        ocr_engine=ocr_engine,
        tesseract_lang=str(ing_merged.get("tesseract_lang", "chi_sim+eng")),
        tesseract_cmd=resolve_tesseract_cmd(ing_merged.get("tesseract_cmd") or None),
    )


def _resize_for_ocr(img: Any, max_side: int) -> Any:
    from PIL import Image

    img = img.convert("RGB")
    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img
    scale = max_side / float(m)
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    return img.resize((nw, nh), Image.Resampling.LANCZOS).convert("RGB")


_paddle_ocr_instance: Any = None
_paddle_ocr_failed: bool = False
_paddle_ocr_warned: bool = False

_vision_ollama_warned: bool = False


def _jpeg_bytes_for_vision(image_bytes: bytes, max_side: int) -> bytes:
    """Shrink + JPEG re-encode so Ollama requests stay small and stable."""
    from PIL import Image

    img = Image.open(io.BytesIO(image_bytes))
    img = _resize_for_ocr(img, max_side)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88, optimize=True)
    return buf.getvalue()


def _ollama_vision_warn_once(msg: str) -> None:
    global _vision_ollama_warned
    if _vision_ollama_warned:
        return
    _vision_ollama_warned = True
    print("[RAG-Lite] " + msg, flush=True)


def _ocr_tesseract_bytes(image_bytes: bytes, opts: ImageEnrichOptions) -> str:
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return ""

    if opts.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = opts.tesseract_cmd

    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = _resize_for_ocr(img, opts.max_image_side_px)
        w, h = img.size
        if min(w, h) < opts.min_image_side_px:
            return ""
        text = pytesseract.image_to_string(img, lang=opts.tesseract_lang)
        return (text or "").strip()
    except Exception:
        return ""


def _parse_paddle_v2_lines(page: Any) -> list[str]:
    """Classic PP-OCR: list of [box, (text, score)]."""
    if not isinstance(page, list):
        return []
    lines: list[str] = []
    for item in page:
        if not item or len(item) < 2:
            continue
        seg = item[1]
        if isinstance(seg, (list, tuple)) and len(seg) >= 1:
            lines.append(str(seg[0]))
        else:
            lines.append(str(seg))
    return lines


def _lines_from_paddle_result(result: Any) -> list[str]:
    """PP-OCR 2.x list layout; 3.x may return dict / OCRResult / rec_texts."""
    if result is None:
        return []
    # 2.x: usually [ [ line, ... ] ]?????????????????? [ line, ... ]
    if isinstance(result, list) and result:
        for candidate in (result[0], result):
            if not isinstance(candidate, list):
                continue
            v2 = _parse_paddle_v2_lines(candidate)
            if v2:
                return [x.strip() for x in v2 if x.strip()]
    # 3.x / dict / object
    if hasattr(result, "rec_texts"):
        rt = getattr(result, "rec_texts", None)
        if isinstance(rt, (list, tuple)):
            out = [str(x).strip() for x in rt if str(x).strip()]
            if out:
                return out
    if isinstance(result, dict):
        for key in ("rec_texts", "texts", "res_texts"):
            rt = result.get(key)
            if isinstance(rt, (list, tuple)):
                out = [str(x).strip() for x in rt if str(x).strip()]
                if out:
                    return out
        if "result" in result:
            return _lines_from_paddle_result(result["result"])
    if isinstance(result, (list, tuple)) and len(result) == 1:
        return _lines_from_paddle_result(result[0])
    return []


def _ocr_paddleocr_bytes(image_bytes: bytes, opts: ImageEnrichOptions) -> str:
    global _paddle_ocr_instance, _paddle_ocr_failed, _paddle_ocr_warned
    if _paddle_ocr_failed:
        return ""
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        return ""

    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = _resize_for_ocr(img, opts.max_image_side_px)
        w, h = img.size
        if min(w, h) < opts.min_image_side_px:
            return ""
        arr_rgb = np.asarray(img.convert("RGB"))
    except Exception:
        return ""

    if _paddle_ocr_instance is None:
        try:
            from paddleocr import PaddleOCR

            # PaddleOCR 3.x removed show_log; some builds reject unknown kwargs ?? try fallbacks.
            _last: Exception | None = None
            for kwargs in (
                {"use_angle_cls": True, "lang": "ch"},
                {"lang": "ch"},
                {},
            ):
                try:
                    _paddle_ocr_instance = PaddleOCR(**kwargs)
                    _last = None
                    break
                except Exception as e:
                    _last = e
                    _paddle_ocr_instance = None
                    continue
            if _paddle_ocr_instance is None:
                raise _last if _last is not None else RuntimeError("PaddleOCR() failed")
        except Exception as e:
            _paddle_ocr_failed = True
            if not _paddle_ocr_warned:
                _paddle_ocr_warned = True
                print(
                    "[RAG-Lite] PaddleOCR unavailable (install paddlepaddle + paddleocr; "
                    f"see requirements-paddleocr.txt): {e}",
                    flush=True,
                )
            return ""

    try:
        ocr = _paddle_ocr_instance
        lines: list[str] = []

        def _run(img_arr: Any) -> list[str]:
            r: Any = None
            # 3.x: ocr() forwards to predict() ¡ª do not pass cls= (predict() rejects it).
            # 2.x: optional cls=True for angle classifier.
            if hasattr(ocr, "ocr") and callable(getattr(ocr, "ocr")):
                try:
                    r = ocr.ocr(img_arr)
                except Exception:
                    try:
                        r = ocr.ocr(img_arr, cls=True)
                    except Exception:
                        r = None
            if r is None and hasattr(ocr, "predict") and callable(getattr(ocr, "predict")):
                try:
                    r = ocr.predict(img_arr)
                except Exception:
                    r = None
            if r is None:
                return []
            return _lines_from_paddle_result(r)

        lines = _run(arr_rgb)
        # If RGB yields no text, retry with BGR (common OpenCV convention).
        if not lines:
            try:
                bgr = arr_rgb[:, :, ::-1].copy()
                lines = _run(bgr)
            except Exception:
                pass

        return "\n".join(lines).strip()
    except Exception as e:
        if not _paddle_ocr_warned:
            _paddle_ocr_warned = True
            print(f"[RAG-Lite] PaddleOCR inference failed: {e}", flush=True)
        return ""


def ocr_image_bytes(image_bytes: bytes, opts: ImageEnrichOptions) -> str:
    eng = (opts.ocr_engine or "tesseract").strip().lower()
    if eng == "paddleocr":
        return _ocr_paddleocr_bytes(image_bytes, opts)
    return _ocr_tesseract_bytes(image_bytes, opts)


def ollama_vision_caption(image_bytes: bytes, opts: ImageEnrichOptions) -> str:
    """
    Prefer POST /api/chat (recommended for vision models); fallback /api/generate.
    Raw screenshots can be huge and break /generate for some builds; we JPEG-shrink first.
    """
    try:
        side = min(1280, int(opts.max_image_side_px))
        side = max(256, side)
        small = _jpeg_bytes_for_vision(image_bytes, side)
    except Exception as e:
        _ollama_vision_warn_once(f"vision: could not prepare image ({e}); check Pillow.")
        return ""

    b64 = base64.b64encode(small).decode("ascii")
    base = opts.base_url.rstrip("/")
    model = (opts.vision_model or "").strip()
    if not model:
        return ""
    prompt = opts.vision_prompt or ""
    timeout = min(float(opts.request_timeout), 120.0)

    def _post_json(path: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        url = base + path
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        return json.loads(raw)

    # 1) /api/chat ?? Ollama docs recommend this for multimodal
    try:
        chat_payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt, "images": [b64]}],
            "stream": False,
        }
        j = _post_json("/api/chat", chat_payload)
        if isinstance(j, dict):
            err = j.get("error")
            if err:
                _ollama_vision_warn_once(f"Ollama /api/chat error: {err}")
            msg = j.get("message")
            if isinstance(msg, dict):
                out = (msg.get("content") or "").strip()
                if out:
                    return out
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            pass
        _ollama_vision_warn_once(f"Ollama /api/chat HTTP {e.code}: {body or e.reason}")
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError, ValueError, TypeError) as e:
        _ollama_vision_warn_once(f"Ollama /api/chat failed: {e}")

    # 2) /api/generate ?? legacy multimodal
    try:
        gen_payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "images": [b64],
            "stream": False,
        }
        j2 = _post_json("/api/generate", gen_payload)
        if isinstance(j2, dict):
            err = j2.get("error")
            if err:
                _ollama_vision_warn_once(f"Ollama /api/generate error: {err}")
            out = (j2.get("response") or "").strip()
            if out:
                return out
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            pass
        _ollama_vision_warn_once(f"Ollama /api/generate HTTP {e.code}: {body or e.reason}")
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError, ValueError, TypeError):
        pass

    return ""


def hybrid_image_to_text(
    image_bytes: bytes,
    opts: ImageEnrichOptions,
    *,
    allow_vision: bool = True,
) -> tuple[str, str]:
    """
    Returns (text_block, tag) where tag is 'ocr' | 'vision' | 'ocr+vision' | ''.
    """
    ocr = ocr_image_bytes(image_bytes, opts)
    if len(ocr) >= opts.ocr_skip_vlm_min_chars:
        return ocr, "ocr" if ocr else ""
    if not allow_vision or not opts.vision_model:
        return ocr, "ocr" if ocr else ""
    vision = ollama_vision_caption(image_bytes, opts)
    vision = vision.strip()
    if ocr and vision:
        return f"{ocr}\n{vision}", "ocr+vision"
    if vision:
        return vision, "vision"
    return ocr, "ocr" if ocr else ""
