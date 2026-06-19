"""Document loaders for the bilingual KB.

支持:.txt / .md / .jsonl(语料) / .pdf(含扫描件 OCR 回退)。
扫描 PDF:先用 pdfplumber 抽取文本;若页面文本过少,判定为扫描件,
回退到 pdf2image + pytesseract OCR(中英双语)。OCR 依赖系统 tesseract/poppler。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .lang import detect_lang


@dataclass
class RawDoc:
    doc_id: str
    text: str
    source: str
    lang: str = "unknown"
    meta: dict = field(default_factory=dict)


_OCR_MIN_CHARS_PER_PAGE = 40  # below this → assume scanned page → OCR


def load_dir(corpus_dir: str) -> list[RawDoc]:
    docs: list[RawDoc] = []
    root = Path(corpus_dir)
    if not root.exists():
        return docs
    for p in sorted(root.rglob("*")):
        if p.is_dir():
            continue
        suffix = p.suffix.lower()
        try:
            if suffix in {".txt", ".md"}:
                docs.append(_load_text(p))
            elif suffix == ".jsonl":
                docs.extend(_load_jsonl(p))
            elif suffix == ".pdf":
                docs.append(_load_pdf(p))
        except Exception as e:  # keep ingesting other files
            print(f"[loader] skip {p}: {e}")
    return docs


def _load_text(p: Path) -> RawDoc:
    text = p.read_text(encoding="utf-8", errors="ignore")
    return RawDoc(doc_id=p.stem, text=text, source=str(p), lang=detect_lang(text))


def _load_jsonl(p: Path) -> list[RawDoc]:
    """Each line: {"id":..., "text":..., "title":..., "lang":...}."""
    out: list[RawDoc] = []
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        text = obj.get("text") or obj.get("content") or ""
        if not text:
            continue
        title = obj.get("title", "")
        full = f"{title}\n{text}" if title else text
        out.append(
            RawDoc(
                doc_id=str(obj.get("id", f"{p.stem}-{i}")),
                text=full,
                source=str(p),
                lang=obj.get("lang") or detect_lang(full),
                meta={"title": title, **{k: v for k, v in obj.items() if k not in {"text", "content"}}},
            )
        )
    return out


def _load_pdf(p: Path) -> RawDoc:
    pages_text: list[str] = []
    scanned_pages = 0
    try:
        import pdfplumber

        with pdfplumber.open(str(p)) as pdf:
            for page in pdf.pages:
                txt = (page.extract_text() or "").strip()
                if len(txt) < _OCR_MIN_CHARS_PER_PAGE:
                    ocr = _ocr_page(p, page.page_number - 1)
                    if ocr:
                        txt = ocr
                        scanned_pages += 1
                pages_text.append(txt)
    except Exception as e:
        raise RuntimeError(f"pdf parse failed: {e}")
    text = "\n".join(pages_text)
    return RawDoc(
        doc_id=p.stem,
        text=text,
        source=str(p),
        lang=detect_lang(text),
        meta={"scanned_pages": scanned_pages, "is_scanned": scanned_pages > 0},
    )


def _ocr_page(pdf_path: Path, page_index: int) -> str:
    """OCR a single page. Returns '' if OCR deps are unavailable."""
    try:
        from pdf2image import convert_from_path
        import pytesseract

        images = convert_from_path(str(pdf_path), first_page=page_index + 1, last_page=page_index + 1, dpi=200)
        if not images:
            return ""
        # bilingual OCR: Simplified Chinese + English
        return pytesseract.image_to_string(images[0], lang="chi_sim+eng").strip()
    except Exception as e:
        print(f"[loader] OCR unavailable for {pdf_path} p{page_index}: {e}")
        return ""
