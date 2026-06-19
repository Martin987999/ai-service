"""Character-based chunking with overlap (robust for mixed CN/EN).

中英混合按字符切分更稳;优先在句界(。.!?\n)切,带 overlap 保上下文连续。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .loader import RawDoc
from .lang import detect_lang


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    source: str
    lang: str
    meta: dict = field(default_factory=dict)


_SENT_BOUND = re.compile(r"(?<=[。.!?！？\n])")


def chunk_docs(docs: list[RawDoc], chunk_size: int, overlap: int, min_chars: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for d in docs:
        for i, piece in enumerate(_split(d.text, chunk_size, overlap, min_chars)):
            chunks.append(
                Chunk(
                    chunk_id=f"{d.doc_id}::{i}",
                    doc_id=d.doc_id,
                    text=piece,
                    source=d.source,
                    lang=detect_lang(piece) if d.lang == "unknown" else d.lang,
                    meta=dict(d.meta),
                )
            )
    return chunks


def _split(text: str, size: int, overlap: int, min_chars: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    # build by accumulating sentence-ish spans until size, then back off by overlap
    spans = [s for s in _SENT_BOUND.split(text) if s]
    out: list[str] = []
    buf = ""
    for s in spans:
        if len(buf) + len(s) <= size:
            buf += s
        else:
            if buf.strip():
                out.append(buf.strip())
            # start next buffer with tail overlap of previous
            tail = buf[-overlap:] if overlap and buf else ""
            buf = tail + s
            # a single span longer than size: hard-wrap it
            while len(buf) > size:
                out.append(buf[:size].strip())
                buf = buf[size - overlap :]
    if buf.strip():
        out.append(buf.strip())
    return [c for c in out if len(c) >= min_chars] or [text[:size]]
