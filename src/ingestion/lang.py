"""Tiny language detector (CN vs EN) — no external dependency.

按 CJK 字符占比判断中英文。用于:① 文档/分块语言标注;② 答案语言对齐(回答语言跟随提问语言)。
"""
from __future__ import annotations

import re

_CJK = re.compile(r"[一-鿿]")
_LATIN = re.compile(r"[A-Za-z]")


def detect_lang(text: str) -> str:
    if not text:
        return "unknown"
    cjk = len(_CJK.findall(text))
    latin = len(_LATIN.findall(text))
    if cjk == 0 and latin == 0:
        return "unknown"
    if cjk >= max(1, latin):
        return "zh"
    return "en"
