"""Basic PII detection & redaction for outputs and logs.

基础 PII 脱敏:邮箱、电话、身份证、银行卡、IP、信用卡。
用于:① 最终答案输出;② 结构化日志。规则可扩展。
"""
from __future__ import annotations

import re
from typing import Any

# Each pattern -> placeholder. Order matters (longer/more-specific first).
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("[EMAIL]", re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")),
    # China resident ID (18 digits, last may be X)
    ("[ID_CARD]", re.compile(r"\b\d{17}[\dXx]\b")),
    # bank / credit card (13-19 digits, optional spaces/dashes)
    ("[CARD]", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    # China mobile (11 digits starting 1) and generic intl phone
    ("[PHONE]", re.compile(r"\b1[3-9]\d{9}\b")),
    ("[PHONE]", re.compile(r"(?<!\d)\+?\d{1,3}[ -]?\(?\d{2,4}\)?[ -]?\d{3,4}[ -]?\d{3,4}(?!\d)")),
    ("[IP]", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
]


def redact_text(text: str) -> tuple[str, dict[str, int]]:
    """Redact PII in a string. Returns (redacted_text, counts_by_type)."""
    if not text:
        return text, {}
    counts: dict[str, int] = {}
    out = text
    for placeholder, pattern in _PATTERNS:
        out, n = pattern.subn(placeholder, out)
        if n:
            counts[placeholder] = counts.get(placeholder, 0) + n
    return out, counts


def redact_obj(obj: Any) -> Any:
    """Recursively redact PII in a dict/list/str structure (used for logs)."""
    if isinstance(obj, str):
        return redact_text(obj)[0]
    if isinstance(obj, dict):
        return {k: redact_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact_obj(v) for v in obj]
    return obj


def has_pii(text: str) -> bool:
    return any(p.search(text) for _, p in _PATTERNS)
