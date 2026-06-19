"""Structured JSON logging + per-request trace context.

结构化日志:每条 QA 请求一个 trace_id,贯穿检索/生成/缓存/安全各阶段。
字段字典见 docs/LOG_FIELDS.md。日志会按配置做 PII 脱敏。
"""
from __future__ import annotations

import contextvars
import json
import logging
import sys
import time
import uuid
from pathlib import Path
from typing import Any

# request-scoped trace id (set per request, read everywhere)
_trace_id: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="-")

_PII_REDACTOR = None  # injected lazily to avoid import cycle


def new_trace_id() -> str:
    tid = uuid.uuid4().hex[:16]
    _trace_id.set(tid)
    return tid


def current_trace_id() -> str:
    return _trace_id.get()


def set_pii_redactor(fn) -> None:
    global _PII_REDACTOR
    _PII_REDACTOR = fn


class JsonFormatter(logging.Formatter):
    """Render each log record as a single JSON line."""

    RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys())

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created))
            + f".{int(record.msecs):03d}",
            "level": record.levelname,
            "logger": record.name,
            "trace_id": current_trace_id(),
            "event": record.getMessage(),
        }
        # merge any structured fields passed via `extra={"fields": {...}}`
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            payload.update(fields)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)

        if _PII_REDACTOR is not None:
            try:
                payload = _PII_REDACTOR(payload)
            except Exception:  # never let logging crash the request
                pass
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(level: str = "INFO", json_mode: bool = True, file: str | None = None) -> logging.Logger:
    root = logging.getLogger("rag")
    root.setLevel(level.upper())
    root.handlers.clear()
    root.propagate = False

    fmt: logging.Formatter = JsonFormatter() if json_mode else logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s"
    )

    # Ensure stdout can emit CJK (Windows consoles default to cp1252 → UnicodeEncodeError).
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # py3.7+
    except Exception:
        pass
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    if file:
        Path(file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(file, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
    return root


def log_event(logger: logging.Logger, level: str, event: str, **fields: Any) -> None:
    """Helper: emit a structured event with arbitrary fields."""
    logger.log(getattr(logging, level.upper(), logging.INFO), event, extra={"fields": fields})


def get_logger(name: str = "rag") -> logging.Logger:
    return logging.getLogger(name)
