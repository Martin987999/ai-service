"""FastAPI service.

接口:
  GET  /health        健康检查 + 索引/provider 状态
  POST /qa            单次问答(支持 per-request 覆盖 mode/rerank)
  GET  /metrics       运行期指标快照(缓存命中率、累计请求/拒答、token、成本)

并发:单实例线程池 + 信号量限流(对应「>=5 并发」「90% < 10s」约束)。
"""
from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .app import build_pipeline
from .config import load_settings
from .session import Turn

settings = load_settings()
pipeline = build_pipeline(settings)

_max_conc = settings.get("service", "max_concurrency", default=8)
_executor = ThreadPoolExecutor(max_workers=_max_conc)
_semaphore = asyncio.Semaphore(_max_conc)

app = FastAPI(title=settings.get("service", "name", default="bilingual-rag-qa"))

# ---- lightweight rolling metrics (for /metrics and ops report) ----
import threading

_METRICS = {
    "requests": 0, "refusals": 0, "errors": 0,
    "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0,
    "latencies_ms": [],  # capped
}
_LAT_CAP = 5000
# guards _METRICS: _record() runs on the event loop, but the sync /metrics endpoint
# runs in FastAPI's threadpool and sorts latencies_ms — concurrent mutation otherwise.
_metrics_lock = threading.Lock()


class Msg(BaseModel):
    role: str
    content: str


class QARequest(BaseModel):
    query: str = Field(..., min_length=1)
    mode: Optional[str] = Field(None, description="vector | hybrid (覆盖默认)")
    rerank: Optional[bool] = Field(None, description="覆盖默认重排开关")
    session_id: Optional[str] = Field(None, description="多轮会话 ID(服务端记忆);留空=单轮")
    history: Optional[list[Msg]] = Field(None, description="无状态多轮:直接传历史 (role/content)")


@app.post("/session")
def new_session():
    """Mint a multi-turn session id. 客户端用它在后续 /qa 里传 session_id 维持多轮。"""
    return {"session_id": pipeline.session_store.new_id()}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "index_size": pipeline.retriever.store.size,
        "llm_mock": pipeline.llm.is_mock,
        "embed_mock": pipeline.embedder.is_mock,
        "default_mode": settings.get("retrieval", "mode"),
        "reranker_enabled": settings.get("retrieval", "reranker_enabled"),
    }


@app.post("/qa")
async def qa(req: QARequest):
    async with _semaphore:
        loop = asyncio.get_event_loop()
        hist = [Turn(m.role, m.content) for m in req.history] if req.history else None
        try:
            resp = await loop.run_in_executor(
                _executor,
                lambda: pipeline.answer(req.query, mode=req.mode, rerank=req.rerank,
                                        session_id=req.session_id, history=hist),
            )
        except Exception as e:
            with _metrics_lock:
                _METRICS["errors"] += 1
            return {"error": str(e)}
        _record(resp)
        return resp.__dict__


@app.get("/metrics")
def metrics():
    with _metrics_lock:  # snapshot under lock to avoid sorting a list being appended to
        lats = sorted(_METRICS["latencies_ms"])
        m = dict(_METRICS); m.pop("latencies_ms", None)
    return {
        "requests": m["requests"],
        "refusals": m["refusals"],
        "refusal_rate": round(m["refusals"] / m["requests"], 4) if m["requests"] else 0.0,
        "errors": m["errors"],
        "p50_latency_ms": _pct(lats, 50),
        "p95_latency_ms": _pct(lats, 95),
        "input_tokens": m["input_tokens"],
        "output_tokens": m["output_tokens"],
        "cost_usd": round(m["cost_usd"], 6),
        "cache": pipeline.cache.stats(),
    }


def _record(resp) -> None:
    with _metrics_lock:
        _METRICS["requests"] += 1
        if resp.refused:
            _METRICS["refusals"] += 1
        u = resp.usage or {}
        _METRICS["input_tokens"] += u.get("input_tokens", 0)
        _METRICS["output_tokens"] += u.get("output_tokens", 0)
        _METRICS["cost_usd"] += u.get("cost_usd", 0.0)
        lats = _METRICS["latencies_ms"]
        lats.append(resp.latency_ms)
        if len(lats) > _LAT_CAP:
            del lats[: len(lats) - _LAT_CAP]


def _pct(sorted_vals: list[float], p: int) -> float:
    if not sorted_vals:
        return 0.0
    k = max(0, min(len(sorted_vals) - 1, int(round((p / 100) * (len(sorted_vals) - 1)))))
    return round(sorted_vals[k], 2)
