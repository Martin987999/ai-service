"""RAG QA pipeline orchestration.

单次问答全流程:
  trace 开始 → 安全检查(注入) → 缓存查 → 检索(模式/重排) → 置信度/拒答判定
  → grounded 生成 → 拒答兜底 → PII 脱敏 → 缓存写 → 结构化日志 + 指标。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .cache import AnswerCache, cache_key
from .config import Settings
from .ingestion.lang import detect_lang
from .logging_setup import get_logger, log_event, new_trace_id
from .pii import redact_text
from .prompts import REFUSAL_SENTINEL, build_answer_prompt
from .providers.embeddings import EmbeddingProvider
from .providers.llm import LLMProvider
from .retrieval.retriever import Retriever
from .rewrite import condense_query
from .safety import detect_injection, sanitize_context, should_refuse_low_confidence
from .session import SessionStore, Turn


@dataclass
class QAResponse:
    trace_id: str
    answer: str
    refused: bool
    refusal_reason: str | None
    lang: str
    mode: str
    reranked: bool
    confidence: float
    sources: list[dict] = field(default_factory=list)
    cache_hit: str = "miss"
    latency_ms: float = 0.0
    usage: dict = field(default_factory=dict)
    pii_redactions: dict = field(default_factory=dict)
    is_mock: bool = False
    session_id: str | None = None       # multi-turn session id
    rewritten_query: str | None = None  # standalone query used for retrieval (if rewritten)


class RAGPipeline:
    def __init__(self, settings: Settings, retriever: Retriever, llm: LLMProvider,
                 embedder: EmbeddingProvider, cache: AnswerCache,
                 session_store: SessionStore | None = None):
        self.s = settings
        self.retriever = retriever
        self.llm = llm
        self.embedder = embedder
        self.cache = cache
        self.session_store = session_store
        self.log = get_logger("rag.pipeline")

    def answer(self, query: str, *, mode: str | None = None, rerank: bool | None = None,
               session_id: str | None = None, history: list[Turn] | None = None,
               include_context_text: bool = False) -> QAResponse:
        t0 = time.perf_counter()
        trace_id = new_trace_id()
        rcfg = self.s.get("retrieval")
        scfg = self.s.get("safety")
        mode = mode or rcfg["mode"]
        rerank = rcfg["reranker_enabled"] if rerank is None else rerank
        lang = detect_lang(query)

        # ---- multi-turn: resolve history (passed in, or from server-side session store) ----
        if history is None and session_id and self.session_store:
            history = self.session_store.history(session_id)
        history = history or []
        rewritten = query  # updated after the injection check

        def finalize(resp: QAResponse) -> QAResponse:
            resp.session_id = session_id
            resp.rewritten_query = rewritten if rewritten != query else None
            # persist the turn (skip injection attempts so they don't pollute history)
            if session_id and self.session_store and resp.refusal_reason != "prompt_injection":
                self.session_store.append(session_id, query, resp.answer)
            return resp

        log_event(self.log, "INFO", "request.start", query_lang=lang, mode=mode, rerank=rerank,
                  multi_turn=bool(history), query_preview=redact_text(query)[0][:120])

        # ---- 1. safety: prompt injection ----
        verdict = detect_injection(query)
        if scfg.get("prompt_injection_defense", True) and verdict.blocked:
            return finalize(self._refuse(trace_id, lang, mode, rerank, 0.0, "prompt_injection", t0))

        # ---- 1b. history-aware query rewrite (condense follow-up → standalone) ----
        rewritten = condense_query(
            self.llm, self.s.get("models", "generation", "cheap_model"), history, query)
        if rewritten != query:
            log_event(self.log, "INFO", "query.rewritten",
                      original=redact_text(query)[0][:80], rewritten=redact_text(rewritten)[0][:120])

        # ---- 2. cache lookup (keyed on the standalone query so follow-ups don't collide) ----
        key = cache_key(rewritten, mode, rerank)
        qvec = None
        if self.cache.semantic_enabled:
            qvec = self.embedder.embed_query(rewritten)
        cached, hit_type = self.cache.get(key, qvec)
        if cached is not None:
            cached = dict(cached)
            cached["cache_hit"] = hit_type
            cached["trace_id"] = trace_id
            cached["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            log_event(self.log, "INFO", "cache.hit", hit_type=hit_type,
                      latency_ms=cached["latency_ms"])
            return finalize(QAResponse(**{k: v for k, v in cached.items()
                                          if k in QAResponse.__annotations__}))

        # ---- 3. retrieval (uses the standalone query) ----
        outcome = self.retriever.retrieve(rewritten, mode=mode, rerank=rerank)
        log_event(self.log, "INFO", "retrieval.done", mode=outcome.mode, reranked=outcome.reranked,
                  n_chunks=len(outcome.chunks), confidence=round(outcome.confidence, 4),
                  **outcome.debug)

        # ---- 4. refusal: low confidence / out-of-scope ----
        if not outcome.chunks or should_refuse_low_confidence(outcome.confidence, rcfg["min_confidence"]):
            return finalize(self._refuse(trace_id, lang, outcome.mode, outcome.reranked,
                                         outcome.confidence, "low_confidence", t0))

        # ---- 5. grounded generation (original query + history for coherence) ----
        contexts = [(c.doc_id, sanitize_context(c.text)) for c in outcome.chunks]
        hist_pairs = [(t.role, t.content) for t in history]
        system, user = build_answer_prompt(query, contexts, lang, history=hist_pairs)
        gen = self.llm.complete(
            system=system, user=user,
            model=self.s.get("models", "generation", "answer_model"),
            max_tokens=self.s.get("models", "generation", "max_tokens", default=1024),
            effort=self.s.get("models", "generation", "effort", default="medium"),
        )

        # ---- 6. refusal sentinel from the model ----
        if gen.text.strip().upper().startswith(REFUSAL_SENTINEL):
            return finalize(self._refuse(trace_id, lang, outcome.mode, outcome.reranked,
                                         outcome.confidence, "model_refusal", t0, usage=_usage(gen)))

        # ---- 7. PII redaction on output ----
        answer = gen.text
        pii_counts: dict = {}
        if scfg.get("pii_redaction_outputs", True):
            answer, pii_counts = redact_text(answer)

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        resp = QAResponse(
            trace_id=trace_id, answer=answer, refused=False, refusal_reason=None,
            lang=lang, mode=outcome.mode, reranked=outcome.reranked, confidence=outcome.confidence,
            sources=[{"doc_id": c.doc_id, "chunk_id": c.chunk_id, "score": round(c.score, 4),
                      "source": c.source,
                      **({"text": c.text} if include_context_text else {})}
                     for c in outcome.chunks],
            cache_hit="miss", latency_ms=latency_ms, usage=_usage(gen),
            pii_redactions=pii_counts, is_mock=gen.is_mock,
        )

        # ---- 8. cache store (before finalize, so session-specific fields aren't cached) ----
        self.cache.put(key, _to_cacheable(resp), qvec)

        log_event(self.log, "INFO", "request.done", refused=False, latency_ms=latency_ms,
                  input_tokens=gen.input_tokens, output_tokens=gen.output_tokens,
                  cost_usd=round(gen.cost_usd, 6), pii_redactions=pii_counts,
                  n_sources=len(resp.sources), is_mock=gen.is_mock)
        return finalize(resp)

    # ---------------------------------------------------------------
    def _refuse(self, trace_id, lang, mode, rerank, confidence, reason, t0,
                cache_hit="miss", usage=None) -> QAResponse:
        msg = self.s.get("safety", f"refusal_message_{lang if lang in ('zh', 'en') else 'en'}")
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        log_event(self.log, "WARNING", "request.refused", reason=reason, confidence=round(confidence, 4),
                  mode=mode, latency_ms=latency_ms)
        return QAResponse(
            trace_id=trace_id, answer=msg, refused=True, refusal_reason=reason, lang=lang,
            mode=mode, reranked=rerank, confidence=confidence, sources=[], cache_hit=cache_hit,
            latency_ms=latency_ms, usage=usage or {},
        )


def _usage(gen) -> dict:
    return {"model": gen.model, "input_tokens": gen.input_tokens, "output_tokens": gen.output_tokens,
            "cost_usd": round(gen.cost_usd, 6)}


def _to_cacheable(resp: QAResponse) -> dict:
    d = resp.__dict__.copy()
    for k in ("trace_id", "latency_ms", "cache_hit", "session_id", "rewritten_query"):
        d.pop(k, None)
    return d
