"""Application bootstrap: wire providers, index, retriever, cache, pipeline.

被 FastAPI 服务与评估脚本共用。负责加载配置/索引并组装 RAGPipeline。
"""
from __future__ import annotations

from .cache import AnswerCache
from .config import Settings, load_settings
from .logging_setup import set_pii_redactor, setup_logging
from .pii import redact_obj
from .pipeline import RAGPipeline
from .providers.embeddings import EmbeddingProvider
from .providers.llm import LLMProvider
from .retrieval.retriever import Retriever
from .retrieval.store import IndexStore
from .session import SessionStore


def build_pipeline(settings: Settings | None = None) -> RAGPipeline:
    s = settings or load_settings()

    # logging (with PII redaction on log payloads if enabled)
    lg = s.get("logging", default={})
    setup_logging(level=s.log_level, json_mode=lg.get("json", True), file=lg.get("file"))
    if s.get("safety", "pii_redaction_logs", default=True):
        set_pii_redactor(redact_obj)

    embedder = EmbeddingProvider(
        api_key=s.voyage_api_key,
        model=s.get("models", "embedding", "model", default="voyage-3"),
        dim=s.get("models", "embedding", "dim", default=1024),
        rerank_model=s.get("models", "reranker", "model", default="rerank-2"),
        allow_mock=s.allow_mock_fallback,
    )
    llm = LLMProvider(api_key=s.anthropic_api_key, allow_mock=s.allow_mock_fallback)

    store = IndexStore(index_dir=s.get("paths", "index_dir"), embedder=embedder)
    if not store.load():
        raise RuntimeError(
            "Index not found. Run `python -m scripts.ingest` first to build the index."
        )

    retriever = Retriever(store, embedder, s.get("retrieval"))

    ccfg = s.get("cache")
    cache = AnswerCache(
        enabled=ccfg["enabled"], ttl_s=ccfg["exact_ttl_s"], max_entries=ccfg["max_entries"],
        semantic_enabled=ccfg["semantic_enabled"], semantic_threshold=ccfg["semantic_threshold"],
    )

    session_store = SessionStore(
        max_turns=s.get("session", "max_turns", default=12),
        ttl_s=s.get("session", "ttl_s", default=3600),
        max_sessions=s.get("session", "max_sessions", default=2000),
    )

    return RAGPipeline(s, retriever, llm, embedder, cache, session_store)
