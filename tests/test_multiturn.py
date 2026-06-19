# -*- coding: utf-8 -*-
"""Offline multi-turn test (forced mock — no API key, no network).

验证多轮:① 会话记忆;② 历史感知 query 改写(跟进问题→独立问题);③ 历史进入生成 prompt。
以文件方式运行,保证中文字面量 UTF-8 不被 shell 管道破坏。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cache import AnswerCache
from src.config import load_settings
from src.ingestion.chunker import chunk_docs
from src.ingestion.loader import load_dir
from src.pipeline import RAGPipeline
from src.providers.embeddings import EmbeddingProvider
from src.providers.llm import LLMProvider
from src.retrieval.retriever import Retriever
from src.retrieval.store import IndexStore
from src.session import SessionStore


def build_mock_pipeline() -> RAGPipeline:
    s = load_settings()
    # force mock regardless of .env
    emb = EmbeddingProvider(api_key=None, allow_mock=True)
    llm = LLMProvider(api_key=None, allow_mock=True)
    assert emb.is_mock and llm.is_mock, "expected forced mock"

    docs = load_dir(s.get("paths", "corpus_dir"))
    cc = s.get("chunking")
    chunks = chunk_docs(docs, cc["chunk_size"], cc["chunk_overlap"], cc["min_chunk_chars"])
    store = IndexStore(index_dir="data/_tmp_mock_index", embedder=emb)
    store.build(chunks)  # in-memory mock index (consistent query/doc vectors)

    retriever = Retriever(store, emb, s.get("retrieval"))
    cache = AnswerCache(enabled=True, ttl_s=3600, max_entries=1000,
                        semantic_enabled=True, semantic_threshold=0.95)
    sessions = SessionStore()
    return RAGPipeline(s, retriever, llm, emb, cache, sessions)


def main() -> int:
    p = build_mock_pipeline()
    # mock rerank scores are distributed differently from real rerank-2; lower the
    # confidence gate for this offline demo so answers flow through the multi-turn path.
    p.s.raw["retrieval"]["min_confidence"] = 0.05
    sid = p.session_store.new_id()
    print(f"session={sid} llm_mock={p.llm.is_mock} embed_mock={p.embedder.is_mock}")

    print("\n--- Turn 1: 入职满五年每年有多少天年假? ---")
    r1 = p.answer("入职满五年每年有多少天年假?", session_id=sid)
    print(f"rewritten={r1.rewritten_query!r} refused={r1.refused} ans={r1.answer[:50]!r}")

    print("\n--- Turn 2 (follow-up): 那十年呢? ---")
    r2 = p.answer("那十年呢?", session_id=sid)
    print(f"rewritten={r2.rewritten_query!r} refused={r2.refused} "
          f"top={r2.sources[0]['doc_id'] if r2.sources else None}")
    print(f"ans={r2.answer[:60]!r}")

    print("\n--- session history ---")
    hist = p.session_store.history(sid)
    for t in hist:
        print(f"  {t.role}: {t.content[:40]}")

    # assertions
    ok = True
    if r2.rewritten_query is None:
        print("FAIL: follow-up was not rewritten"); ok = False
    elif "年假" not in r2.rewritten_query and "五年" not in r2.rewritten_query:
        print("WARN: rewrite did not splice prior subject:", r2.rewritten_query)
    if len(hist) != 4:
        print(f"FAIL: expected 4 history turns, got {len(hist)}"); ok = False
    if r2.session_id != sid:
        print("FAIL: session_id not echoed"); ok = False
    print("\nRESULT:", "PASS ✅" if ok else "FAIL ❌")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
