"""Ingest corpus -> chunk -> embed -> build & persist indices.

一键构建索引:加载 data/corpus -> 切分 -> Voyage 向量化 -> FAISS + BM25 -> 落盘 data/index。
"""
from __future__ import annotations

import time

from src.config import load_settings
from src.ingestion.chunker import chunk_docs
from src.ingestion.loader import load_dir
from src.providers.embeddings import EmbeddingProvider
from src.retrieval.store import IndexStore


def main() -> None:
    s = load_settings()
    t0 = time.perf_counter()

    docs = load_dir(s.get("paths", "corpus_dir"))
    if not docs:
        raise SystemExit(
            "No documents found. Run `python -m scripts.make_corpus` first to create the corpus."
        )
    print(f"[ingest] loaded {len(docs)} docs")

    cc = s.get("chunking")
    chunks = chunk_docs(docs, cc["chunk_size"], cc["chunk_overlap"], cc["min_chunk_chars"])
    print(f"[ingest] produced {len(chunks)} chunks")

    embedder = EmbeddingProvider(
        api_key=s.voyage_api_key,
        model=s.get("models", "embedding", "model", default="voyage-3"),
        dim=s.get("models", "embedding", "dim", default=1024),
        rerank_model=s.get("models", "reranker", "model", default="rerank-2"),
        allow_mock=s.allow_mock_fallback,
    )
    if embedder.is_mock:
        print("[ingest] WARNING: VOYAGE_API_KEY not set — using mock hashed embeddings.")

    store = IndexStore(index_dir=s.get("paths", "index_dir"), embedder=embedder)
    store.build(chunks, batch_size=s.get("models", "embedding", "batch_size", default=64))
    store.save()
    print(f"[ingest] index built & saved -> {s.get('paths', 'index_dir')} "
          f"({store.size} chunks, {time.perf_counter() - t0:.1f}s)")


if __name__ == "__main__":
    main()
