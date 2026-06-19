"""Retriever: orchestrates vector / hybrid retrieval + optional reranking.

模式与开关全部来自配置(retrieval.mode, retrieval.reranker_enabled),改配置不改代码。
返回最终上下文块 + 置信度(用于拒答判定)。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..providers.embeddings import EmbeddingProvider
from .store import IndexStore


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    text: str
    source: str
    lang: str
    score: float
    stage: str  # "vector" | "bm25" | "fused" | "rerank"


@dataclass
class RetrievalOutcome:
    mode: str
    reranked: bool
    chunks: list[RetrievedChunk]
    confidence: float  # top score after the final stage (rerank or fused)
    debug: dict = field(default_factory=dict)


class Retriever:
    def __init__(self, store: IndexStore, embedder: EmbeddingProvider, cfg: dict):
        self.store = store
        self.embedder = embedder
        self.cfg = cfg

    def retrieve(self, query: str, *, mode: str | None = None, rerank: bool | None = None) -> RetrievalOutcome:
        c = self.cfg
        mode = mode or c["mode"]
        rerank = c["reranker_enabled"] if rerank is None else rerank

        # ---- stage 1: recall ----
        vector_hits = self.store.search_vector(query, c["top_k_vector"])
        # absolute top cosine (L2-normalized vectors → inner product ∈ [-1,1]); used
        # as the confidence floor signal because min-max normalized fused scores are
        # relative (top is always ~1.0) and useless for out-of-scope detection.
        raw_vec_top = vector_hits[0][1] if vector_hits else 0.0
        if mode == "vector":
            fused = vector_hits[: c["top_k_fused"]]
            fused_norm = _normalize_scores(fused)
        elif mode == "hybrid":
            bm25_hits = self.store.search_bm25(query, c["top_k_bm25"])
            fused_norm = _rrf_fuse(vector_hits, bm25_hits, c["rrf_k"], c["top_k_fused"])
        else:
            raise ValueError(f"unknown retrieval mode: {mode}")

        candidates = [self._mk(idx, score, "fused") for idx, score in fused_norm]

        # ---- stage 2: rerank (optional) ----
        reranked = False
        if rerank and candidates:
            docs = [rc.text for rc in candidates]
            hits = self.embedder.rerank(query, docs, top_k=c["top_k_rerank"])
            reranked = True
            new_list = []
            for h in hits:
                rc = candidates[h.index]
                new_list.append(
                    RetrievedChunk(rc.chunk_id, rc.doc_id, rc.text, rc.source, rc.lang, float(h.score), "rerank")
                )
            candidates = new_list

        final = candidates[: c["top_k_context"]]
        # Confidence = absolute signal: reranker relevance if reranked, else raw top cosine.
        # (Never the min-max normalized fused score, which is always ~1.0 at the top.)
        if not final:
            confidence = 0.0
        elif reranked:
            confidence = final[0].score
        else:
            confidence = raw_vec_top
        return RetrievalOutcome(
            mode=mode,
            reranked=reranked,
            chunks=final,
            confidence=confidence,
            debug={
                "n_vector": len(vector_hits),
                "n_candidates": len(candidates),
                "raw_vec_top": round(raw_vec_top, 4),
            },
        )

    def _mk(self, idx: int, score: float, stage: str) -> RetrievedChunk:
        ch = self.store.chunks[idx]
        return RetrievedChunk(ch.chunk_id, ch.doc_id, ch.text, ch.source, ch.lang, float(score), stage)


def _normalize_scores(hits: list[tuple[int, float]]) -> list[tuple[int, float]]:
    if not hits:
        return []
    scores = [s for _, s in hits]
    lo, hi = min(scores), max(scores)
    rng = (hi - lo) or 1.0
    return [(i, (s - lo) / rng) for i, s in hits]


def _rrf_fuse(
    a: list[tuple[int, float]], b: list[tuple[int, float]], k: int, top_n: int
) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion. Score = sum 1/(k + rank). Returns normalized [0,1]."""
    rrf: dict[int, float] = {}
    for ranked in (a, b):
        for rank, (idx, _s) in enumerate(ranked):
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (k + rank + 1)
    fused = sorted(rrf.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    return _normalize_scores(fused)
