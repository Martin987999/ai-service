"""Index manager: builds & persists vector (FAISS) + BM25 indices over chunks.

构建并持久化两路索引:FAISS(向量)+ BM25(词法)。
检索模式在查询时选择,索引一次构建即可同时支持 vector / hybrid。
"""
from __future__ import annotations

import json
import pickle
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..ingestion.chunker import Chunk
from ..providers.embeddings import EmbeddingProvider
from .bm25 import BM25Index


class IndexStore:
    def __init__(self, index_dir: str, embedder: EmbeddingProvider):
        self.index_dir = Path(index_dir)
        self.embedder = embedder
        self.chunks: list[Chunk] = []
        self._vectors: np.ndarray | None = None
        self._faiss = None
        self.bm25: BM25Index | None = None

    # ---------- build ----------
    def build(self, chunks: list[Chunk], batch_size: int = 64) -> None:
        self.chunks = chunks
        texts = [c.text for c in chunks]
        self._vectors = self.embedder.embed(texts, input_type="document", batch_size=batch_size)
        self._build_faiss()
        self.bm25 = BM25Index([c.text for c in chunks])

    def _build_faiss(self) -> None:
        try:
            import faiss

            dim = self._vectors.shape[1]
            index = faiss.IndexFlatIP(dim)  # inner product on L2-normalized = cosine
            index.add(self._vectors)
            self._faiss = index
        except Exception:
            self._faiss = None  # fall back to numpy brute force

    # ---------- search ----------
    def search_vector(self, query: str, top_k: int) -> list[tuple[int, float]]:
        qv = self.embedder.embed_query(query).astype(np.float32).reshape(1, -1)
        if self._faiss is not None:
            scores, idxs = self._faiss.search(qv, min(top_k, len(self.chunks)))
            return [(int(i), float(s)) for i, s in zip(idxs[0], scores[0]) if i >= 0]
        sims = (self._vectors @ qv[0])
        order = np.argsort(-sims)[:top_k]
        return [(int(i), float(sims[i])) for i in order]

    def search_bm25(self, query: str, top_k: int) -> list[tuple[int, float]]:
        assert self.bm25 is not None
        return self.bm25.search(query, top_k)

    def text(self, idx: int) -> str:
        return self.chunks[idx].text

    # ---------- persistence ----------
    def save(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.index_dir / "vectors.npy", self._vectors)
        with open(self.index_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
        with open(self.index_dir / "bm25.pkl", "wb") as f:
            pickle.dump(self.bm25, f)

    def load(self) -> bool:
        try:
            self._vectors = np.load(self.index_dir / "vectors.npy")
            self.chunks = [
                Chunk(**json.loads(line))
                for line in (self.index_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            with open(self.index_dir / "bm25.pkl", "rb") as f:
                self.bm25 = pickle.load(f)
            self._build_faiss()
            return True
        except Exception:
            return False

    @property
    def size(self) -> int:
        return len(self.chunks)
