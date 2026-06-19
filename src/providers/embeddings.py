"""Voyage embeddings + reranker provider.

在线 embedding (voyage-3, 多语言) 与重排 (rerank-2)。
无 VOYAGE_API_KEY 且允许 mock 时 → 本地确定性哈希向量 / 词重叠重排,保证离线可跑。
"""
from __future__ import annotations

import hashlib
import math
import re
import time
from dataclasses import dataclass

import numpy as np


@dataclass
class RerankHit:
    index: int
    score: float


class EmbeddingProvider:
    def __init__(self, api_key: str | None, model: str = "voyage-3", dim: int = 1024,
                 rerank_model: str = "rerank-2", allow_mock: bool = True):
        self._api_key = api_key
        self._model = model
        self._dim = dim
        self._rerank_model = rerank_model
        self._allow_mock = allow_mock
        self._client = None
        if api_key:
            try:
                import voyageai

                self._client = voyageai.Client(api_key=api_key)
            except Exception:
                self._client = None
        if self._client is None and not allow_mock:
            raise RuntimeError("VOYAGE_API_KEY missing and mock fallback disabled.")

    @property
    def is_mock(self) -> bool:
        return self._client is None

    @property
    def dim(self) -> int:
        return self._dim

    # ---- embeddings ----
    def embed(self, texts: list[str], input_type: str = "document", batch_size: int = 64) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        if self._client is None:
            return np.vstack([self._mock_vec(t) for t in texts])
        vecs: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            r = _with_retry(lambda: self._client.embed(chunk, model=self._model, input_type=input_type))
            vecs.extend(r.embeddings)
        arr = np.asarray(vecs, dtype=np.float32)
        return _l2_normalize(arr)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text], input_type="query")[0]

    # ---- rerank ----
    def rerank(self, query: str, documents: list[str], top_k: int) -> list[RerankHit]:
        if not documents:
            return []
        if self._client is None:
            return self._mock_rerank(query, documents, top_k)
        r = _with_retry(lambda: self._client.rerank(query, documents, model=self._rerank_model, top_k=top_k))
        return [RerankHit(index=item.index, score=float(item.relevance_score)) for item in r.results]

    # ---------------------------------------------------------------
    def _mock_vec(self, text: str) -> np.ndarray:
        """Deterministic hashed bag-of-tokens vector (cosine-meaningful)."""
        v = np.zeros(self._dim, dtype=np.float32)
        for tok in _tokenize(text):
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            v[h % self._dim] += 1.0
        n = np.linalg.norm(v)
        if n > 0:
            v /= n
        return v

    def _mock_rerank(self, query: str, documents: list[str], top_k: int) -> list[RerankHit]:
        q = set(_tokenize(query))
        scored = []
        for i, d in enumerate(documents):
            dt = set(_tokenize(d))
            inter = len(q & dt)
            denom = math.sqrt(len(q) * len(dt)) or 1.0
            scored.append(RerankHit(index=i, score=inter / denom))
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:top_k]


def _with_retry(fn, max_attempts: int = 6, base_delay: float = 22.0):
    """Self-healing retry for Voyage calls under the free-tier 3 RPM rolling-window limit.

    必须比固定预计算节流更稳:免费层是滚动窗口,任何旁路调用(手动测试、并发请求)都会
    叠加占用配额,固定 sleep 间隔仍可能撞 429。命中限速时指数退避重试,而不是让整次评估崩掉。
    """
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            is_rate_limit = "RateLimit" in type(e).__name__ or "429" in str(e)
            if not is_rate_limit or attempt == max_attempts - 1:
                raise
            delay = base_delay * (attempt + 1)
            print(f"[voyage] rate-limited, retrying in {delay:.0f}s "
                  f"(attempt {attempt + 1}/{max_attempts})...", flush=True)
            time.sleep(delay)
    raise last_exc  # pragma: no cover — loop always returns or raises


_WORD = re.compile(r"[a-z0-9]+|[一-鿿]")


def _tokenize(text: str) -> list[str]:
    """Lightweight CN/EN tokenizer: English words + individual CJK chars."""
    return _WORD.findall(text.lower())


def _l2_normalize(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms
