"""Answer cache: exact (hash) + semantic (embedding cosine).

精确缓存:对 (规范化 query + 检索模式 + rerank 开关) 哈希命中。
语义缓存:query embedding 与历史 query 余弦 >= 阈值则复用答案。
命中率写入运维报告(cache_hit_rate)。
"""
from __future__ import annotations

import hashlib
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def normalize_query(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower())


def cache_key(query: str, mode: str, rerank: bool) -> str:
    raw = f"{normalize_query(query)}|{mode}|{int(rerank)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass
class CacheEntry:
    value: dict[str, Any]
    ts: float
    vec: np.ndarray | None = None


class AnswerCache:
    def __init__(self, enabled: bool, ttl_s: int, max_entries: int,
                 semantic_enabled: bool, semantic_threshold: float):
        self.enabled = enabled
        self.ttl_s = ttl_s
        self.max_entries = max_entries
        self.semantic_enabled = semantic_enabled
        self.semantic_threshold = semantic_threshold
        self._store: "OrderedDict[str, CacheEntry]" = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str, query_vec: np.ndarray | None = None) -> tuple[dict | None, str]:
        """Returns (value, hit_type) where hit_type ∈ exact|semantic|miss."""
        if not self.enabled:
            return None, "miss"
        self._evict_expired()
        # exact
        entry = self._store.get(key)
        if entry is not None:
            self._store.move_to_end(key)
            self.hits += 1
            return entry.value, "exact"
        # semantic
        if self.semantic_enabled and query_vec is not None:
            best, best_sim = None, -1.0
            for e in self._store.values():
                if e.vec is None:
                    continue
                sim = float(np.dot(e.vec, query_vec))
                if sim > best_sim:
                    best, best_sim = e, sim
            if best is not None and best_sim >= self.semantic_threshold:
                self.hits += 1
                return {**best.value, "_semantic_sim": round(best_sim, 4)}, "semantic"
        self.misses += 1
        return None, "miss"

    def put(self, key: str, value: dict, query_vec: np.ndarray | None = None) -> None:
        if not self.enabled:
            return
        self._store[key] = CacheEntry(value=value, ts=time.time(), vec=query_vec)
        self._store.move_to_end(key)
        while len(self._store) > self.max_entries:
            self._store.popitem(last=False)

    def _evict_expired(self) -> None:
        now = time.time()
        stale = [k for k, e in self._store.items() if now - e.ts > self.ttl_s]
        for k in stale:
            self._store.pop(k, None)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def stats(self) -> dict:
        return {"hits": self.hits, "misses": self.misses, "hit_rate": round(self.hit_rate, 4),
                "size": len(self._store)}
