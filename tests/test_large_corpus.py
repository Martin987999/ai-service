# -*- coding: utf-8 -*-
"""Large-corpus scaling benchmark.

验证索引构建 + 检索在大语料下可扩展。用 mock embedding(api_key=None):
向量值是 mock 的,但 **FAISS 向量搜索 / BM25 打分 / RRF 融合的算法和延迟是真实的**
(索引结构与真实向量完全一致,只是数值不同),所以这是测"检索规模延迟"的有效手段,不烧额度。
真实 embedding 下,索引构建会多出 Voyage 调用耗时(可批量),检索延迟基本一致。

用法:
  python -m scripts.gen_large_corpus --n 800        # 先生成
  python tests/test_large_corpus.py                  # 再跑这个基准
"""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_settings
from src.ingestion.chunker import chunk_docs
from src.ingestion.loader import load_dir
from src.providers.embeddings import EmbeddingProvider
from src.retrieval.retriever import Retriever
from src.retrieval.store import IndexStore

CORPUS_DIR = "data/corpus_large"
QUERIES = [
    "入职满五年每年有多少天年假?", "对外 API 的访问令牌有效期多久?",
    "远程办公每周最多几天?", "敏感数据如何存储?",
    "How many concurrent requests does one instance support?",
    "What is the session timeout for privileged access?",
    "报销单笔超过多少需要额外审批?", "密钥多久轮换一次?",
]


def pct(vals, p):
    if not vals:
        return 0.0
    s = sorted(vals)
    k = max(0, min(len(s) - 1, int(round(p / 100 * (len(s) - 1)))))
    return round(s[k], 3)


def main() -> int:
    if not Path(CORPUS_DIR).exists():
        print(f"ERROR: {CORPUS_DIR} not found. Run: python -m scripts.gen_large_corpus --n 800")
        return 1

    s = load_settings()
    emb = EmbeddingProvider(api_key=None, allow_mock=True)  # mock vectors; real index algorithms

    t0 = time.perf_counter()
    docs = load_dir(CORPUS_DIR)
    cc = s.get("chunking")
    chunks = chunk_docs(docs, cc["chunk_size"], cc["chunk_overlap"], cc["min_chunk_chars"])
    t_chunk = time.perf_counter() - t0

    t0 = time.perf_counter()
    store = IndexStore(index_dir="data/_tmp_large_index", embedder=emb)
    store.build(chunks)
    t_build = time.perf_counter() - t0

    retriever = Retriever(store, emb, s.get("retrieval"))

    # measure retrieval latency for both modes at scale
    results = {}
    for mode, rerank in (("vector", False), ("hybrid", False), ("hybrid", True)):
        lat = []
        for _ in range(5):           # repeat to smooth noise
            for q in QUERIES:
                t = time.perf_counter()
                retriever.retrieve(q, mode=mode, rerank=rerank)
                lat.append((time.perf_counter() - t) * 1000)
        label = f"{mode}{'+rerank' if rerank else ''}"
        results[label] = {"p50_ms": pct(lat, 50), "p95_ms": pct(lat, 95), "n": len(lat)}

    print(f"docs={len(docs)} chunks={len(chunks)} index_size={store.size}")
    print(f"chunking={t_chunk:.2f}s  index_build(mock embed)={t_build:.2f}s")
    print("retrieval latency at scale:")
    for label, r in results.items():
        print(f"  {label:16s} p50={r['p50_ms']}ms p95={r['p95_ms']}ms (n={r['n']})")

    # sanity gate: retrieval should stay well under the 10s SLA budget even at scale
    worst_p95 = max(r["p95_ms"] for r in results.values())
    ok = worst_p95 < 2000 and store.size > 500
    print(f"\nworst retrieval p95 = {worst_p95}ms (retrieval-only; generation adds ~2-5s)")
    print("RESULT:", "PASS ✅" if ok else "FAIL ❌")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
