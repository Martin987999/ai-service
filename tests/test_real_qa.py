# -*- coding: utf-8 -*-
"""One real end-to-end QA (real Voyage retrieval + real Claude generation).

单次真实问答冒烟:验证整条链路在真实模型下工作。Voyage 免费层 3 RPM,故只问一题、关语义缓存省一次 embed。
运行前需 data/index 为真实 Voyage 索引(python -m scripts.ingest,且配置了 VOYAGE_API_KEY)。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.app import build_pipeline


def main() -> int:
    p = build_pipeline()
    print(f"llm_mock={p.llm.is_mock} embed_mock={p.embedder.is_mock}")
    if p.llm.is_mock or p.embedder.is_mock:
        print("NOTE: a provider is mock — set ANTHROPIC_API_KEY / VOYAGE_API_KEY for a real run.")
    # reduce Voyage calls to 2 (query embed + rerank) by disabling semantic cache embed
    p.cache.semantic_enabled = False

    q = "入职满五年每年有多少天年假?未休的年假最多能结转几天?"
    print(f"\nQ: {q}")
    r = p.answer(q)
    print(f"refused={r.refused} reason={r.refusal_reason} conf={round(r.confidence,3)} lang={r.lang}")
    print(f"ANSWER:\n{r.answer}")
    print(f"sources: {[s['doc_id'] for s in r.sources]}")
    print(f"usage: {r.usage}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
