"""BM25 lexical index (CN/EN tokenization)."""
from __future__ import annotations

import re

_WORD = re.compile(r"[a-z0-9]+|[一-鿿]")


def _tok(text: str) -> list[str]:
    return _WORD.findall(text.lower())


class BM25Index:
    """Thin wrapper over rank_bm25 with a CN/EN tokenizer; numpy fallback if missing."""

    def __init__(self, corpus: list[str]):
        self._tokenized = [_tok(t) for t in corpus]
        self._bm25 = None
        try:
            from rank_bm25 import BM25Okapi

            self._bm25 = BM25Okapi(self._tokenized)
        except Exception:
            self._bm25 = None  # degrade to overlap scoring

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        q = _tok(query)
        if self._bm25 is not None:
            scores = self._bm25.get_scores(q)
        else:
            qset = set(q)
            scores = [len(qset & set(doc)) for doc in self._tokenized]
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(i, float(scores[i])) for i in ranked if scores[i] > 0]
