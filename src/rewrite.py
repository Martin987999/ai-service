"""History-aware query rewriting (condense question).

把依赖上下文的跟进问题(如「那十年呢?」)改写为可独立检索的问题
(「入职满十年每年有多少天年假?」),用于历史感知检索。
使用 cheap_model(haiku)控成本;无 key 时走启发式 mock 回退。
"""
from __future__ import annotations

import re

from .providers.llm import LLMProvider
from .session import Turn

_CONDENSE_SYS = (
    "Given the conversation history and a follow-up question, rewrite the follow-up "
    "into a fully self-contained, standalone question in the SAME language as the follow-up. "
    "Resolve pronouns and ellipses using the history. Output ONLY the rewritten question, nothing else. "
    "If it is already standalone, return it unchanged."
)

# follow-up signals (pronouns / elliptical openers) — used by mock heuristic
_FOLLOWUP = re.compile(
    r"(那|这|它|他|她|他们|它们|呢[??]?$|前者|后者|上面|刚才|"
    r"\b(it|that|this|they|them|those|these|he|she|the former|the latter)\b|"
    r"^\s*(and|what about|how about|then|so)\b)",
    re.I,
)


def needs_rewrite(history: list[Turn], query: str) -> bool:
    if not history:
        return False
    # short queries or those with follow-up markers likely depend on context
    return bool(_FOLLOWUP.search(query)) or len(query.strip()) <= 12


def condense_query(llm: LLMProvider, model: str, history: list[Turn], query: str) -> str:
    if not needs_rewrite(history, query):
        return query
    if llm.is_mock:
        return _mock_condense(history, query)
    hist_txt = "\n".join(f"{t.role}: {t.content}" for t in history[-6:])
    user = f"History:\n{hist_txt}\n\nFollow-up: {query}\n\nStandalone question:"
    res = llm.complete(system=_CONDENSE_SYS, user=user, model=model, max_tokens=128,
                       effort="low", thinking=False)
    out = res.text.strip().splitlines()[0].strip() if res.text.strip() else query
    return out or query


def _mock_condense(history: list[Turn], query: str) -> str:
    """Heuristic offline rewrite: splice the last user question's subject in."""
    last_user = next((t.content for t in reversed(history) if t.role == "user"), "")
    if not last_user:
        return query
    # crude but effective for retrieval: combine prior question + follow-up
    return f"{last_user} {query}".strip()
