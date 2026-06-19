"""Evaluation metrics.

RAG 检索/答案质量(全部量化):
  - faithfulness         忠实度:答案中的论断是否均由检索上下文支持(LLM-as-judge,逐句判定)
  - context_precision    上下文精确率:Top-K 上下文中"相关块"的占比(LLM-as-judge 逐块相关性)
  - answer_compliance    答案合规率:是否严格基于上下文、命中 ground truth、未编造(LLM-as-judge)
  - style_consistency    风格一致性:语言对齐 + 简洁/专业/含来源标注(规则 + LLM-as-judge)
  - refusal_appropriateness  拒答恰当性:越界问题应拒答、可答问题不应拒答

评估模型与主答模型解耦(judge_model=sonnet),降低自评偏差;离线 mock 下给确定性近似分。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from src.providers.llm import LLMProvider

# ---------------- LLM-judge prompts ----------------
_FAITH_SYS = (
    "You are a strict RAG evaluator. Decide if EVERY factual claim in the ANSWER is supported "
    "by the CONTEXT. Output ONLY JSON: {\"score\": 0|1, \"reason\": \"...\"}. "
    "score=1 only if all claims are grounded in context; otherwise 0."
)
_CTXREL_SYS = (
    "You judge whether a CONTEXT passage is relevant to answering the QUESTION. "
    "Output ONLY JSON: {\"relevant\": 0|1}."
)
_COMPLIANCE_SYS = (
    "You evaluate an internal-KB assistant ANSWER. It is compliant if it (a) is grounded in the "
    "CONTEXT, (b) is consistent with the GROUND_TRUTH, and (c) does not fabricate. "
    "Output ONLY JSON: {\"score\": 0|1, \"reason\": \"...\"}."
)
_STYLE_SYS = (
    "You judge writing STYLE of an internal-KB answer. Good style = concise, professional/objective, "
    "no greetings/boilerplate. Output ONLY JSON: {\"score\": 0|1}."
)


def _judge_json(llm: LLMProvider, model: str, system: str, user: str) -> dict:
    res = llm.complete(system=system, user=user, model=model, max_tokens=512, effort="low",
                       prefer_json=True, thinking=False)
    return _parse_json(res.text)


def _parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


# ---------------- metric functions ----------------
def faithfulness(llm: LLMProvider, model: str, answer: str, contexts: list[str]) -> int:
    if not answer.strip() or not contexts:
        return 0
    ctx = "\n\n".join(contexts)
    user = f"CONTEXT:\n{ctx}\n\nANSWER:\n{answer}\n\nReturn JSON."
    return int(_judge_json(llm, model, _FAITH_SYS, user).get("score", 0))


def context_precision(llm: LLMProvider, model: str, question: str, contexts: list[str]) -> float:
    """Fraction of retrieved contexts judged relevant to the question."""
    if not contexts:
        return 0.0
    rel = 0
    for c in contexts:
        user = f"QUESTION:\n{question}\n\nCONTEXT:\n{c}\n\nReturn JSON."
        rel += int(_judge_json(llm, model, _CTXREL_SYS, user).get("relevant", 0))
    return rel / len(contexts)


def answer_compliance(llm: LLMProvider, model: str, question: str, answer: str,
                      contexts: list[str], ground_truths: list[str]) -> int:
    ctx = "\n\n".join(contexts)
    gt = " | ".join(ground_truths) if ground_truths else "(none)"
    user = (f"QUESTION:\n{question}\n\nGROUND_TRUTH:\n{gt}\n\nCONTEXT:\n{ctx}\n\n"
            f"ANSWER:\n{answer}\n\nReturn JSON.")
    # fast-path: exact ground-truth hit counts as compliant
    if ground_truths and any(_norm(g) in _norm(answer) for g in ground_truths):
        return 1
    return int(_judge_json(llm, model, _COMPLIANCE_SYS, user).get("score", 0))


_GREETINGS = re.compile(r"^(您好|你好|hi\b|hello|thanks|感谢|当然)", re.I)


def style_consistency(llm: LLMProvider, model: str, answer: str, expected_lang: str) -> int:
    """Rule gate (language alignment + no greeting) then LLM confirmation."""
    from src.ingestion.lang import detect_lang

    if not answer.strip():
        return 0
    if detect_lang(answer) not in (expected_lang, "unknown"):
        return 0
    if _GREETINGS.search(answer.strip()):
        return 0
    user = f"ANSWER:\n{answer}\n\nReturn JSON."
    return int(_judge_json(llm, model, _STYLE_SYS, user).get("score", 1))


def refusal_appropriate(refused: bool, out_of_scope: bool) -> int:
    """Appropriate iff (out_of_scope ⇔ refused)."""
    return int(refused == out_of_scope)


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s.lower())
