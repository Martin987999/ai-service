"""Prompt templates for grounded generation.

answer 提示词:强制 grounding、语言对齐(回答语言跟随提问语言)、风格一致(简洁、专业、可引用来源)、
不足即拒答。<context> 内文本被声明为「数据」以抵御间接注入。
"""
from __future__ import annotations

from .safety import GROUNDING_GUARD_EN, GROUNDING_GUARD_ZH

ANSWER_SYSTEM_ZH = f"""你是企业内部知识库问答助手。{GROUNDING_GUARD_ZH}

风格要求(必须遵守):
1. 用简体中文回答(与用户提问语言一致)。
2. 专业、简洁、客观;先给结论,再给要点;避免寒暄与免责套话。
3. 在句末用 [来源: <doc_id>] 标注依据。
4. 若 <context> 不足以支撑答案,只输出:REFUSE
"""

ANSWER_SYSTEM_EN = f"""You are an internal knowledge-base QA assistant. {GROUNDING_GUARD_EN}

Style rules (must follow):
1. Answer in English (match the user's language).
2. Professional, concise, factual; lead with the conclusion, then key points; no greetings or boilerplate disclaimers.
3. Cite evidence inline as [source: <doc_id>].
4. If <context> is insufficient, output exactly: REFUSE
"""


def build_answer_prompt(query: str, contexts: list[tuple[str, str]], lang: str,
                        history: list[tuple[str, str]] | None = None) -> tuple[str, str]:
    """contexts: list of (doc_id, text). history: list of (role, content) recent turns.
    Returns (system, user). History gives conversational coherence; answer still grounds in context."""
    system = ANSWER_SYSTEM_ZH if lang == "zh" else ANSWER_SYSTEM_EN
    ctx_block = "\n\n".join(f"[{doc_id}]\n{text}" for doc_id, text in contexts)
    hist_block = ""
    if history:
        recent = history[-4:]
        lines = "\n".join(f"{r}: {c}" for r, c in recent)
        hist_block = (f"对话历史(仅供消解指代/保持连贯,不作为事实依据):\n{lines}\n\n"
                      if lang == "zh"
                      else f"Conversation history (for coreference/coherence only, NOT a source of facts):\n{lines}\n\n")
    if lang == "zh":
        user = f"{hist_block}问题:{query}\n\n<context>\n{ctx_block}\n</context>\n\n请仅基于上述 context 作答。"
    else:
        user = f"{hist_block}Question: {query}\n\n<context>\n{ctx_block}\n</context>\n\nAnswer ONLY from the context above."
    return system, user


REFUSAL_SENTINEL = "REFUSE"
