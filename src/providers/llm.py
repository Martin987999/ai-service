"""Claude LLM provider (generation + LLM-as-judge).

封装 Claude 调用。包含 token 用量统计与成本估算。
无 ANTHROPIC_API_KEY 且允许 mock 时 → 确定性 mock,保证流程/评估可离线跑通。
模型与定价依据 Claude API 技能(2026 价表):
  opus-4-8   $5 / $25   per 1M tok
  sonnet-4-6 $3 / $15   per 1M tok
  haiku-4-5  $1 / $5    per 1M tok
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# input/output USD per 1M tokens
PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-4-8": (5.0, 25.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}

# Models that accept output_config.effort + adaptive thinking.
# effort errors on Haiku 4.5 / Sonnet 4.5; gate both params on this set.
_EFFORT_CAPABLE = {
    "claude-opus-4-8", "claude-opus-4-7", "claude-opus-4-6", "claude-opus-4-5",
    "claude-sonnet-4-6", "claude-fable-5", "claude-mythos-5",
}


@dataclass
class LLMResult:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str | None = None
    is_mock: bool = False

    @property
    def cost_usd(self) -> float:
        pin, pout = PRICING.get(self.model, (0.0, 0.0))
        return self.input_tokens / 1e6 * pin + self.output_tokens / 1e6 * pout


class LLMProvider:
    def __init__(self, api_key: str | None, allow_mock: bool = True):
        self._api_key = api_key
        self._allow_mock = allow_mock
        self._client = None
        if api_key:
            try:
                import anthropic

                self._client = anthropic.Anthropic(api_key=api_key)
            except Exception:
                self._client = None
        if self._client is None and not allow_mock:
            raise RuntimeError("ANTHROPIC_API_KEY missing and mock fallback disabled.")

    @property
    def is_mock(self) -> bool:
        return self._client is None

    def complete(
        self,
        system: str,
        user: str,
        model: str,
        max_tokens: int = 1024,
        effort: str = "medium",
        prefer_json: bool = False,
        thinking: bool = True,
    ) -> LLMResult:
        if self._client is None:
            return self._mock(system, user, model, prefer_json)

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        # effort + adaptive thinking are only valid on the effort-capable family
        # (they 400 on Haiku 4.5 / Sonnet 4.5). Gate both on model capability.
        if model in _EFFORT_CAPABLE:
            kwargs["output_config"] = {"effort": effort}
            # adaptive thinking recommended; disable for tight-budget JSON judge calls
            kwargs["thinking"] = {"type": "adaptive"} if thinking else {"type": "disabled"}
        resp = self._client.messages.create(**kwargs)
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        return LLMResult(
            text=text.strip(),
            model=model,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            stop_reason=resp.stop_reason,
            is_mock=False,
        )

    # ---------------------------------------------------------------
    def _mock(self, system: str, user: str, model: str, prefer_json: bool) -> LLMResult:
        """Deterministic offline behaviour.

        - judge prompts (contain 'score 0' / 'JSON') → return a plausible JSON verdict
        - answer prompts → echo a grounded answer built from the <context> block
        """
        in_tok = _approx_tokens(system) + _approx_tokens(user)
        if prefer_json or "JSON" in user or "json" in user:
            text = '{"score": 1, "verdict": "supported", "reason": "mock"}'
        else:
            ctx = _extract_context(user)
            if not ctx.strip():
                text = "REFUSE: insufficient context."
            else:
                # naive: return the first 1-2 sentences of the context as the answer
                sents = re.split(r"(?<=[。.!?\n])", ctx)
                text = "".join(s for s in sents[:2]).strip() or ctx[:200]
        return LLMResult(
            text=text,
            model=model,
            input_tokens=in_tok,
            output_tokens=_approx_tokens(text),
            stop_reason="end_turn",
            is_mock=True,
        )


def _approx_tokens(s: str) -> int:
    # rough offline estimate only (real path uses Anthropic usage). ~4 chars/token.
    return max(1, len(s) // 4)


def _extract_context(user: str) -> str:
    m = re.search(r"<context>(.*?)</context>", user, re.S)
    return m.group(1) if m else ""
