"""
llm/anthropic.py

AnthropicProvider — future provider for AGENTX.

Not implemented yet. Set LLM_PROVIDER=anthropic in .env to
activate. Will raise NotImplementedError until implemented in
Week 3 alongside the Planner.

To implement:
  pip install anthropic
  Replace the NotImplementedError bodies below with real calls
  to anthropic.AsyncAnthropic().messages.create(...)
"""

from __future__ import annotations

from llm.base import LLMProvider, LLMResponse, LLMUnavailableError


class AnthropicProvider(LLMProvider):
    """
    Calls the Anthropic Messages API.

    Requires: ANTHROPIC_API_KEY in .env
    Requires: pip install anthropic
    """

    def __init__(self, model: str, api_key: str, **_) -> None:
        super().__init__(model=model)
        self._api_key = api_key

    async def _complete(
        self,
        prompt: str,
        system: str,
        max_tokens: int,
        temperature: float,
    ) -> LLMResponse:
        raise NotImplementedError(
            "AnthropicProvider is not implemented yet. "
            "Use LLM_PROVIDER=ollama (default) for now. "
            "Anthropic support will be added in Week 3."
        )

    async def is_available(self) -> bool:
        if not self._api_key:
            return False
        # A non-empty key is our only check until the provider is implemented.
        return True