"""
llm/openai.py

OpenAIProvider — future provider for AGENTX.

Not implemented yet. Set LLM_PROVIDER=openai in .env to activate.
Will raise NotImplementedError until implemented.

To implement:
  pip install openai
  Replace the NotImplementedError bodies below with real calls
  to openai.AsyncOpenAI().chat.completions.create(...)
"""

from __future__ import annotations

from llm.base import LLMProvider, LLMResponse


class OpenAIProvider(LLMProvider):
    """
    Calls the OpenAI Chat Completions API.

    Requires: OPENAI_API_KEY in .env
    Requires: pip install openai
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
            "OpenAIProvider is not implemented yet. "
            "Use LLM_PROVIDER=ollama (default) for now."
        )

    async def is_available(self) -> bool:
        if not self._api_key:
            return False
        return True