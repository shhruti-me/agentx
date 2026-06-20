"""
llm/groq.py

GroqProvider — free, fast LLM inference via Groq's API.

Groq uses the OpenAI-compatible chat completions format.
We call it directly with httpx — no SDK needed.

Free tier: 14,400 requests/day, 500,000 tokens/minute.
Recommended model: llama-3.3-70b-versatile

Groq API docs: https://console.groq.com/docs/openai
"""

from __future__ import annotations

import json
import logging

import httpx

from llm.base import (
    LLMProvider,
    LLMProviderError,
    LLMResponse,
    LLMUnavailableError,
)

logger = logging.getLogger(__name__)

_GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class GroqProvider(LLMProvider):
    """
    Calls Groq's OpenAI-compatible chat completions API.

    Requires: GROQ_API_KEY in .env
    Recommended model: llama-3.3-70b-versatile
    """

    def __init__(self, model: str, api_key: str, timeout_seconds: int = 60) -> None:
        super().__init__(model=model)
        self._api_key = api_key
        self._timeout = timeout_seconds

        if not self._api_key:
            raise LLMUnavailableError(
                "GROQ_API_KEY is not set. "
                "Get a free key at https://console.groq.com then add it to .env"
            )

    async def _complete(
        self,
        prompt: str,
        system: str,
        max_tokens: int,
        temperature: float,
    ) -> LLMResponse:
        """
        POST /openai/v1/chat/completions

        Groq uses the standard OpenAI messages format:
          [{"role": "system", ...}, {"role": "user", ...}]
        """
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        logger.debug(
            "groq_request",
            extra={
                "model": self.model,
                "prompt_len": len(prompt),
                "system_len": len(system),
            },
        )

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    f"{_GROQ_BASE_URL}/chat/completions",
                    json=payload,
                    headers=headers,
                )
        except httpx.ConnectError as exc:
            raise LLMUnavailableError(
                "Cannot reach Groq API. Check your internet connection."
            ) from exc
        except httpx.TimeoutException as exc:
            raise LLMProviderError(
                f"Groq request timed out after {self._timeout}s."
            ) from exc

        if resp.status_code == 401:
            raise LLMUnavailableError(
                "Groq API key is invalid or expired. "
                "Check GROQ_API_KEY in your .env file."
            )
        if resp.status_code == 429:
            raise LLMProviderError(
                "Groq rate limit hit. Free tier: 14,400 requests/day. "
                "Wait a moment and try again."
            )
        if resp.status_code != 200:
            raise LLMProviderError(
                f"Groq returned HTTP {resp.status_code}: {resp.text[:400]}"
            )

        try:
            data = resp.json()
        except json.JSONDecodeError as exc:
            raise LLMProviderError(
                f"Groq response was not valid JSON: {resp.text[:400]}"
            ) from exc

        # OpenAI response shape:
        # {
        #   "choices": [{"message": {"content": "..."}}],
        #   "usage": {"prompt_tokens": 42, "completion_tokens": 138}
        # }
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise LLMProviderError(
                f"Unexpected Groq response shape: {data}"
            ) from exc

        if not content:
            raise LLMProviderError("Groq returned empty content.")

        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        logger.debug(
            "groq_response",
            extra={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "content_preview": content[:120],
            },
        )

        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self.model,
            provider="groq",
        )

    async def is_available(self) -> bool:
        """
        GET /openai/v1/models — returns 200 if key is valid and API is reachable.
        Returns False on any error. Never raises.
        """
        if not self._api_key:
            return False
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(
                    f"{_GROQ_BASE_URL}/models",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                )
                return resp.status_code == 200
        except Exception:
            return False