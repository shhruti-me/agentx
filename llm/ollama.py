"""
llm/ollama.py

OllamaProvider — default LLM backend for AGENTX.

Calls Ollama's REST API directly with httpx.
No Ollama SDK. No extra dependencies beyond httpx.

Ollama API used:
  POST /api/chat       — text generation
  GET  /api/tags       — health check (lists loaded models)

Ollama docs: https://github.com/ollama/ollama/blob/main/docs/api.md
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


class OllamaProvider(LLMProvider):
    """
    Calls a locally running Ollama instance.

    Default base_url: http://localhost:11434
    Default model:    qwen3:latest

    Qwen3 note: Qwen3 models support a /think block before their
    response. We strip it — callers only want the final answer,
    not the chain-of-thought. See _strip_think() below.
    """

    def __init__(self, model: str, base_url: str, timeout_seconds: int = 120) -> None:
        super().__init__(model=model, base_url=base_url.rstrip("/"))
        self._timeout = timeout_seconds

    # ── Core completion ───────────────────────────────────────────────

    async def _complete(
        self,
        prompt: str,
        system: str,
        max_tokens: int,
        temperature: float,
    ) -> LLMResponse:
        """
        POST /api/chat with stream=False.

        Ollama's /api/chat uses the OpenAI-style messages array.
        We build: [system (optional), user].

        Returns LLMResponse. Raises on any failure so the caller
        can decide whether to retry or abort.
        """
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        logger.debug(
            "ollama_request",
            extra={
                "model": self.model,
                "system_len": len(system),
                "prompt_len": len(prompt),
            },
        )

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                )
        except httpx.ConnectError as exc:
            raise LLMUnavailableError(
                f"Cannot reach Ollama at {self.base_url}. "
                "Is Ollama running? Try: ollama serve"
            ) from exc
        except httpx.TimeoutException as exc:
            raise LLMProviderError(
                f"Ollama request timed out after {self._timeout}s. "
                "Model may still be loading — try again or increase LLM_TIMEOUT_SECONDS."
            ) from exc

        if resp.status_code != 200:
            raise LLMProviderError(
                f"Ollama returned HTTP {resp.status_code}: {resp.text[:400]}"
            )

        try:
            data = resp.json()
        except json.JSONDecodeError as exc:
            raise LLMProviderError(
                f"Ollama response was not valid JSON: {resp.text[:400]}"
            ) from exc

        # Ollama /api/chat response shape:
        # {
        #   "message": {"role": "assistant", "content": "..."},
        #   "prompt_eval_count": 42,    # input tokens (may be absent on cache hit)
        #   "eval_count": 138,          # output tokens
        #   "done": true
        # }
        content_raw: str = data.get("message", {}).get("content", "")
        if not content_raw:
            raise LLMProviderError(
                f"Ollama returned an empty response. Full body: {data}"
            )

        content = _strip_think(content_raw)

        input_tokens: int = data.get("prompt_eval_count", 0)
        output_tokens: int = data.get("eval_count", 0)

        logger.debug(
            "ollama_response",
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
            provider="ollama",
        )

    # ── Health check ──────────────────────────────────────────────────

    async def is_available(self) -> bool:
        """
        GET /api/tags — returns 200 if Ollama is running.
        Returns False on any error. Never raises.
        """
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False


# ── Helpers ───────────────────────────────────────────────────────────────────


def _strip_think(content: str) -> str:
    """
    Remove Qwen3's <think>...</think> block from the response.

    Qwen3 emits its chain-of-thought inside <think> tags before
    the actual answer. The Planner and Verifier only want the
    final answer — the think block would pollute JSON parsing
    and verification logic.

    If no think block is present (other models, or Qwen3 with
    thinking disabled), the content is returned unchanged.

    Examples
    --------
    Input:  "<think>\\nLet me reason...\\n</think>\\n\\nThe answer is 42."
    Output: "The answer is 42."

    Input:  "The answer is 42."
    Output: "The answer is 42."
    """
    if "<think>" not in content:
        return content.strip()

    # Find the closing tag — everything after it is the real answer
    end_tag = "</think>"
    idx = content.find(end_tag)
    if idx == -1:
        # Malformed: opening tag but no closing tag.
        # Strip from <think> to end-of-string to avoid leaking
        # partial reasoning into the response.
        start = content.find("<think>")
        return content[:start].strip()

    return content[idx + len(end_tag):].strip()