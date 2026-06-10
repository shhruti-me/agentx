"""
llm/base.py

Provider-agnostic LLM interface for AGENTX.

Every component that calls an LLM (Planner, LLM Verifier,
Self-Correction Engine) depends only on this module.
No component outside llm/ ever imports httpx, anthropic, or openai.

Two things live here:

  LLMResponse   — the normalised return value from any provider
  LLMProvider   — the ABC every concrete provider must implement

Adding a new provider means:
  1. Create llm/<name>.py implementing LLMProvider
  2. Add one case to llm/factory.py
  3. Zero changes anywhere else in the codebase
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


# ── Response ──────────────────────────────────────────────────────────────────


@dataclass
class LLMResponse:
    """
    Normalised response returned by every provider.

    Callers only ever see this type — never a provider-specific
    response object. This is what makes the abstraction work.

    Fields
    ------
    content       : The text the model generated. Always a non-empty
                    string on success. Raise on failure — don't return
                    an empty string silently.

    input_tokens  : Tokens consumed by the prompt. Used for cost
                    tracking and stored in the tasks table.

    output_tokens : Tokens in the model's response.

    model         : The exact model string used (e.g. "qwen3:latest").
                    Logged with every call so benchmark reports can
                    attribute results to a specific model version.

    provider      : Which backend produced this response
                    ("ollama" | "anthropic" | "openai"). Useful in
                    logs when debugging provider-specific behaviour.

    latency_ms    : Wall-clock milliseconds from request to response.
                    Populated automatically by the base class's
                    complete() wrapper — providers don't set this.
    """

    content: str
    input_tokens: int
    output_tokens: int
    model: str
    provider: str
    latency_ms: int = 0

    @property
    def total_tokens(self) -> int:
        """Convenience sum used when writing to the tasks table."""
        return self.input_tokens + self.output_tokens


# ── Provider ABC ──────────────────────────────────────────────────────────────


class LLMProvider(ABC):
    """
    Abstract base class for all LLM providers.

    Concrete implementations: OllamaProvider, AnthropicProvider,
    OpenAIProvider. The factory (llm/factory.py) returns one of
    these based on settings.llm_provider.

    Subclasses must implement:
      _complete()     — the actual API call
      is_available()  — lightweight health check

    Subclasses must NOT override complete() — the timing wrapper
    lives there and must run for every call.
    """

    def __init__(self, model: str, base_url: str = "") -> None:
        self.model = model
        self.base_url = base_url

    # ── Public interface (do not override) ────────────────────────────

    async def complete(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.2,
    ) -> LLMResponse:
        """
        Call the LLM and return a normalised LLMResponse.

        This method wraps _complete() with timing. Every provider
        gets latency tracking for free without duplicating the code.

        Parameters
        ----------
        prompt      : The user-turn message.
        system      : Optional system prompt. Passed as a system
                      message where the provider supports it; prepended
                      to the prompt where it doesn't.
        max_tokens  : Hard cap on response length.
        temperature : Sampling temperature. Keep low (≤ 0.3) for
                      planning tasks where determinism matters.

        Raises
        ------
        LLMProviderError    : API call failed, model not found, etc.
        LLMUnavailableError : Provider is not reachable at all
                              (Ollama not running, no API key, etc.)
        """
        start = time.monotonic()
        response = await self._complete(
            prompt=prompt,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        response.latency_ms = int((time.monotonic() - start) * 1000)
        return response

    # ── Abstract methods (must implement) ─────────────────────────────

    @abstractmethod
    async def _complete(
        self,
        prompt: str,
        system: str,
        max_tokens: int,
        temperature: float,
    ) -> LLMResponse:
        """
        Provider-specific API call. Returns LLMResponse without
        latency_ms set — the public complete() method fills that in.
        """

    @abstractmethod
    async def is_available(self) -> bool:
        """
        Lightweight check that the provider is reachable.

        Must return False cleanly if the provider is down —
        never raise. Used by the /health endpoint.

        For Ollama: GET /api/tags and check for HTTP 200.
        For cloud providers: validate API key is non-empty;
          optionally ping a cheap endpoint.
        """

    # ── Repr ──────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model!r})"


# ── Exceptions ────────────────────────────────────────────────────────────────


class LLMError(Exception):
    """Base class for all LLM-related errors."""


class LLMProviderError(LLMError):
    """
    The provider was reachable but returned an error.

    Examples: model not found, context too long, rate limited,
    malformed response JSON.
    """


class LLMUnavailableError(LLMError):
    """
    The provider could not be reached at all.

    Examples: Ollama not running, no network, invalid base URL,
    missing API key for cloud provider.
    """