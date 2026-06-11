"""
llm/factory.py

get_llm_client() — the single entry point for obtaining an LLM provider.

Every component that needs an LLM calls this function.
No component instantiates a provider directly.

Switching providers = changing LLM_PROVIDER in .env.
Zero code changes required anywhere else.

Usage
-----
    from llm.factory import get_llm_client

    client = get_llm_client()
    response = await client.complete(
        prompt="What is the capital of France?",
        system="You are a helpful assistant.",
    )
    print(response.content)   # "Paris"
    print(response.total_tokens)
"""

from __future__ import annotations

import logging

from config.settings import settings
from llm.base import LLMProvider

logger = logging.getLogger(__name__)

# Module-level cache — provider is instantiated once per process.
# get_llm_client() is called frequently (once per LLM call site);
# we don't want to re-read settings or re-instantiate on every call.
_provider_instance: LLMProvider | None = None


def get_llm_client(force_refresh: bool = False) -> LLMProvider:
    """
    Return the configured LLM provider, instantiated from settings.

    Parameters
    ----------
    force_refresh : If True, discard the cached instance and
                    re-instantiate. Useful in tests that change
                    settings between calls.

    Returns
    -------
    LLMProvider concrete instance ready to call .complete() on.

    Raises
    ------
    ValueError : LLM_PROVIDER value is not one of the supported options.
    """
    global _provider_instance

    if _provider_instance is not None and not force_refresh:
        return _provider_instance

    provider_name = settings.llm_provider.lower().strip()

    logger.info(
        "llm_provider_init",
        extra={
            "provider": provider_name,
            "model": settings.llm_model,
        },
    )

    if provider_name == "ollama":
        from llm.ollama import OllamaProvider

        _provider_instance = OllamaProvider(
            model=settings.llm_model,
            base_url=settings.llm_base_url,
            timeout_seconds=settings.llm_timeout_seconds,
        )

    elif provider_name == "anthropic":
        from llm.anthropic import AnthropicProvider

        _provider_instance = AnthropicProvider(
            model=settings.llm_model,
            api_key=settings.anthropic_api_key,
        )

    elif provider_name == "openai":
        from llm.openai import OpenAIProvider

        _provider_instance = OpenAIProvider(
            model=settings.llm_model,
            api_key=settings.openai_api_key,
        )

    else:
        supported = ("ollama", "anthropic", "openai")
        raise ValueError(
            f"Unknown LLM_PROVIDER: {provider_name!r}. "
            f"Supported values: {supported}. "
            "Check your .env file."
        )

    logger.info(
        "llm_provider_ready",
        extra={"provider": provider_name, "model": settings.llm_model},
    )

    return _provider_instance