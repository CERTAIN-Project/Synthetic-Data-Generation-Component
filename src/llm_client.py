"""Lightweight async wrapper around the OpenAI Responses API."""

from __future__ import annotations

import logging
import os
from typing import Any

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class OpenAIClient:
    """Async client for OpenAI text generation and tool use."""

    def __init__(self, api_key: str, max_retries: int = 2, base_url: str | None = None) -> None:
        """Initialise the OpenAI client.

        Args:
            api_key: OpenAI API key for authentication.
            max_retries: Maximum retry attempts for failed requests.
            base_url: Optional custom API base URL.
        """
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Add it to your environment or .env file."
            )

        normalized_base_url = (base_url or "").strip() or None
        if normalized_base_url is None:
            # The OpenAI SDK also reads OPENAI_BASE_URL from the environment.
            # Remove blank values so an empty .env entry does not become an invalid URL.
            os.environ.pop("OPENAI_BASE_URL", None)

        client_kwargs: dict[str, Any] = {"api_key": api_key, "max_retries": max_retries}
        if normalized_base_url:
            client_kwargs["base_url"] = normalized_base_url

        self._async_client = AsyncOpenAI(**client_kwargs)

    async def achat(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.0,
        model_name: str = "gpt-4.1-mini",
        max_tokens: int = 4096,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Send an async request to OpenAI and return the response text."""
        try:
            request_kwargs: dict[str, Any] = {
                "model": model_name,
                "instructions": system_prompt,
                "input": prompt,
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }

            if tools:
                request_kwargs["tools"] = tools
                request_kwargs["tool_choice"] = "required"

            response = await self._async_client.responses.create(**request_kwargs)

            text = getattr(response, "output_text", "") or ""
            if not text and getattr(response, "output", None):
                text_blocks: list[str] = []
                for item in response.output:
                    for content in getattr(item, "content", []) or []:
                        chunk = getattr(content, "text", "")
                        if chunk:
                            text_blocks.append(chunk)
                text = "".join(text_blocks)

            return {"text": text}

        except Exception as exc:
            logger.error(
                f"Async chat completion failed with model '{model_name}': {exc}"
            )
            raise


# Backwards compatibility for older notebooks/imports.
AnthropicClient = OpenAIClient
