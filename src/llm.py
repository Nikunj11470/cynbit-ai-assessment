"""Thin wrapper around the Claude API (optional - the app runs fully offline without it)."""
from __future__ import annotations

import json
import os

from .config import CLAUDE_MODEL


class LLMError(RuntimeError):
    pass


def available() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY"))


def complete(system: str, user: str, schema: dict | None = None,
             max_tokens: int = 8000, effort: str = "low") -> dict | str:
    """Call Claude. Returns parsed JSON when `schema` is given, else plain text."""
    import anthropic

    output_config: dict = {"effort": effort}
    if schema:
        output_config["format"] = {"type": "json_schema", "schema": schema}

    client = anthropic.Anthropic()
    try:
        resp = client.beta.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config=output_config,
            # Server-side fallback: if a request is declined, the API retries on a suitable model
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
    except anthropic.RateLimitError as e:
        raise LLMError(f"Rate limited: {e}") from e
    except anthropic.APIStatusError as e:
        raise LLMError(f"API error {e.status_code}: {e.message}") from e
    except anthropic.APIConnectionError as e:
        raise LLMError(f"Connection error: {e}") from e

    if resp.stop_reason == "refusal":
        raise LLMError("Model declined the request")
    if resp.stop_reason == "max_tokens":
        raise LLMError("Response truncated (max_tokens)")

    text = "".join(block.text for block in resp.content if block.type == "text").strip()
    if not schema:
        return text
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise LLMError(f"Invalid JSON from model: {e}") from e
