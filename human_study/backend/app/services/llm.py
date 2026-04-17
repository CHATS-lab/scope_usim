"""OpenAI-compatible chat client with tool calling."""
from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI

from ..config import settings


def build_client(base_url: str | None = None) -> AsyncOpenAI:
    return AsyncOpenAI(api_key=settings.openai_api_key, base_url=base_url or None)


async def chat_once(
    *,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    base_url: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> dict[str, Any]:
    """Single non-streaming chat completion. Returns the raw assistant message dict.

    The caller is responsible for executing any tool calls and looping back.
    """
    client = build_client(base_url)
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    # GPT-5.x and o-series use max_completion_tokens; older models use max_tokens.
    if model.startswith(("gpt-5", "o1", "o3", "o4")):
        kwargs["max_completion_tokens"] = max_tokens
    else:
        kwargs["temperature"] = temperature
        kwargs["max_tokens"] = max_tokens
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    resp = await client.chat.completions.create(**kwargs)
    choice = resp.choices[0]
    msg = choice.message
    return {
        "role": "assistant",
        "content": msg.content,
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in (msg.tool_calls or [])
        ]
        or None,
        "finish_reason": choice.finish_reason,
    }
