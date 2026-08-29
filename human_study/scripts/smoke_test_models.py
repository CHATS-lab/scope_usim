"""Smoke-test all 4 model endpoints for the human study.

Verifies each endpoint can:
  1. Return a chat completion to a plain "hello"
  2. Honor tool-calling on a minimal tau2-style tool schema

Reads endpoint config from backend/.env so the .env is the single source of
truth for what's deployed.

Usage:
    python scripts/smoke_test_models.py
    python scripts/smoke_test_models.py --models base,rl_single
    python scripts/smoke_test_models.py --skip-tools     # skip tool-calling tests
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

from dotenv import dotenv_values
from openai import AsyncOpenAI, APIConnectionError, APIStatusError


ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / "backend" / ".env"


def load_endpoints() -> dict[str, dict[str, str]]:
    env = dotenv_values(ENV_PATH)
    fallback_key = env.get("OPENAI_API_KEY", "")

    def cfg(name: str) -> dict[str, str]:
        return {
            "model": env.get(f"MODEL_{name}", ""),
            "base_url": env.get(f"OPENAI_BASE_URL_{name}") or None,
            "api_key": env.get(f"OPENAI_API_KEY_{name}") or fallback_key,
        }

    return {
        "base": cfg("BASE"),
        "rl_single": cfg("RL_SINGLE"),
        "cotraining": cfg("COTRAINING"),
    }


# Minimal tau2-style tool to verify the OpenAI tools API is wired correctly
# end-to-end on each endpoint (including the qwen25 tool-call parser on SGLang).
TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_user_orders",
            "description": "Look up a user's recent orders by user_id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "The user identifier, e.g. user_42",
                    }
                },
                "required": ["user_id"],
            },
        },
    }
]


def is_reasoning_model(model: str) -> bool:
    return model.startswith(("gpt-5", "o1", "o3", "o4"))


def kwargs_for(model: str, max_tokens: int, has_tools: bool = False) -> dict:
    # gpt-5.x + reasoning_effort + tools is unsupported on /v1/chat/completions,
    # so we drop reasoning_effort when tools are present (matches chat.py).
    if is_reasoning_model(model):
        kw: dict = {"max_completion_tokens": max_tokens}
        if not has_tools:
            kw["reasoning_effort"] = "medium"
        return kw
    return {"max_tokens": max_tokens, "temperature": 0.7}


async def test_simple(client: AsyncOpenAI, model: str, base_url: str | None) -> tuple[bool, str, float]:
    t0 = time.monotonic()
    try:
        extra: dict = {}
        # Qwen3 base models default to chain-of-thought; tell SGLang to disable
        # it so `content` is populated (otherwise it lands in reasoning_content).
        if base_url:
            extra["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Reply in 1 short sentence."},
                {"role": "user", "content": "Say hi and tell me what 2+2 is."},
            ],
            **kwargs_for(model, 128),
            **extra,
        )
        text = (resp.choices[0].message.content or "").strip()
        return True, text, time.monotonic() - t0
    except (APIConnectionError, APIStatusError, Exception) as e:
        return False, f"{type(e).__name__}: {e}", time.monotonic() - t0


async def test_tool_call(client: AsyncOpenAI, model: str, base_url: str | None) -> tuple[bool, str, float]:
    t0 = time.monotonic()
    try:
        extra: dict = {}
        if base_url:
            extra["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a customer-service agent. Use tools when needed."},
                {"role": "user", "content": "I'm user_42. Can you look up my orders?"},
            ],
            tools=TOOL_SCHEMA,
            tool_choice="auto",
            **kwargs_for(model, 256, has_tools=True),
            **extra,
        )
        msg = resp.choices[0].message
        tool_calls = msg.tool_calls or []
        if tool_calls:
            tc = tool_calls[0]
            return True, f"tool_call: {tc.function.name}({tc.function.arguments})", time.monotonic() - t0
        return False, f"no tool call. content={msg.content!r}", time.monotonic() - t0
    except (APIConnectionError, APIStatusError, Exception) as e:
        return False, f"{type(e).__name__}: {e}", time.monotonic() - t0


async def run_one(name: str, cfg: dict[str, str], skip_tools: bool) -> dict:
    model = cfg["model"]
    base_url = cfg["base_url"]
    api_key = cfg["api_key"]

    if not model:
        return {"name": name, "skipped": True, "reason": "no model configured"}

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    print(f"\n[{name}] model={model} base_url={base_url or 'openai-default'}")

    ok1, text1, t1 = await test_simple(client, model, base_url)
    print(f"  simple chat: {'OK' if ok1 else 'FAIL'} ({t1:.1f}s)")
    print(f"    -> {text1[:200]}")

    result = {"name": name, "model": model, "simple_ok": ok1, "simple_time": t1, "simple_text": text1}

    if not skip_tools and ok1:
        ok2, text2, t2 = await test_tool_call(client, model, base_url)
        print(f"  tool call:   {'OK' if ok2 else 'FAIL'} ({t2:.1f}s)")
        print(f"    -> {text2[:200]}")
        result.update({"tool_ok": ok2, "tool_time": t2, "tool_text": text2})

    return result


async def main_async(args):
    endpoints = load_endpoints()
    if args.models:
        wanted = set(args.models.split(","))
        endpoints = {k: v for k, v in endpoints.items() if k in wanted}

    results = []
    for name, cfg in endpoints.items():
        r = await run_one(name, cfg, skip_tools=args.skip_tools)
        results.append(r)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        if r.get("skipped"):
            print(f"  {r['name']:12s} SKIPPED ({r['reason']})")
            continue
        simple = "OK" if r["simple_ok"] else "FAIL"
        tool = "OK" if r.get("tool_ok") else ("FAIL" if "tool_ok" in r else "skipped")
        print(f"  {r['name']:12s} simple={simple:4s}  tool={tool:7s}  model={r['model']}")

    any_fail = any(
        (not r.get("skipped") and (not r["simple_ok"] or r.get("tool_ok") is False))
        for r in results
    )
    return 1 if any_fail else 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", help="comma-separated subset, e.g. base,rl_single")
    p.add_argument("--skip-tools", action="store_true", help="skip tool-calling tests")
    args = p.parse_args()
    sys.exit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
