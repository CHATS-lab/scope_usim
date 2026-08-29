"""Multi-turn rollouts: 4 agent models × 2 settings (tau2 retail, p4g) with
gpt-5.4 as the fixed user simulator.

Smoke test that the 4 agent endpoints can actually carry a conversation
end-to-end, not just respond to a single prompt:
  - tau2: Tau2Runtime executes tools, gpt-5.4 plays the customer persona.
  - p4g:  gpt-5.4 plays the persuadee, the agent plays the persuader.

Per (model, setting, episode) we record:
  - turn-by-turn transcript
  - tool calls made (tau2)
  - termination reason (max_turns / agent_transferred / error)
  - quick success indicators (any tool calls / final donation mention)

Usage:
    python scripts/simulate_rollouts.py
    python scripts/simulate_rollouts.py --models base,qwen3_moe --settings tau2
    python scripts/simulate_rollouts.py --episodes-per-combo 2 --max-turns 8

Requires tau2-bench installed (`pip install -e /path/to/tau2-bench`).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import dotenv_values
from openai import AsyncOpenAI


ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / "backend" / ".env"
sys.path.insert(0, str(ROOT / "backend"))

from app.services.tau2_tools import Tau2Runtime, parse_tool_arguments  # noqa: E402


USER_SIM_MODEL = "gpt-5.4"


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
    }, fallback_key


def is_reasoning(model: str) -> bool:
    return model.startswith(("gpt-5", "o1", "o3", "o4"))


def gen_kwargs(model: str, max_tokens: int, has_tools: bool = False) -> dict:
    if is_reasoning(model):
        kw: dict = {"max_completion_tokens": max_tokens}
        if not has_tools:
            kw["reasoning_effort"] = "medium"
        return kw
    return {"max_tokens": max_tokens, "temperature": 0.7}


def maybe_extra_body(base_url: str | None) -> dict:
    if base_url:
        return {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    return {}


# -------------------------- TAU2 -------------------------- #

def tau2_user_system_prompt(user_scenario: str) -> str:
    """Build the customer-persona system prompt for the gpt-5.4 user sim."""
    return (
        "You are role-playing as a customer chatting with a customer-service "
        "agent. Stay in character. Reveal information gradually, only when the "
        "agent asks. Keep messages natural and one short message per turn. "
        "Do NOT call tools — you're the customer, the agent uses tools.\n\n"
        f"YOUR SCENARIO:\n{user_scenario}\n\n"
        "When the agent has resolved your request (or said it cannot), say "
        "thanks and end with '<<END>>' on its own line."
    )


async def run_tau2_episode(
    *,
    agent_client: AsyncOpenAI,
    agent_model: str,
    agent_base_url: str | None,
    user_client: AsyncOpenAI,
    task_split: str,
    task_id: str,
    max_turns: int,
) -> dict[str, Any]:
    rt = Tau2Runtime(split=task_split, task_id=task_id)
    rt.reset()

    user_scenario = str(rt.task.user_scenario).strip()
    agent_system = (
        "You are a helpful customer-service agent. Use the provided tools to "
        "look up information and complete the user's request. Be concise. "
        "Verify details before any irreversible action."
    )
    user_system = tau2_user_system_prompt(user_scenario)

    transcript: list[dict[str, Any]] = []
    agent_messages: list[dict[str, Any]] = [{"role": "system", "content": agent_system}]
    user_messages: list[dict[str, Any]] = [{"role": "system", "content": user_system}]
    tool_call_count = 0
    termination = "max_turns"

    # User opens the conversation.
    t0 = time.monotonic()
    user_resp = await user_client.chat.completions.create(
        model=USER_SIM_MODEL,
        messages=user_messages + [{"role": "user", "content": "Start the conversation. Greet the agent and state your goal in one short message."}],
        **gen_kwargs(USER_SIM_MODEL, 200, has_tools=False),
    )
    user_text = (user_resp.choices[0].message.content or "").strip()
    transcript.append({"role": "user", "content": user_text})
    agent_messages.append({"role": "user", "content": user_text})
    user_messages.append({"role": "assistant", "content": user_text})

    for turn in range(max_turns):
        # --- Agent turn (with tool-call loop, max 4 rounds) ---
        for _ in range(4):
            try:
                resp = await agent_client.chat.completions.create(
                    model=agent_model,
                    messages=agent_messages,
                    tools=rt.tools_openai,
                    tool_choice="auto",
                    **gen_kwargs(agent_model, 800, has_tools=True),
                    **maybe_extra_body(agent_base_url),
                )
            except Exception as e:
                termination = f"agent_error: {type(e).__name__}: {e}"
                return _wrap_result(transcript, tool_call_count, termination, time.monotonic() - t0, task_id, task_split)
            msg = resp.choices[0].message
            content = msg.content or ""
            tool_calls = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in (msg.tool_calls or [])
            ] or None

            transcript.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
            agent_messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls} if tool_calls else {"role": "assistant", "content": content})

            if not tool_calls:
                break

            for tc in tool_calls:
                fn = tc["function"]
                if fn["name"] == "transfer_to_human_agents":
                    termination = "agent_transferred"
                args = parse_tool_arguments(fn["arguments"])
                result = rt.execute(fn["name"], args)
                tool_call_count += 1
                tool_content = json.dumps(result, ensure_ascii=False, default=str)
                transcript.append({"role": "tool", "tool_call_id": tc["id"], "name": fn["name"], "content": tool_content})
                agent_messages.append({"role": "tool", "tool_call_id": tc["id"], "name": fn["name"], "content": tool_content})
            if termination == "agent_transferred":
                return _wrap_result(transcript, tool_call_count, termination, time.monotonic() - t0, task_id, task_split)

        # --- User turn ---
        user_messages.append({"role": "user", "content": f"Agent: {transcript[-1]['content'] or '(only tool calls)'}\n\nReply as the customer."})
        try:
            user_resp = await user_client.chat.completions.create(
                model=USER_SIM_MODEL,
                messages=user_messages,
                **gen_kwargs(USER_SIM_MODEL, 200, has_tools=False),
            )
        except Exception as e:
            termination = f"user_error: {type(e).__name__}: {e}"
            return _wrap_result(transcript, tool_call_count, termination, time.monotonic() - t0, task_id, task_split)
        user_text = (user_resp.choices[0].message.content or "").strip()
        user_messages.append({"role": "assistant", "content": user_text})
        transcript.append({"role": "user", "content": user_text})
        agent_messages.append({"role": "user", "content": user_text})

        if "<<END>>" in user_text:
            termination = "user_ended"
            break

    return _wrap_result(transcript, tool_call_count, termination, time.monotonic() - t0, task_id, task_split)


def _wrap_result(transcript, tool_calls, termination, duration, task_id, task_split):
    return {
        "transcript": transcript,
        "tool_call_count": tool_calls,
        "termination": termination,
        "duration_s": round(duration, 2),
        "task_id": task_id,
        "task_split": task_split,
    }


# -------------------------- P4G -------------------------- #

# Hardcoded persona for smoke testing. In production these come from the
# convokit corpus via usim/p4g/persona.py.
_P4G_PERSONA = """<persona>
Demographics:
- Age: 34, Female, Master's degree
- Income: $55,000/year, employed full-time as a software developer
- Religion: Agnostic
- Political: Moderate
- Race: Asian

Personality (Big Five, 1-5 scale):
- Openness: 4, Conscientiousness: 4, Extraversion: 2
- Agreeableness: 4, Neuroticism: 2

Values: education, fairness, family. Slightly skeptical of large charities
because of past news about overhead costs. Donates occasionally to specific
local causes she's researched.
</persona>"""

_P4G_PERSUADEE_PROMPT = f"""You are an Amazon Mechanical Turk worker completing a $2 communication task.

Act naturally as the person in the <persona> tag. Respond in a real-time chat
interface. Keep replies under 50 words, conversational, and authentic. Do not
explain your reasoning. Do not call tools.

{_P4G_PERSONA}

<scenario>
You are randomly assigned to be the Persuadee. Your partner (the persuader)
will try to convince you to donate some portion of your $2 payment to Save the
Children. Engage naturally — you may agree to donate any amount from $0 to
$2.00, or decline. State a specific dollar amount if you decide to donate.
</scenario>

When the conversation feels done (or you've stated a final donation amount),
end your reply with '<<END>>' on its own line."""


_P4G_PERSUADER_PROMPT = """You are a persuader trying to encourage the user to
donate to Save the Children, an international charity helping children in need.
Engage naturally and respectfully. Build rapport first. Ask for a specific
dollar amount between $0 and $2.00 (their MTurk task payment). The donation
will be deducted directly from their payment. Keep replies under 50 words."""


async def run_p4g_episode(
    *,
    agent_client: AsyncOpenAI,
    agent_model: str,
    agent_base_url: str | None,
    user_client: AsyncOpenAI,
    max_turns: int,
) -> dict[str, Any]:
    transcript: list[dict[str, Any]] = []
    agent_messages = [{"role": "system", "content": _P4G_PERSUADER_PROMPT}]
    user_messages = [{"role": "system", "content": _P4G_PERSUADEE_PROMPT}]
    termination = "max_turns"
    t0 = time.monotonic()

    # Persuader (agent) opens.
    try:
        resp = await agent_client.chat.completions.create(
            model=agent_model,
            messages=agent_messages + [{"role": "user", "content": "Start the conversation with a friendly opener."}],
            **gen_kwargs(agent_model, 200, has_tools=False),
            **maybe_extra_body(agent_base_url),
        )
    except Exception as e:
        return {"transcript": transcript, "termination": f"agent_error: {type(e).__name__}: {e}", "duration_s": round(time.monotonic() - t0, 2)}
    opener = (resp.choices[0].message.content or "").strip()
    transcript.append({"role": "assistant", "content": opener})
    agent_messages.append({"role": "assistant", "content": opener})

    for _ in range(max_turns):
        # Persuadee (gpt-5.4) responds.
        try:
            user_resp = await user_client.chat.completions.create(
                model=USER_SIM_MODEL,
                messages=user_messages + [{"role": "user", "content": f"Persuader: {transcript[-1]['content']}\n\nReply as the persuadee."}],
                **gen_kwargs(USER_SIM_MODEL, 200, has_tools=False),
            )
        except Exception as e:
            termination = f"user_error: {type(e).__name__}: {e}"
            break
        user_text = (user_resp.choices[0].message.content or "").strip()
        user_messages.append({"role": "user", "content": f"Persuader: {transcript[-1]['content']}"})
        user_messages.append({"role": "assistant", "content": user_text})
        transcript.append({"role": "user", "content": user_text})
        agent_messages.append({"role": "user", "content": user_text})

        if "<<END>>" in user_text:
            termination = "user_ended"
            break

        # Persuader replies.
        try:
            resp = await agent_client.chat.completions.create(
                model=agent_model,
                messages=agent_messages,
                **gen_kwargs(agent_model, 200, has_tools=False),
                **maybe_extra_body(agent_base_url),
            )
        except Exception as e:
            termination = f"agent_error: {type(e).__name__}: {e}"
            break
        msg_text = (resp.choices[0].message.content or "").strip()
        transcript.append({"role": "assistant", "content": msg_text})
        agent_messages.append({"role": "assistant", "content": msg_text})

    duration = time.monotonic() - t0

    # Simple donation detection — look for $X.XX patterns in final user msgs.
    import re
    donation = None
    for t in reversed(transcript):
        if t["role"] == "user":
            m = re.search(r"\$\s*(\d+(?:\.\d{1,2})?)", t["content"] or "")
            if m:
                donation = float(m.group(1))
                break

    return {
        "transcript": transcript,
        "termination": termination,
        "duration_s": round(duration, 2),
        "donation_usd": donation,
    }


# -------------------------- Driver -------------------------- #


async def main_async(args):
    endpoints, fallback_key = load_endpoints()
    if args.models:
        wanted = set(args.models.split(","))
        endpoints = {k: v for k, v in endpoints.items() if k in wanted}

    settings_to_run = args.settings.split(",")

    user_client = AsyncOpenAI(api_key=fallback_key)

    results: list[dict[str, Any]] = []
    tau2_task_ids = ["0", "5"]  # 2 retail tasks for variety

    for name, cfg in endpoints.items():
        if not cfg["model"]:
            continue
        agent_client = AsyncOpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])
        agent_model = cfg["model"]
        agent_base = cfg["base_url"]

        if "tau2" in settings_to_run:
            for ep in range(args.episodes_per_combo):
                task_id = tau2_task_ids[ep % len(tau2_task_ids)]
                print(f"\n[tau2 / {name} / ep{ep} / task={task_id}] model={agent_model}")
                r = await run_tau2_episode(
                    agent_client=agent_client,
                    agent_model=agent_model,
                    agent_base_url=agent_base,
                    user_client=user_client,
                    task_split="retail",
                    task_id=task_id,
                    max_turns=args.max_turns,
                )
                r.update({"agent_model": name, "setting": "tau2", "episode": ep})
                print(f"  -> {r['termination']} | {len(r['transcript'])} msgs | {r['tool_call_count']} tools | {r['duration_s']}s")
                results.append(r)

        if "p4g" in settings_to_run:
            for ep in range(args.episodes_per_combo):
                print(f"\n[p4g  / {name} / ep{ep}] model={agent_model}")
                r = await run_p4g_episode(
                    agent_client=agent_client,
                    agent_model=agent_model,
                    agent_base_url=agent_base,
                    user_client=user_client,
                    max_turns=args.max_turns,
                )
                r.update({"agent_model": name, "setting": "p4g", "episode": ep})
                donation = r.get("donation_usd")
                print(f"  -> {r['termination']} | {len(r['transcript'])} msgs | donation={donation} | {r['duration_s']}s")
                results.append(r)

    # Write JSONL.
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    # Summary table.
    print("\n" + "=" * 70)
    print(f"SUMMARY ({len(results)} episodes, output: {out_path})")
    print("=" * 70)
    print(f"  {'model':12s} {'setting':6s} {'ep':3s} {'msgs':5s} {'tools':5s} {'donation':9s} {'duration':9s} termination")
    for r in results:
        donation = f"${r.get('donation_usd')}" if r.get("donation_usd") is not None else "-"
        tools = str(r.get("tool_call_count", "-"))
        print(
            f"  {r['agent_model']:12s} {r['setting']:6s} {r['episode']:<3d} "
            f"{len(r['transcript']):<5d} {tools:<5s} {donation:<9s} "
            f"{r['duration_s']:<9.1f} {r['termination']}"
        )

    failures = [r for r in results if "error" in r["termination"]]
    return 1 if failures else 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", default="", help="comma-separated subset (default: all 4)")
    p.add_argument("--settings", default="tau2,p4g")
    p.add_argument("--episodes-per-combo", type=int, default=1)
    p.add_argument("--max-turns", type=int, default=6)
    p.add_argument("--output", default="results/simulate_rollouts.jsonl")
    args = p.parse_args()
    sys.exit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
