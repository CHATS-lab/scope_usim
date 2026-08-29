"""Offline VS simulation harness.

Drives N multi-turn dialogues between a frontier-model "agent" and the
Verbalized-Sampling user simulator we plan to train against. For each user
turn we log whether the JSON parsed, how many distinct candidate texts the
model proposed, which one was sampled, and (for P4G) whether the persuadee
agreed to donate. The output is a JSONL of episode records plus a one-page
summary printed at the end.

The script does NOT need a GPU, the slime package, tau2-bench, or convokit.
It uses the same prompts that production training uses
(``usim.p4g.prompts``, the tau2 simulation guidelines file), the same VS
parser (``usim.core.vs_parsing``), and the same OpenAI ``response_format``
JSON schema (``usim.core.vs_schema``).

Usage:
    export OPENAI_API_KEY="<your OpenAI API key>"
    export OPENROUTER_API_KEY="<your OpenRouter API key>"
    python3 scripts/diagnostics/simulate_vs_episodes.py \\
        --env p4g --num-episodes 20 --num-turns 10 \\
        --output results/diagnostics/vs_sim/p4g.jsonl

    python3 scripts/diagnostics/simulate_vs_episodes.py \\
        --env tau2 --num-episodes 20 --num-turns 20 \\
        --output results/diagnostics/vs_sim/tau2.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import litellm

# Make `usim` importable when the script is invoked from anywhere.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from usim.core.vs_parsing import (  # noqa: E402
    extract_json_object,
    sample_candidate,
    validate_candidates,
)
from usim.core.vs_schema import get_vs_response_format  # noqa: E402
from usim.p4g.prompts import (  # noqa: E402
    build_persuadee_system_prompt,
    build_persuader_system_prompt,
)


# Tau2 VS instructions. Kept in sync with
# usim/core/environment/tau2/vs_user_simulator.py (which we can't import here
# without pulling tau2-bench + loguru into the harness).
_TAU2_VS_PROB_INSTRUCTION = """
## Response Diversity Instructions (Overrides Earlier Guidelines)

IMPORTANT: The "Generate one message at a time" guideline above is replaced
for this turn. Instead of committing to a single message, you MUST generate
exactly {n} plausible alternative messages you might naturally send as the
user in your current situation, along with an estimated probability for
each. The {n} alternatives should be genuinely diverse — different
phrasings, different levels of detail, different tones, and different
information you might reveal. Each alternative is still a single chat
message; you are NOT sending {n} messages, you are listing {n} options the
downstream sampler will pick from.

Return ONLY a JSON object with the key "responses" (an array of exactly
{n} dictionaries, each with 'text' and 'probability'):
- 'text': the message you would send, no preamble or explanation
- 'probability': a number in [0.0, 1.0] for how likely you are to send
  this particular message (probabilities should roughly sum to 1.0)

Output ONLY the JSON object, no markdown fences, no explanations, no extra
text before or after.
""".strip()

_TAU2_VS_RANDOM_INSTRUCTION = """
## Response Diversity Instructions (Overrides Earlier Guidelines)

IMPORTANT: The "Generate one message at a time" guideline above is replaced
for this turn. Instead of committing to a single message, you MUST generate
exactly {n} plausible alternative messages you might naturally send as the
user in your current situation. The {n} alternatives should be genuinely
diverse — different phrasings, different levels of detail, different tones,
and different information you might reveal. Each alternative is still a
single chat message; you are NOT sending {n} messages, you are listing {n}
options the downstream sampler will pick from.

Return ONLY a JSON object with the key "responses" (an array of exactly
{n} dictionaries, each with a 'text' field):
- 'text': the message you would send, no preamble or explanation

Output ONLY the JSON object, no markdown fences, no explanations.
""".strip()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("vs_sim")


# Defaults. The user sim and the agent are independently configurable so
# we can test different providers / models without editing this file. The
# defaults below run the user sim against gpt-5-mini direct (training
# target) and the agent against claude-haiku via OpenRouter (cheap, no
# OpenAI quota required). Override per-run with --user-model / --agent-model.
DEFAULT_USER_MODEL = "gpt-5-mini"
DEFAULT_USER_BASE_URL = "https://api.openai.com/v1"
DEFAULT_USER_API_KEY_VAR = "OPENAI_API_KEY"

DEFAULT_AGENT_MODEL = "openrouter/anthropic/claude-haiku-4.5"
DEFAULT_AGENT_API_KEY_VAR = "OPENROUTER_API_KEY"


# --- P4G synthetic personas -------------------------------------------------
# We avoid the convokit dependency by hand-writing three short personas in
# the same style as ``data/p4g/corpus``. Real training uses ~700 personas;
# three is enough to exercise the VS code path across different priors.
_P4G_PERSONAS = [
    {
        "name": "skeptic",
        "text": (
            "<persona>\n"
            "I am a 34-year-old accountant. I am skeptical of charities and "
            "have read about overhead ratios eating most donations. I work "
            "hard for my money and I am wary of being persuaded out of it. "
            "I value clear facts and dislike emotional appeals.\n"
            "</persona>"
        ),
    },
    {
        "name": "warm",
        "text": (
            "<persona>\n"
            "I am a 27-year-old teacher who already volunteers in my community. "
            "I have a soft spot for children's causes, but I am also paying off "
            "student loans, so I am cautious about giving large amounts. I "
            "respond well to specific stories rather than abstract statistics.\n"
            "</persona>"
        ),
    },
    {
        "name": "busy",
        "text": (
            "<persona>\n"
            "I am a 45-year-old contractor. I am brief and to the point, and "
            "I dislike long conversations. I have given to local causes before "
            "but I usually want to wrap things up quickly. I am open to "
            "donating a small amount if the ask is clear.\n"
            "</persona>"
        ),
    },
]


def _persuader_persona() -> str:
    return (
        "<persona>\n"
        "I am a 29-year-old social worker who has worked with international "
        "aid organizations. I genuinely believe in Save the Children's mission "
        "and I am warm but persistent in conversations about giving.\n"
        "</persona>"
    )


# --- tau2 retail prompt ----------------------------------------------------
_TAU2_GUIDELINES_PATH = (
    _REPO_ROOT
    / "external"
    / "tau2-bench"
    / "data"
    / "tau2"
    / "user_simulator"
    / "simulation_guidelines.md"
)

_TAU2_TASKS_PATH = (
    _REPO_ROOT
    / "external"
    / "tau2-bench"
    / "data"
    / "tau2"
    / "domains"
    / "retail"
    / "tasks.json"
)


def _format_tau2_user_system_prompt(
    task: Dict[str, Any],
    vs_method: Optional[str] = None,
    vs_num_samples: int = 5,
) -> str:
    """Build the tau2 user-sim system prompt for a retail task.

    Mirrors ``tau2.user.user_simulator.SYSTEM_PROMPT``: global guidelines
    followed by a ``<scenario>`` block with the task instructions. We
    inline the file read so the harness doesn't need tau2-bench installed.
    If ``vs_method`` is provided, the VS instruction block is appended
    exactly as ``VerbalizedSamplingUserSimulator.system_prompt`` does it.
    """
    guidelines = _TAU2_GUIDELINES_PATH.read_text()
    inst = task["user_scenario"]["instructions"]
    persona = task["user_scenario"].get("persona") or ""
    scenario_lines = []
    if persona:
        scenario_lines.append(f"Persona: {persona}")
    scenario_lines.append(f"Task: {inst.get('task_instructions', '').strip()}")
    if inst.get("reason_for_call"):
        scenario_lines.append(f"Reason for call: {inst['reason_for_call'].strip()}")
    if inst.get("known_info"):
        scenario_lines.append(f"Known info: {inst['known_info'].strip()}")
    if inst.get("unknown_info"):
        scenario_lines.append(f"Unknown info: {inst['unknown_info'].strip()}")
    scenario = "\n".join(scenario_lines)
    base = f"{guidelines}\n\n<scenario>\n{scenario}\n</scenario>"
    if vs_method == "prob":
        return base + "\n\n" + _TAU2_VS_PROB_INSTRUCTION.format(n=vs_num_samples)
    if vs_method == "random":
        return base + "\n\n" + _TAU2_VS_RANDOM_INSTRUCTION.format(n=vs_num_samples)
    return base


def _tau2_agent_system_prompt() -> str:
    """A generic retail-CS agent prompt. Production training uses the tau2
    retail agent prompt; we approximate here since the offline harness only
    cares about the *user*-side metrics."""
    return (
        "You are a customer service agent for a large online retailer. "
        "Help the customer resolve their issue (returns, exchanges, order "
        "lookups, etc.). Ask for the information you need (name, zip, order "
        "ID). Keep responses concise (1-3 sentences). When you have enough "
        "information, propose concrete next steps. If the customer indicates "
        "the task is complete or says ###STOP###, acknowledge and end."
    )


# --- VS user-side call -----------------------------------------------------


_VS_PARSE_FAILURE_SENTINEL = "<<VS_PARSE_FAILED>>"


async def _user_vs_turn(
    user_messages: List[Dict[str, str]],
    response_format: Dict[str, Any],
    method: str,
    semaphore: asyncio.Semaphore,
    user_model: str,
    user_api_key_var: str,
    user_base_url: Optional[str],
    max_retries: int = 2,
    max_tokens: int = 1024,
) -> Dict[str, Any]:
    """Issue one VS user turn and return parse/diversity metrics + chosen text.

    The semaphore caps concurrent API calls. We collapse the VS-retry loop
    into a single record: ``parse_ok`` is True iff *any* attempt parsed,
    and ``num_attempts`` records how many we used.
    """
    last_content = ""
    parse_ok = False
    candidates: List[Dict[str, Any]] = []
    attempts = 0
    error: Optional[str] = None

    for attempt in range(max_retries + 1):
        attempts += 1
        try:
            call_kwargs: Dict[str, Any] = dict(
                model=user_model,
                messages=user_messages,
                temperature=1.0,
                max_tokens=max_tokens,
                response_format=response_format,
                api_key=os.environ[user_api_key_var],
            )
            if user_base_url:
                call_kwargs["base_url"] = user_base_url
            async with semaphore:
                resp = await litellm.acompletion(**call_kwargs)
        except Exception as exc:  # network / API errors — log and retry
            error = f"{type(exc).__name__}: {exc}"
            logger.warning("VS user call failed (attempt %d): %s", attempt + 1, error)
            continue

        last_content = resp.choices[0].message.content or ""
        parsed = extract_json_object(last_content)
        if parsed is None:
            continue
        cands = validate_candidates(parsed)
        if not cands:
            continue
        candidates = cands
        parse_ok = True
        break

    if parse_ok:
        chosen_text = sample_candidate(candidates, method=method)
        unique_texts = {c["text"].strip() for c in candidates}
        return {
            "parse_ok": True,
            "attempts": attempts,
            "num_candidates": len(candidates),
            "num_unique_candidates": len(unique_texts),
            "candidates": [c["text"] for c in candidates],
            "chosen_text": chosen_text,
            "raw_content": last_content,
            "error": None,
        }

    # All retries exhausted. Fall back to raw content if non-empty (matches
    # the production fallback) so the episode can continue.
    fallback_text = (last_content or _VS_PARSE_FAILURE_SENTINEL).strip()
    return {
        "parse_ok": False,
        "attempts": attempts,
        "num_candidates": 0,
        "num_unique_candidates": 0,
        "candidates": [],
        "chosen_text": fallback_text,
        "raw_content": last_content,
        "error": error,
    }


# --- agent-side call -------------------------------------------------------


async def _agent_turn(
    agent_messages: List[Dict[str, str]],
    semaphore: asyncio.Semaphore,
    agent_model: str,
    agent_api_key_var: str,
    max_tokens: int = 512,
) -> str:
    """Plain agent turn (no VS)."""
    try:
        async with semaphore:
            resp = await litellm.acompletion(
                model=agent_model,
                messages=agent_messages,
                temperature=0.7,
                max_tokens=max_tokens,
                api_key=os.environ[agent_api_key_var],
            )
        return (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        logger.warning("Agent call failed: %s", exc)
        return ""


# --- P4G episode runner ----------------------------------------------------


_DONATE_RE = re.compile(r"\[(?:DONATE|GIVE)\s*\$?(\d+(?:\.\d+)?)\]", re.IGNORECASE)


def _extract_donation(text: str) -> float:
    """Return the largest [DONATE $N]/[GIVE $N] amount in ``text`` (else 0)."""
    amounts = [float(m.group(1)) for m in _DONATE_RE.finditer(text or "")]
    return max(amounts) if amounts else 0.0


async def _run_p4g_episode(
    episode_idx: int,
    persona: Dict[str, str],
    num_turns: int,
    vs_num_samples: int,
    vs_method: str,
    response_format: Dict[str, Any],
    semaphore: asyncio.Semaphore,
    user_model: str,
    user_api_key_var: str,
    user_base_url: Optional[str],
    agent_model: str,
    agent_api_key_var: str,
) -> Dict[str, Any]:
    """One full P4G dialogue. Persuader=claude-haiku, persuadee=VS gpt-5-mini."""
    persuader_system = build_persuader_system_prompt(
        _persuader_persona(), word_limit=50, num_turns=num_turns
    )
    persuadee_system = build_persuadee_system_prompt(
        persona["text"],
        word_limit=50,
        verbalized_sampling=True,
        vs_num_samples=vs_num_samples,
        vs_method=vs_method,
    )

    # We track two parallel views of the same conversation, with roles
    # flipped: ``agent_msgs`` is from the persuader's POV, ``user_msgs`` is
    # from the persuadee's POV. Anthropic rejects system-only conversations,
    # so we prime the persuader with a stage-direction user message.
    agent_msgs: List[Dict[str, str]] = [
        {"role": "system", "content": persuader_system},
        {"role": "user", "content": "Begin the conversation with your chat partner."},
    ]
    user_msgs: List[Dict[str, str]] = [{"role": "system", "content": persuadee_system}]

    turns: List[Dict[str, Any]] = []
    donation = 0.0
    half_turns = num_turns // 2

    for t in range(half_turns):
        agent_text = await _agent_turn(
            agent_msgs, semaphore, agent_model, agent_api_key_var
        )
        agent_msgs.append({"role": "assistant", "content": agent_text})
        user_msgs.append({"role": "user", "content": agent_text})

        user_turn = await _user_vs_turn(
            user_msgs, response_format, vs_method, semaphore,
            user_model=user_model,
            user_api_key_var=user_api_key_var,
            user_base_url=user_base_url,
        )
        chosen = user_turn["chosen_text"]
        agent_msgs.append({"role": "user", "content": chosen})
        user_msgs.append({"role": "assistant", "content": chosen})

        turn_donation = _extract_donation(chosen)
        if turn_donation > 0:
            donation = max(donation, turn_donation)

        turns.append({
            "turn": t,
            "agent_text": agent_text,
            **{k: v for k, v in user_turn.items() if k != "raw_content"},
            "turn_donation": turn_donation,
        })

    return {
        "episode": episode_idx,
        "env": "p4g",
        "persona": persona["name"],
        "num_turns": len(turns),
        "donation": donation,
        "turns": turns,
    }


# --- tau2 episode runner ---------------------------------------------------


_TAU2_STOP_TOKENS = ("###STOP###", "###TRANSFER###", "###OUT-OF-SCOPE###")


async def _run_tau2_episode(
    episode_idx: int,
    task: Dict[str, Any],
    num_turns: int,
    vs_num_samples: int,
    vs_method: str,
    response_format: Dict[str, Any],
    semaphore: asyncio.Semaphore,
    user_model: str,
    user_api_key_var: str,
    user_base_url: Optional[str],
    agent_model: str,
    agent_api_key_var: str,
) -> Dict[str, Any]:
    """One full tau2-retail dialogue. Agent=claude-haiku, user=VS gpt-5-mini.

    In real tau2 the user goes first; we follow that convention here so the
    user opens by stating the reason for the call. We cap dialogues at
    ``num_turns`` agent/user exchanges or earlier if the user emits a stop
    token.
    """
    user_system = _format_tau2_user_system_prompt(
        task, vs_method=vs_method, vs_num_samples=vs_num_samples
    )
    agent_system = _tau2_agent_system_prompt()

    user_msgs: List[Dict[str, str]] = [{"role": "system", "content": user_system}]
    agent_msgs: List[Dict[str, str]] = [{"role": "system", "content": agent_system}]

    # Prime the user with a "Hi, how can I help?" so it opens the call.
    opener = "Hi, how can I help you today?"
    agent_msgs.append({"role": "assistant", "content": opener})
    user_msgs.append({"role": "user", "content": opener})

    turns: List[Dict[str, Any]] = []
    stopped = False
    stop_reason: Optional[str] = None

    for t in range(num_turns):
        user_turn = await _user_vs_turn(
            user_msgs, response_format, vs_method, semaphore,
            user_model=user_model,
            user_api_key_var=user_api_key_var,
            user_base_url=user_base_url,
        )
        chosen = user_turn["chosen_text"]
        user_msgs.append({"role": "assistant", "content": chosen})
        agent_msgs.append({"role": "user", "content": chosen})

        if any(tok in chosen for tok in _TAU2_STOP_TOKENS):
            stopped = True
            stop_reason = next(tok for tok in _TAU2_STOP_TOKENS if tok in chosen)
            turns.append({
                "turn": t,
                "agent_text": None,
                **{k: v for k, v in user_turn.items() if k != "raw_content"},
            })
            break

        agent_text = await _agent_turn(
            agent_msgs, semaphore, agent_model, agent_api_key_var
        )
        agent_msgs.append({"role": "assistant", "content": agent_text})
        user_msgs.append({"role": "user", "content": agent_text})

        turns.append({
            "turn": t,
            "agent_text": agent_text,
            **{k: v for k, v in user_turn.items() if k != "raw_content"},
        })

    return {
        "episode": episode_idx,
        "env": "tau2",
        "task_id": task["id"],
        "num_turns": len(turns),
        "stopped": stopped,
        "stop_reason": stop_reason,
        "turns": turns,
    }


# --- summary ---------------------------------------------------------------


def _summarize(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    all_turns = [t for ep in episodes for t in ep["turns"]]
    parse_ok = sum(1 for t in all_turns if t["parse_ok"])
    n_turns = len(all_turns)
    n_cand_when_ok = [t["num_candidates"] for t in all_turns if t["parse_ok"]]
    n_unique_when_ok = [t["num_unique_candidates"] for t in all_turns if t["parse_ok"]]
    attempts = [t["attempts"] for t in all_turns]

    summary: Dict[str, Any] = {
        "num_episodes": len(episodes),
        "num_turns": n_turns,
        "parse_success_rate": parse_ok / n_turns if n_turns else 0.0,
        "mean_attempts_per_turn": sum(attempts) / n_turns if n_turns else 0.0,
        "mean_candidates_per_turn": (
            sum(n_cand_when_ok) / len(n_cand_when_ok) if n_cand_when_ok else 0.0
        ),
        "mean_unique_candidates_per_turn": (
            sum(n_unique_when_ok) / len(n_unique_when_ok) if n_unique_when_ok else 0.0
        ),
    }

    if episodes and episodes[0]["env"] == "p4g":
        donations = [ep["donation"] for ep in episodes]
        summary["mean_donation"] = sum(donations) / len(donations)
        summary["donation_rate"] = sum(1 for d in donations if d > 0) / len(donations)

    if episodes and episodes[0]["env"] == "tau2":
        stops = sum(1 for ep in episodes if ep.get("stopped"))
        summary["stop_rate"] = stops / len(episodes)

    return summary


# --- main ------------------------------------------------------------------


async def _run(args: argparse.Namespace) -> None:
    if not os.environ.get(args.user_api_key_var):
        raise SystemExit(f"{args.user_api_key_var} must be set for the VS user sim.")
    if not os.environ.get(args.agent_api_key_var):
        raise SystemExit(f"{args.agent_api_key_var} must be set for the agent.")

    response_format = get_vs_response_format(args.vs_method, args.vs_num_samples)
    semaphore = asyncio.Semaphore(args.concurrency)

    common = dict(
        num_turns=args.num_turns,
        vs_num_samples=args.vs_num_samples,
        vs_method=args.vs_method,
        response_format=response_format,
        semaphore=semaphore,
        user_model=args.user_model,
        user_api_key_var=args.user_api_key_var,
        user_base_url=args.user_base_url or None,
        agent_model=args.agent_model,
        agent_api_key_var=args.agent_api_key_var,
    )

    if args.env == "p4g":
        tasks = [
            _run_p4g_episode(
                episode_idx=i,
                persona=_P4G_PERSONAS[i % len(_P4G_PERSONAS)],
                **common,
            )
            for i in range(args.num_episodes)
        ]
    elif args.env == "tau2":
        all_tasks = json.loads(_TAU2_TASKS_PATH.read_text())
        chosen_tasks = all_tasks[: args.num_episodes]
        tasks = [
            _run_tau2_episode(episode_idx=i, task=t, **common)
            for i, t in enumerate(chosen_tasks)
        ]
    else:
        raise SystemExit(f"unknown env: {args.env}")

    started = time.time()
    logger.info(
        "Running %d %s episodes (concurrency=%d, vs_num_samples=%d, vs_method=%s)",
        args.num_episodes, args.env, args.concurrency,
        args.vs_num_samples, args.vs_method,
    )
    episodes = await asyncio.gather(*tasks)
    elapsed = time.time() - started
    logger.info("Done in %.1fs", elapsed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for ep in episodes:
            f.write(json.dumps(ep) + "\n")

    summary = _summarize(episodes)
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\n=== VS simulation summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nPer-episode log: {args.output}")
    print(f"Summary:         {summary_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env", choices=("p4g", "tau2"), required=True)
    p.add_argument("--num-episodes", type=int, default=30)
    p.add_argument(
        "--num-turns",
        type=int,
        default=None,
        help="Default 10 for p4g, 20 for tau2.",
    )
    p.add_argument("--vs-num-samples", type=int, default=5)
    p.add_argument("--vs-method", choices=("prob", "random"), default="random")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--output", type=Path, required=True)

    # User sim (the model that does VS).
    p.add_argument("--user-model", default=DEFAULT_USER_MODEL)
    p.add_argument("--user-base-url", default=DEFAULT_USER_BASE_URL,
                   help="Empty string for litellm's provider-default routing.")
    p.add_argument("--user-api-key-var", default=DEFAULT_USER_API_KEY_VAR)

    # Agent (the conversation partner, no VS).
    p.add_argument("--agent-model", default=DEFAULT_AGENT_MODEL)
    p.add_argument("--agent-api-key-var", default=DEFAULT_AGENT_API_KEY_VAR)
    args = p.parse_args()

    if args.num_turns is None:
        args.num_turns = 10 if args.env == "p4g" else 20

    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
