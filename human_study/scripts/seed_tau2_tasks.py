"""Seed τ²-bench task YAMLs from the installed tau2-bench package.

Run once tau2-bench is importable in the active venv:

    python human_study/scripts/seed_tau2_tasks.py --domain retail --limit 15
    python human_study/scripts/seed_tau2_tasks.py --domain airline --limit 15

Writes to human_study/tasks/tau2_{domain}.yaml.

Each task is emitted with:
    - ``instruction``: flat rendered text (used by the v1 instruction panel)
    - ``structured``: parsed fields (profile / goal / unknown_info /
        conditional_behaviors / style_notes) used by the v2 panel
    - ``metadata``: {domain, tau2_task_id}
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import yaml

TASKS_DIR = Path(__file__).resolve().parents[1] / "tasks"


_TASK_RULES = (
    "Rules:\n"
    "- Just generate one line at a time to simulate the user's message.\n"
    "- Do not give away all the instruction at once. Only provide the "
    "information that is necessary for the current step.\n"
    "- Do not hallucinate information that is not provided in the instruction.\n"
    "- Do not repeat the exact instruction in the conversation.\n"
    "- Try to make the conversation as natural as possible.\n"
)


def _render_instruction(scenario_str: str) -> str:
    return (
        "Instruction: You are role-playing as a user interacting with an agent.\n\n"
        f"{scenario_str.strip()}\n\n"
        f"{_TASK_RULES}\n"
        "Send /stop when the agent has finished the task."
    )


# -----------------------------------------------------------------------------
# Structured parser
# -----------------------------------------------------------------------------

SECTION_LABELS = [
    "Domain",
    "Reason for call",
    "Known info",
    "Unknown info",
    "Task instructions",
]


def _split_sections(raw: str) -> dict[str, str]:
    """Split tau2's tab-indented `Instructions:` block into a dict of sections."""
    lines = raw.replace("\r\n", "\n").split("\n")
    # Drop leading tabs/spaces from all lines so we don't depend on indent level
    current_label: str | None = None
    buckets: dict[str, list[str]] = {}
    for raw_line in lines:
        line = raw_line.replace("\t", "  ")
        stripped = line.strip()
        if not stripped:
            # keep blank lines inside a section so paragraphs are preserved
            if current_label is not None:
                buckets[current_label].append("")
            continue
        matched_label: str | None = None
        for lab in SECTION_LABELS:
            if stripped.rstrip(":").strip().lower() == lab.lower() and stripped.endswith(":"):
                matched_label = lab
                break
        if matched_label:
            current_label = matched_label
            buckets.setdefault(current_label, [])
            continue
        if current_label is None:
            # Top-level noise like "Instructions:" line — ignore
            continue
        buckets[current_label].append(stripped)
    return {k: "\n".join(v).strip() for k, v in buckets.items()}


def _parse_profile(known_text: str) -> list[dict[str, str]]:
    """Convert the "Known info" section into a list of {label, value} pairs."""
    if not known_text:
        return []
    out: list[dict[str, str]] = []
    # Each non-empty line is one fact. Try to pattern-match common ones.
    for raw in known_text.splitlines():
        s = raw.strip()
        if not s:
            continue
        # "You are X in zip code Y"
        m = re.match(r"^you are ([^.,]+?)(?:\s+(?:in zip code|living in zipcode|in zipcode)\s+([A-Za-z0-9-]+))?\.?\s*$", s, re.I)
        if m:
            out.append({"label": "Name", "value": m.group(1).strip()})
            if m.group(2):
                out.append({"label": "ZIP code", "value": m.group(2).strip()})
            continue
        m = re.match(r"^your user id is[:\s]+([A-Za-z0-9_-]+[A-Za-z0-9_])\.?\s*$", s, re.I)
        if m:
            out.append({"label": "User ID", "value": m.group(1).rstrip(".")})
            continue
        m = re.match(r"^your confirmation (?:number|#|code) is[:\s]+([A-Za-z0-9_-]+[A-Za-z0-9_])\.?\s*$", s, re.I)
        if m:
            out.append({"label": "Confirmation #", "value": m.group(1).rstrip(".")})
            continue
        m = re.match(r"^(?:you(?:'re| are) (?:a )?|your membership is\s*)(gold|silver|bronze|platinum|regular)(?:\s+member)?\.?\s*$", s, re.I)
        if m:
            out.append({"label": "Membership", "value": m.group(1).capitalize()})
            continue
        m = re.search(r"(gold|silver|bronze|platinum)\s+member", s, re.I)
        if m:
            # Capture membership from a longer sentence, append the rest verbatim
            out.append({"label": "Membership", "value": m.group(1).capitalize()})
            # Also preserve the full sentence as context
            out.append({"label": "Info", "value": s})
            continue
        # Fallback: attach as generic "Info" line
        out.append({"label": "Info", "value": s})
    return out


def _first_sentence(text: str) -> tuple[str, str]:
    """Return (first_sentence, remaining_paragraphs) for the goal block."""
    text = (text or "").strip()
    if not text:
        return "", ""
    # Split on first period/question mark/exclamation followed by whitespace/EOL
    m = re.search(r"([.!?])(?:\s+|$)", text)
    if not m:
        return text, ""
    head = text[: m.end(1)].strip()
    tail = text[m.end() :].strip()
    return head, tail


def _split_paragraphs(text: str) -> list[str]:
    """Split on blank-line-separated paragraphs, strip each."""
    if not text:
        return []
    return [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]


def _split_sentences(text: str) -> list[str]:
    """Naive sentence splitter: split on period/question/exclamation + whitespace.

    Good enough for tau2 scenarios (no abbreviations like "Dr." in practice).
    """
    if not text:
        return []
    # Collapse any internal newlines to single spaces first so sentences
    # straddling a \n still split correctly.
    compact = re.sub(r"\s+", " ", text.strip())
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", compact)
    return [p.strip() for p in parts if p.strip()]


def _parse_task_instructions(task_text: str) -> tuple[list[dict[str, str]], list[str]]:
    """Return (conditional_behaviors, style_notes).

    Splits into sentences (not paragraphs) so a paragraph like
    ``If X, insist on Y. If A, escalate B.`` produces two behaviors instead
    of one blob. Sentences starting with "If " become conditional behaviors;
    everything else becomes a style note.
    """
    behaviors: list[dict[str, str]] = []
    notes: list[str] = []

    ESCALATE_PATTERNS = re.compile(
        r"\b(supervisor|transfer(?:red)?|escalat\w+|human\s+agent|speak\s+to\s+a\s+person)\b",
        re.I,
    )

    for sentence in _split_sentences(task_text):
        compact = re.sub(r"\s+", " ", sentence).strip()
        if compact in (".", "", "·"):
            continue
        if re.match(r"^if\b", compact, re.I):
            m = re.match(r"^(if[^,]+),\s*(.+)$", compact, re.I)
            if m:
                trigger_raw, action = m.group(1).strip(), m.group(2).strip()
            elif ", " in compact:
                trigger_raw, action = compact.split(", ", 1)
            else:
                trigger_raw, action = compact, ""
            severity = (
                "escalate"
                if ESCALATE_PATTERNS.search(action or trigger_raw)
                else "warn"
            )
            behaviors.append(
                {
                    "severity": severity,
                    "trigger": trigger_raw.rstrip(".,"),
                    "action": action.rstrip(".").strip() if action else "",
                }
            )
        else:
            notes.append(compact)

    return behaviors, notes


_MEMBERSHIP_IN_GOAL = re.compile(
    r"\b(?:you(?:'re| are)\s+)?(?:pretty\s+sure|think|believe)(?:\s+that)?\s+you(?:'re| are)\s+a\s+(gold|silver|bronze|platinum)\s+member\b",
    re.I,
)
_MEMBERSHIP_PLAIN_IN_GOAL = re.compile(
    r"\byou(?:'re| are)\s+a\s+(gold|silver|bronze|platinum)\s+member\b",
    re.I,
)


def _lift_membership_from_goal(
    goal_detail: str | None, profile: list[dict[str, str]]
) -> str | None:
    """If the goal_detail mentions membership (common in airline scenarios),
    hoist it into the profile as a proper field and strip from goal_detail so
    it doesn't get duplicated in the v2 UI."""
    if not goal_detail:
        return goal_detail
    existing_labels = {p["label"] for p in profile}
    if "Membership" in existing_labels:
        return goal_detail

    m = _MEMBERSHIP_IN_GOAL.search(goal_detail) or _MEMBERSHIP_PLAIN_IN_GOAL.search(
        goal_detail
    )
    if not m:
        return goal_detail
    tier = m.group(1).capitalize()
    qualifier = " (pretty sure)" if "pretty" in m.group(0).lower() or "think" in m.group(0).lower() or "believe" in m.group(0).lower() else ""
    profile.append({"label": "Membership", "value": f"{tier}{qualifier}"})
    # Remove the membership sentence from goal_detail so it isn't repeated.
    cleaned = re.sub(
        r"(?:^|(?<=\n))[^\n]*"
        r"(?:pretty\s+sure|think|believe)[^\n]*"
        r"(?:gold|silver|bronze|platinum)\s+member[^\n]*\.?",
        "",
        goal_detail,
        flags=re.I,
    )
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned or None


def _structure_scenario(raw: str, domain: str) -> dict[str, Any]:
    sections = _split_sections(raw)
    reason = sections.get("Reason for call", "")
    goal, goal_detail = _first_sentence(reason)
    profile = _parse_profile(sections.get("Known info", ""))
    unknown = sections.get("Unknown info", "") or None
    # tau2 sometimes has a bare "." placeholder for unknown info
    if unknown and unknown.strip(" .\n\t") == "":
        unknown = None
    behaviors, style_notes = _parse_task_instructions(
        sections.get("Task instructions", "")
    )
    goal_detail = _lift_membership_from_goal(goal_detail or None, profile)
    return {
        "domain": sections.get("Domain", domain),
        "goal": goal,
        "goal_detail": goal_detail,
        "profile": profile,
        "unknown_info": unknown,
        "conditional_behaviors": behaviors,
        "style_notes": style_notes,
    }


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=["retail", "airline"], required=True)
    ap.add_argument("--limit", type=int, default=15)
    ap.add_argument(
        "--split",
        default="base",
        help="tau2 task split name (default 'base'; try None for all).",
    )
    args = ap.parse_args()

    if args.domain == "retail":
        from tau2.domains.retail.environment import get_tasks
    else:
        from tau2.domains.airline.environment import get_tasks

    tasks = get_tasks(task_split_name=args.split)[: args.limit]
    out = []
    for i, t in enumerate(tasks):
        scenario = str(t.user_scenario)
        structured = _structure_scenario(scenario, args.domain)
        out.append(
            {
                "idx": i,
                "instruction": _render_instruction(scenario),
                "structured": structured,
                "metadata": {
                    "domain": args.domain,
                    "tau2_task_id": t.id,
                },
            }
        )

    path = TASKS_DIR / f"tau2_{args.domain}.yaml"
    with path.open("w") as f:
        yaml.safe_dump(out, f, sort_keys=False, width=100, allow_unicode=True)
    print(f"Wrote {len(out)} tasks to {path}")


if __name__ == "__main__":
    main()
