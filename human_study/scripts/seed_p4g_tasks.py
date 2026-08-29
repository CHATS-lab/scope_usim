"""Seed the p4g task pool from the convokit P4G corpus.

Picks N diverse persuadee speakers and emits a tasks/p4g.yaml entry per
speaker. Each entry's `instruction` block is the participant-facing prompt
(rendered in the right panel of the study UI) — humanised from the speaker's
demographic/personality metadata so the participant has a concrete persona to
role-play.

Usage:
    python scripts/seed_p4g_tasks.py                       # 30 personas
    python scripts/seed_p4g_tasks.py --num-tasks 50 --seed 7
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT.parent / "data" / "p4g"
CORPUS_USERS = DATA_DIR / "corpus" / "users.json"
TRAIN_DIR = DATA_DIR / "train"
OUT_PATH = ROOT / "tasks" / "p4g.yaml"


def find_persuadee_speaker_ids() -> set[str]:
    """Scan train/*.jsonl and collect speaker_ids that appear with role=1 (EE)."""
    ee_speakers: set[str] = set()
    for fp in sorted(TRAIN_DIR.glob("*.jsonl")):
        with fp.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("role") == 1 and r.get("speaker"):
                    ee_speakers.add(r["speaker"])
                    break  # one EE per file
    return ee_speakers


def load_personas() -> dict[str, dict[str, Any]]:
    with CORPUS_USERS.open() as f:
        return json.load(f)


def _bucket_age(age: float | None) -> str:
    if not age or age != age:  # NaN
        return "adult"
    if age < 25:
        return "young"
    if age < 40:
        return "younger middle-aged"
    if age < 60:
        return "middle-aged"
    return "older"


def _bucket_personality(p: dict[str, Any]) -> str:
    """Pull 1-2 personality traits that stand out (>=4 or <=2 on the 1-5 scale)."""
    salient: list[str] = []
    for name, key in [
        ("extraverted", "extrovert"),
        ("introverted", "extrovert"),
        ("agreeable", "agreeable"),
        ("disagreeable", "agreeable"),
        ("conscientious", "conscientious"),
        ("neurotic", "neurotic"),
        ("emotionally stable", "neurotic"),
        ("open to new experiences", "open"),
    ]:
        v = p.get(key)
        if v is None:
            continue
        if "in" + name.split(" ")[0][:3].lower() in ("inext", "inint", "indis", "inemo"):
            # antonym entries — use inverted threshold
            if v <= 2.0:
                salient.append(name)
        elif v >= 4.0 and name not in (
            "introverted",
            "disagreeable",
            "emotionally stable",
        ):
            salient.append(name)
    return ", ".join(salient[:2]) if salient else "balanced personality"


def _build_instruction(persona: dict[str, Any]) -> str:
    """Compose a participant-facing instruction blurb from raw persona meta."""
    age = persona.get("age")
    sex = (persona.get("sex") or "").lower()
    race = persona.get("race") or ""
    edu = persona.get("edu") or ""
    marital = (persona.get("marital") or "").lower()
    employ = persona.get("employment") or ""
    religion = persona.get("religion") or ""
    ideology = persona.get("ideology") or ""
    traits = _bucket_personality(persona)

    # Build a one-paragraph humanised summary. Avoid medical/sensitive
    # claims; stick to what's in the corpus.
    parts: list[str] = []
    if age and sex:
        parts.append(f"{int(age)}-year-old {sex}")
    elif sex:
        parts.append(sex)
    if race and race.lower() not in {"other", "prefer not to say"}:
        parts.append(race.lower())
    if marital and marital not in {"prefer not to say", "other", ""}:
        parts.append(marital)
    descriptor = ", ".join(parts) if parts else "adult"

    blurb_bits: list[str] = []
    if edu and edu.lower() not in {"other", "prefer not to say"}:
        blurb_bits.append(f"You finished {edu.lower()}.")
    if employ and employ.lower() not in {"other", "prefer not to say"}:
        blurb_bits.append(f"You are {employ.lower()}.")
    if religion and religion.lower() not in {"other religion", "other", "prefer not to say"}:
        blurb_bits.append(f"You identify as {religion.lower()}.")
    if ideology and ideology.lower() not in {"other", "prefer not to say"}:
        blurb_bits.append(f"Politically you describe yourself as {ideology.lower()}.")
    if traits:
        blurb_bits.append(f"You tend to be {traits}.")
    blurb = " ".join(blurb_bits)

    return (
        f"You are role-playing as a {descriptor}. "
        + blurb
        + "\n\n"
        "You're chatting with someone who will try to convince you to donate "
        "some portion of your $2 task payment to **Save the Children**, an "
        "international charity. Engage naturally:\n"
        "- Let the conversation develop. Don't agree or refuse on the first "
        "message — ask questions, react authentically to what they say.\n"
        "- If you decide to donate, state a specific dollar amount between "
        "$0.00 and $2.00.\n"
        "- It's fine to decline if they don't convince you.\n"
        "- Stay in character based on the persona above.\n\n"
        "Rules:\n"
        "- Send one short message at a time, like a real chat.\n"
        "- Don't paste this instruction text into the chat.\n"
        "- Don't hallucinate extra biography beyond what's above.\n\n"
        "When the conversation has run its course (or you've stated a final "
        "donation amount), send `/stop` to end and continue to the survey."
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-tasks", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out", default=str(OUT_PATH), help="output yaml path"
    )
    args = p.parse_args()

    if not CORPUS_USERS.exists():
        raise SystemExit(
            f"Corpus users.json missing at {CORPUS_USERS}. "
            "Check that data/p4g/corpus is checked out."
        )

    print(f"Scanning {TRAIN_DIR} for persuadee speaker ids…")
    ee_ids = find_persuadee_speaker_ids()
    print(f"  found {len(ee_ids)} persuadee speakers")

    personas = load_personas()
    candidates = [sid for sid in ee_ids if sid in personas]
    print(f"  {len(candidates)} of those have persona metadata")

    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    picked = candidates[: args.num_tasks]
    print(f"Sampling {len(picked)} personas (seed={args.seed})")

    tasks: list[dict[str, Any]] = []
    for idx, sid in enumerate(picked):
        instruction = _build_instruction(personas[sid])
        tasks.append(
            {
                "idx": idx,
                "instruction": instruction,
                "metadata": {
                    "corpus": "p4g",
                    "speaker_id": sid,
                    "age": personas[sid].get("age"),
                    "ideology": personas[sid].get("ideology"),
                },
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Persuasion for Good task pool for the human study.\n"
        f"# Auto-generated by scripts/seed_p4g_tasks.py (seed={args.seed}, n={len(tasks)}).\n"
        "# Edit the seeder, not this file directly.\n\n"
    )
    with out.open("w") as f:
        f.write(header)
        yaml.safe_dump(tasks, f, sort_keys=False, default_flow_style=False, width=100)

    print(f"Wrote {len(tasks)} tasks to {out}")


if __name__ == "__main__":
    main()
