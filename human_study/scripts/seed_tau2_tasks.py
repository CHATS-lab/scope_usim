"""Seed τ²-bench task YAMLs from the installed tau2-bench package.

Run from the usim/ repo root:

    python human_study/scripts/seed_tau2_tasks.py --limit 15 --domain retail
    python human_study/scripts/seed_tau2_tasks.py --limit 15 --domain airline

Writes to usim/human_study/tasks/tau2_{domain}.yaml.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

TASKS_DIR = Path(__file__).resolve().parents[1] / "tasks"


def _render_instruction(scenario: str) -> str:
    """Wrap a τ²-bench user scenario string into the two-panel instruction shown
    on the right of the UI, matching the Sim2Real annotation format."""
    return (
        "Instruction: You are role-playing as a user interacting with an agent.\n\n"
        f"{scenario.strip()}\n\n"
        "Rules:\n"
        "- Just generate one line at a time to simulate the user's message.\n"
        "- Do not give away all the instruction at once. Only provide the "
        "information that is necessary for the current step.\n"
        "- Do not hallucinate information that is not provided in the instruction.\n"
        "- Do not repeat the exact instruction in the conversation.\n"
        "- Try to make the conversation as natural as possible.\n\n"
        "Send /stop when the agent has finished the task."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=["retail", "airline"], required=True)
    ap.add_argument("--limit", type=int, default=15)
    args = ap.parse_args()

    from tau2.data_model.tasks import get_tasks  # imported lazily

    tasks = get_tasks(args.domain)[: args.limit]
    out = []
    for i, t in enumerate(tasks):
        scenario = getattr(t, "user_scenario", None) or str(t)
        out.append(
            {
                "idx": i,
                "instruction": _render_instruction(str(scenario)),
                "metadata": {
                    "domain": args.domain,
                    "tau2_task_id": getattr(t, "id", f"{args.domain}_task_{i:03d}"),
                },
            }
        )

    path = TASKS_DIR / f"tau2_{args.domain}.yaml"
    with path.open("w") as f:
        yaml.safe_dump(out, f, sort_keys=False, width=100)
    print(f"Wrote {len(out)} tasks to {path}")


if __name__ == "__main__":
    main()
