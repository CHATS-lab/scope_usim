"""Seed τ²-bench task YAMLs from the installed tau2-bench package.

Run once tau2-bench is importable in the active venv:

    python human_study/scripts/seed_tau2_tasks.py --domain retail --limit 15
    python human_study/scripts/seed_tau2_tasks.py --domain airline --limit 15

Writes to human_study/tasks/tau2_{domain}.yaml.
"""
from __future__ import annotations

import argparse
from pathlib import Path

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
        out.append(
            {
                "idx": i,
                "instruction": _render_instruction(str(t.user_scenario)),
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
