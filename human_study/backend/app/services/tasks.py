"""Task pool loaders for τ²-bench and P4G.

Tasks are stored as YAML under usim/human_study/tasks/. Each YAML file
corresponds to a task-type split (tau2_retail.yaml, tau2_airline.yaml, p4g.yaml)
and contains a list of task dicts with at least: idx, instruction, metadata.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

TASKS_DIR = Path(__file__).resolve().parents[2].parent / "tasks"


@lru_cache(maxsize=8)
def load_tasks(split: str) -> list[dict[str, Any]]:
    path = TASKS_DIR / f"{split}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No task file at {path}. Run scripts/seed_tasks.py.")
    with path.open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must be a YAML list of task dicts.")
    return data


def get_task(split: str, idx: int) -> dict[str, Any]:
    tasks = load_tasks(split)
    for t in tasks:
        if t.get("idx") == idx:
            return t
    raise KeyError(f"Task idx={idx} not found in split={split}")


def pick_task_for_session(task_type: str, participants_so_far: int) -> tuple[str, int]:
    """Simple round-robin task assignment.

    Returns (split, idx). For τ²-bench, we alternate retail/airline to balance.
    """
    if task_type == "p4g":
        tasks = load_tasks("p4g")
        return "p4g", tasks[participants_so_far % len(tasks)]["idx"]
    # τ²-bench: alternate retail/airline
    split = "tau2_retail" if participants_so_far % 2 == 0 else "tau2_airline"
    tasks = load_tasks(split)
    idx_in_split = (participants_so_far // 2) % len(tasks)
    return split, tasks[idx_in_split]["idx"]
