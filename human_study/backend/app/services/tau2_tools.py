"""Thin wrapper around tau2-bench's Environment for the human study.

We intentionally bypass tau2's `AgentGymEnv` (which couples tool execution with
its own LLM-based user simulator) because the human *is* the user. What we need
from tau2 is just:

  1. The per-domain Environment (with DB + tools preloaded)
  2. The OpenAI-schema tool definitions so the policy can call them
  3. A dispatch function that executes a named tool with kwargs

Everything else (messages, turns, conversation state) is owned by our backend.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

try:
    from tau2.domains.airline.environment import get_environment as _get_airline_env
    from tau2.domains.airline.environment import get_tasks as _get_airline_tasks
    from tau2.domains.retail.environment import get_environment as _get_retail_env
    from tau2.domains.retail.environment import get_tasks as _get_retail_tasks

    TAU2_AVAILABLE = True
except Exception as _e:  # pragma: no cover
    TAU2_AVAILABLE = False
    _IMPORT_ERROR = _e


def _domain_env_factory(split: str):
    if not TAU2_AVAILABLE:
        raise RuntimeError(f"tau2-bench not importable: {_IMPORT_ERROR!r}")
    if split == "retail":
        return _get_retail_env, _get_retail_tasks
    if split == "airline":
        return _get_airline_env, _get_airline_tasks
    raise ValueError(f"Unknown tau2 split: {split!r}")


@dataclass
class Tau2Runtime:
    """Per-session tool runtime. Owns one tau2 Environment and the task."""

    split: str  # "retail" | "airline"
    task_id: str
    env: Any = None
    task: Any = None
    tools_openai: list[dict[str, Any]] = field(default_factory=list)

    def reset(self) -> None:
        get_env, get_tasks = _domain_env_factory(self.split)
        self.env = get_env(solo_mode=False)
        # Task lookup is by string id (tau2 uses "0", "1", ... for retail/airline).
        tasks = get_tasks()
        matches = [t for t in tasks if t.id == self.task_id]
        if not matches:
            raise KeyError(f"tau2 task id={self.task_id!r} not found in split={self.split}")
        self.task = matches[0]
        self.tools_openai = [t.openai_schema for t in self.env.get_tools()]

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Dispatch a tool call. Returns JSON-serialisable result wrapper."""
        if self.env is None:
            raise RuntimeError("Tau2Runtime.reset() must be called before execute().")
        try:
            result = self.env.use_tool(tool_name=name, **arguments)
            # tau2 tools return pydantic models, dicts, primitives, or lists.
            return {"ok": True, "result": _jsonify(result)}
        except Exception as e:  # noqa: BLE001
            logger.warning("tau2 tool %s(%s) failed: %s", name, arguments, e)
            return {"ok": False, "error": str(e), "error_type": type(e).__name__}


def _jsonify(value: Any) -> Any:
    """Best-effort conversion of tau2 return types to JSON-friendly values."""
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, list):
        return [_jsonify(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonify(v) for k, v in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def parse_tool_arguments(raw: str) -> dict[str, Any]:
    """OpenAI returns tool arguments as a JSON string; decode defensively."""
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Tool arguments were not valid JSON: %r", raw[:200])
        return {"_raw": raw}


def instruction_from_task(task: Any) -> str:
    """Produce a human-readable, Sim2Real-style instruction for the right panel."""
    return str(task.user_scenario).strip()
