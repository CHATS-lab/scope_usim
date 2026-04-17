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


def evaluate_completed_session(
    split: str,
    task_id: str,
    turns: list[dict[str, Any]],
) -> dict[str, Any]:
    """Run tau2-bench's evaluator against a completed conversation.

    Args:
        split: "retail" or "airline".
        task_id: the tau2 task id.
        turns: a list of OpenAI-format message dicts in order. Must include
            assistant tool_calls and tool results.

    Returns:
        {success: bool, reward: float, breakdown: dict, reason: str}.
        If tau2 evaluator is unavailable or fails, returns success=False with
        a descriptive reason.
    """
    if not TAU2_AVAILABLE:
        return {
            "success": False,
            "reward": 0.0,
            "breakdown": {},
            "reason": "tau2-bench not available on the backend.",
        }
    try:
        from tau2.data_model.message import (
            AssistantMessage,
            ToolCall as Tau2ToolCall,
            ToolMessage as Tau2ToolMessage,
            UserMessage,
        )
        from tau2.data_model.simulation import SimulationRun, TerminationReason
        from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
        from tau2.utils.utils import get_now
    except Exception as e:  # noqa: BLE001
        return {
            "success": False,
            "reward": 0.0,
            "breakdown": {},
            "reason": f"tau2 evaluator import failed: {e}",
        }

    # Look up the Task so we have evaluation_criteria.
    _, get_tasks = _domain_env_factory(split)
    tasks = get_tasks()
    matches = [t for t in tasks if t.id == task_id]
    if not matches:
        return {
            "success": False,
            "reward": 0.0,
            "breakdown": {},
            "reason": f"tau2 task id {task_id!r} not found",
        }
    task = matches[0]

    tau2_msgs: list[Any] = []
    for m in turns:
        role = m.get("role")
        if role == "user":
            tau2_msgs.append(UserMessage(content=m.get("content")))
        elif role == "assistant":
            tool_calls = None
            if m.get("tool_calls"):
                tool_calls = []
                for tc in m["tool_calls"]:
                    fn = tc.get("function") or {}
                    tool_calls.append(
                        Tau2ToolCall(
                            id=tc.get("id", ""),
                            name=fn.get("name") or tc.get("name", ""),
                            arguments=fn.get("arguments") or tc.get("arguments") or {},
                            requestor="assistant",
                        )
                    )
            tau2_msgs.append(
                AssistantMessage(content=m.get("content"), tool_calls=tool_calls)
            )
        elif role == "tool":
            tau2_msgs.append(
                Tau2ToolMessage(
                    id=m.get("tool_call_id", ""),
                    content=m.get("content"),
                    requestor="assistant",
                )
            )

    try:
        simulation = SimulationRun(
            id="human-study",
            task_id=task.id,
            start_time=get_now(),
            end_time=get_now(),
            duration=0.0,
            termination_reason=TerminationReason.USER_STOP,
            messages=tau2_msgs,
        )
        reward_info = evaluate_simulation(
            simulation=simulation,
            task=task,
            evaluation_type=EvaluationType.ALL,
            solo_mode=False,
            domain=split,
        )
        reward = float(reward_info.reward)
        return {
            "success": reward >= 0.999,
            "reward": reward,
            "breakdown": dict(getattr(reward_info, "reward_breakdown", {}) or {}),
            "reason": "",
        }
    except Exception as e:  # noqa: BLE001
        logger.exception("tau2 evaluation failed for task %s: %s", task.id, e)
        return {
            "success": False,
            "reward": 0.0,
            "breakdown": {},
            "reason": f"evaluation error: {type(e).__name__}: {e}",
        }
