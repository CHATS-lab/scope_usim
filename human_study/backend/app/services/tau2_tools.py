"""Thin wrapper around tau2-bench's tool runtime.

For each active session we instantiate one `Tau2Runtime` that holds the
per-session database state (e.g., flight bookings, retail orders). Tool calls
from the policy are dispatched here; results are JSON-serialisable dicts that
we both return to the policy and persist in the Turn table.

We deliberately keep the interface small so the backend doesn't depend on
tau2-bench internals directly.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

# NOTE: tau2-bench is assumed to be importable in the runtime environment. The
# actual import path depends on how tau2-bench is installed; adjust here if
# needed.
try:
    from tau2.data_model import Task as Tau2Task  # type: ignore
    from tau2.domains import airline as tau2_airline  # type: ignore
    from tau2.domains import retail as tau2_retail  # type: ignore
    from tau2.environment import Environment as Tau2Env  # type: ignore
    TAU2_AVAILABLE = True
except Exception:
    TAU2_AVAILABLE = False


@dataclass
class Tau2Runtime:
    """Holds the per-session tool runtime state."""

    split: str  # "retail" | "airline"
    task_idx: int
    env: Any = None
    tools_openai: list[dict[str, Any]] = field(default_factory=list)

    def reset(self) -> None:
        if not TAU2_AVAILABLE:
            raise RuntimeError(
                "tau2-bench is not importable; install it and expose its modules "
                "before using the tau2 task flow."
            )
        domain = {"retail": tau2_retail, "airline": tau2_airline}[self.split]
        self.env = Tau2Env.from_domain(domain, task_index=self.task_idx)
        self.tools_openai = _tau2_tools_to_openai(self.env.get_tools())

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if self.env is None:
            raise RuntimeError("Tau2Runtime.reset() must be called before execute().")
        try:
            result = self.env.step_tool(name=name, arguments=arguments)
            return {"ok": True, "result": result}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": str(e), "error_type": type(e).__name__}


def _tau2_tools_to_openai(tools: list[Any]) -> list[dict[str, Any]]:
    """Translate tau2 tool definitions to OpenAI function-calling schema."""
    openai_tools = []
    for t in tools:
        # tau2 tools expose {name, description, parameters (JSON schema)}
        openai_tools.append(
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": getattr(t, "description", "") or "",
                    "parameters": getattr(t, "parameters", {"type": "object", "properties": {}}),
                },
            }
        )
    return openai_tools


def parse_tool_arguments(raw: str) -> dict[str, Any]:
    """OpenAI returns tool arguments as a JSON string; decode defensively."""
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"_raw": raw}
