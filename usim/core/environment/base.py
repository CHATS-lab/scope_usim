"""Base protocol for Gym-based environment implementations.

Environments wrap domain-specific logic (user simulation, tool execution,
observation conversion, response parsing) behind a standard async Gym interface.
The orchestrator is environment-agnostic — it only calls reset/step/parse.
"""

from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, runtime_checkable


@runtime_checkable
class BaseEnvironment(Protocol):
    """Protocol for Gym-based environments.

    Each environment handles:
    - User simulation and tool execution (inside step)
    - Observation → OpenAI message conversion (inside reset)
    - Agent response parsing (parse_response)
    - Optional prompt postprocessing (prompt_postprocess_fn)

    Implementations:
    - tau2-bench: AgentGymEnv wrapper with tool call parsing
    - Persuasion for Good: text-only conversation (no tools)
    """

    async def reset(self) -> Tuple[List[Dict[str, Any]], Optional[List[Dict[str, Any]]], Dict[str, Any]]:
        """Reset environment and return initial state.

        Returns:
            Tuple of:
            - initial_messages: OpenAI-style conversation messages (system + observation)
            - tools_schema: Tool schemas in OpenAI format, or None if no tools
            - task_info: Task metadata dict (id, domain, etc.)
        """
        ...

    async def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        """Step environment with agent action.

        The environment handles user simulation and tool execution internally.

        Args:
            action: Agent's action (plain text or JSON-encoded tool call)

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        ...

    def parse_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Parse agent response into structured format.

        Uses internally stored tools_schema from reset().

        Args:
            response_text: Raw LLM response text

        Returns:
            {"normal_text": str, "calls": list} or None on failure
        """
        ...

    @property
    def prompt_postprocess_fn(self) -> Optional[Callable[[str], str]]:
        """Optional function to transform prompt text after chat template.

        Environment-specific text transformation applied to prompts before
        tokenization (e.g., tau2-bench replaces multi-tool instruction with
        single-tool instruction for Qwen3).
        """
        ...
