"""Tau2-bench environment wrapping AgentGymEnv.

Encapsulates all tau2-specific logic:
- AgentGymEnv setup and async reset/step
- Observation → OpenAI message conversion
- Tool call parsing via sglang's FunctionCallParser
- Prompt postprocessing (single-tool instruction for Qwen3)
"""

import asyncio
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.managers.io_struct import Function, Tool
from tau2.agent.llm_agent import AGENT_INSTRUCTION, SYSTEM_PROMPT
from tau2.data_model.message import AssistantMessage, ToolMessage, UserMessage
from tau2.gym import AgentGymEnv


logger = logging.getLogger(__name__)

# Replaces Qwen3's default multi-tool instruction with single-tool for tau2-bench
_TOOL_INSTRUCTION = (
    " At each turn, you are allowed to call one or no function to assist "
    "with task execution using <tools></tools> XML tags.\n"
    "YOU MUST EXECUTE TOOLS TO MAKE ANY MODIFICATIONS OR CANCELLATIONS. "
    "Each tool call leads to a message returned by the system.\n"
    "NEVER confirm execution to the user without seeing confirmation "
    "from the tool system.\n"
)


def _tau2_prompt_postprocess(text: str) -> str:
    """Reformulate tool instruction for tau2-bench (at most one tool call per turn)."""
    return text.replace(
        "You may call one or more functions to assist with the user query.",
        _TOOL_INSTRUCTION,
    )


def _observation_to_messages(
    observation: list, system_prompt: str
) -> List[Dict[str, Any]]:
    """Convert gym observation (tau2 Message objects) to OpenAI-style messages."""
    messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    if not observation:
        return messages

    for msg in observation:
        if isinstance(msg, AssistantMessage):
            if msg.content:
                messages.append({"role": "assistant", "content": msg.content})
            elif msg.tool_calls:
                tool_calls = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": (
                                json.dumps(tc.arguments)
                                if isinstance(tc.arguments, dict)
                                else str(tc.arguments)
                            ),
                        },
                    }
                    for tc in msg.tool_calls
                ]
                messages.append({"role": "assistant", "tool_calls": tool_calls})
        elif isinstance(msg, UserMessage):
            if msg.content:
                messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, ToolMessage):
            messages.append({
                "role": "tool",
                "tool_call_id": msg.id,
                "content": msg.content or "",
            })

    return messages


def _parse_tool_calls(
    response_text: str, tools_schema: Optional[List[Dict[str, Any]]]
) -> Optional[Dict[str, Any]]:
    """Parse tool calls from LLM response using sglang's FunctionCallParser.

    Returns {"normal_text": str, "calls": list} or None on failure.
    """
    if response_text.endswith("<|im_end|>"):
        response_text = response_text[:-10]
    response_text = response_text.strip()

    if not tools_schema:
        return {"normal_text": response_text, "calls": []}

    try:
        tools_list = [
            Tool(
                function=Function(
                    name=t["function"]["name"],
                    description=t["function"]["description"],
                    parameters=t["function"]["parameters"],
                ),
                type=t["type"],
            )
            for t in tools_schema
        ]
        parser = FunctionCallParser(tools=tools_list, tool_call_parser="qwen")
        normal_text, calls = parser.parse_non_stream(response_text)

        normalized_calls = []
        for call in calls:
            call_dict = call.model_dump()
            params = call_dict.get("parameters", call_dict.get("arguments", {}))
            if isinstance(params, str):
                try:
                    params = json.loads(params)
                except (json.JSONDecodeError, TypeError):
                    params = {}
            normalized_calls.append({
                "name": call_dict.get("name", ""),
                "arguments": params,
                "id": call_dict.get("id", ""),
            })

        return {"normal_text": normal_text, "calls": normalized_calls}
    except Exception as e:
        logger.warning(f"Tool parsing failed: {e}")
        return None


class _ConfiguredAgentGymEnv(AgentGymEnv):
    """AgentGymEnv subclass that injects a behavioral prefix into the user simulator prompt."""

    def __init__(self, *args, user_prompt_prefix: str = "", **kwargs):
        self._user_prompt_prefix = user_prompt_prefix
        super().__init__(*args, **kwargs)

    def _get_user(self):
        user = super()._get_user()
        if self._user_prompt_prefix and hasattr(user, "system_prompt"):
            original_fn = type(user).system_prompt.fget
            prefix = self._user_prompt_prefix

            @property
            def patched_system_prompt(self_inner):
                return prefix.strip() + "\n\n" + original_fn(self_inner)

            # Per-instance subclass to avoid mutating the shared base class
            patched_cls = type(
                f"_Patched{type(user).__name__}",
                (type(user),),
                {"system_prompt": patched_system_prompt},
            )
            user.__class__ = patched_cls
        return user


class _VerbalizedSamplingAgentGymEnv(AgentGymEnv):
    """AgentGymEnv subclass that uses a Verbalized Sampling user simulator.

    Overrides ``_get_user()`` to return a
    ``VerbalizedSamplingUserSimulator`` instead of the default
    ``UserSimulator``. All other gym behavior (tool dispatch, reward
    computation, etc.) is unchanged.
    """

    def __init__(
        self,
        *args,
        vs_num_samples: int = 5,
        vs_method: str = "prob",
        user_prompt_prefix: str = "",
        **kwargs,
    ):
        self._vs_num_samples = vs_num_samples
        self._vs_method = vs_method
        self._user_prompt_prefix = user_prompt_prefix
        super().__init__(*args, **kwargs)

    def _get_user(self):
        # We cannot call super()._get_user() because that would construct
        # a plain UserSimulator. Replicate the setup with our VS subclass.
        from tau2.user.user_simulator import DummyUser

        from usim.core.environment.tau2.vs_user_simulator import (
            VerbalizedSamplingUserSimulator,
        )

        environment = self._get_environment()
        task = self._get_task()
        try:
            user_tools = environment.get_user_tools()
        except ValueError:
            user_tools = None

        if self.solo_mode:
            return DummyUser()

        user = VerbalizedSamplingUserSimulator(
            tools=user_tools,
            instructions=task.user_scenario,
            llm=self.user_llm,
            llm_args=self.user_llm_args,
            vs_num_samples=self._vs_num_samples,
            vs_method=self._vs_method,
        )

        # Optional: also stack the behavioral prefix on top of VS, so
        # --usim-user-prompt-prefix and --usim-verbalized-sampling compose.
        if self._user_prompt_prefix:
            original_fn = type(user).system_prompt.fget
            prefix = self._user_prompt_prefix

            @property
            def patched_system_prompt(self_inner):
                return prefix.strip() + "\n\n" + original_fn(self_inner)

            patched_cls = type(
                f"_Patched{type(user).__name__}",
                (type(user),),
                {"system_prompt": patched_system_prompt},
            )
            user.__class__ = patched_cls

        return user


class Tau2Environment:
    """Tau2-bench environment wrapping AgentGymEnv.

    Implements the BaseEnvironment protocol for the orchestrator.
    AgentGymEnv handles user simulation, tool execution, and turn management
    internally — this class just provides the async Gym interface + parsing.
    """

    def __init__(
        self,
        domain: str,
        task_id: str,
        user_model: Optional[str] = None,
        user_base_url: str = "https://api.openai.com/v1",
        user_api_key: Optional[str] = None,
        max_turns: int = 30,
        user_prompt_prefix: str = "",
        verbalized_sampling: bool = False,
        vs_num_samples: int = 5,
        vs_method: str = "prob",
    ):
        self._domain = domain
        self._task_id = task_id

        # litellm needs provider prefix for non-OpenAI models
        # OpenRouter models: openrouter/google/gemini-3-flash-preview
        # OpenAI models: gpt-5-mini (litellm recognizes natively)
        litellm_model = user_model
        if user_model and "openrouter" in (user_base_url or ""):
            if not user_model.startswith("openrouter/"):
                litellm_model = f"openrouter/{user_model}"

        # Pick the right env subclass. VS takes precedence over the
        # configured-prefix baseline: the VS env class still accepts a
        # ``user_prompt_prefix`` kwarg and composes both.
        if verbalized_sampling:
            env_cls = _VerbalizedSamplingAgentGymEnv
        elif user_prompt_prefix:
            env_cls = _ConfiguredAgentGymEnv
        else:
            env_cls = AgentGymEnv

        env_kwargs = dict(
            domain=domain,
            task_id=task_id,
            max_steps=max_turns,
            solo_mode=False,
            user_llm=litellm_model,
            user_llm_args={
                "api_key": user_api_key,
            },
        )
        if verbalized_sampling:
            env_kwargs["vs_num_samples"] = vs_num_samples
            env_kwargs["vs_method"] = vs_method
            env_kwargs["user_prompt_prefix"] = user_prompt_prefix
        elif user_prompt_prefix:
            env_kwargs["user_prompt_prefix"] = user_prompt_prefix

        self._env = env_cls(**env_kwargs)
        self._tools_schema: Optional[List[Dict[str, Any]]] = None

    async def reset(
        self,
    ) -> Tuple[List[Dict[str, Any]], Optional[List[Dict[str, Any]]], Dict[str, Any]]:
        """Reset environment and return initial state."""
        loop = asyncio.get_event_loop()
        _, info = await loop.run_in_executor(None, self._env.reset)

        tools = info.get("tools", [])
        self._tools_schema = [t.openai_schema for t in tools] if tools else None
        policy = info.get("policy", "")

        system_prompt = SYSTEM_PROMPT.format(
            agent_instruction=AGENT_INSTRUCTION,
            domain_policy=policy,
        )

        initial_messages = _observation_to_messages(
            self._env._agent.observation, system_prompt
        )
        task_info = {"id": self._task_id, "domain": self._domain}

        return initial_messages, self._tools_schema, task_info

    async def step(
        self, action: str
    ) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        """Step environment with agent action."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._env.step, action)

    def parse_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Parse agent response for tool calls using sglang's parser."""
        return _parse_tool_calls(response_text, self._tools_schema)

    @property
    def prompt_postprocess_fn(self) -> Optional[Callable[[str], str]]:
        """Tau2-bench prompt postprocessing (single-tool instruction)."""
        return _tau2_prompt_postprocess
