"""Core types for user simulator training.

This module defines the data structures used throughout the usim package.
These types are framework-agnostic and can be used with any RL backend.
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple


class TrajectoryStatus(str, Enum):
    """Status of a trajectory episode.

    Similar to Slime's Sample.Status and tau2-bench's InteractionResult status.
    """

    PENDING = "pending"  # Created but not executed
    RUNNING = "running"  # Currently executing
    COMPLETED = "completed"  # Completed successfully
    TRUNCATED = "truncated"  # Hit max turns/length limit
    TIMEOUT = "timeout"  # Took too long
    FAILED = "failed"  # Failed with recoverable error
    ABORTED = "aborted"  # Aborted (critical failure)


class TrainableRole(Enum):
    """Specifies which role(s) produce training trajectories.

    - AGENT: Only agent responses have loss_mask=1
    - USER: Only user simulator responses have loss_mask=1
    - BOTH: Both agent and user simulator responses have loss_mask=1
    """
    AGENT = "agent"
    USER = "user"
    BOTH = "both"


class TrainingMode(str, Enum):
    """Multi-agent training mode.

    - SINGLE: 1 trainable model + fixed API opponent (existing behavior)
    - COTRAIN: 2 trainable models, each with SGLang engines + training workers
    - SELFPLAY: 2 trainable models + opponent checkpoint pool for diversity
    """
    SINGLE = "single"
    COTRAIN = "cotrain"
    SELFPLAY = "selfplay"


@dataclass
class ToolCall:
    """Represents a tool/function call made by agent or user.

    Attributes:
        id: Unique identifier for the tool call
        name: Name of the tool/function being called
        arguments: Arguments to pass to the tool
        requestor: Who made the tool call ("user" or "assistant")
    """
    id: str
    name: str
    arguments: Dict[str, Any]
    requestor: Literal["user", "assistant"] = "assistant"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "id": self.id,
            "name": self.name,
            "arguments": self.arguments,
            "requestor": self.requestor,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolCall":
        """Create from dictionary."""
        return cls(
            id=data.get("id", ""),
            name=data["name"],
            arguments=data["arguments"],
            requestor=data.get("requestor", "assistant"),
        )


@dataclass
class Message:
    """Represents a message in the conversation.

    A message can contain either text content OR tool calls, but not both.
    This follows the OpenAI message format convention.

    Attributes:
        role: One of "user", "assistant", "system", or "tool"
        content: Text content of the message (optional if tool_calls present)
        tool_calls: List of tool calls (optional if content present)
        tool_call_id: ID of the tool call this message responds to (for tool messages)
        timestamp: ISO timestamp of the message
        cost: Cost of generating this message (for LLM calls)
        usage: Token usage info (for LLM calls)
    """
    role: Literal["user", "assistant", "system", "tool"]
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None
    timestamp: Optional[str] = None
    cost: Optional[float] = None
    usage: Optional[Dict[str, int]] = None

    def is_tool_call(self) -> bool:
        """Check if this message contains tool calls."""
        return self.tool_calls is not None and len(self.tool_calls) > 0

    def has_text_content(self) -> bool:
        """Check if this message has non-empty text content."""
        return self.content is not None and self.content.strip() != ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to OpenAI-style message dict."""
        result: Dict[str, Any] = {"role": self.role}
        if self.content is not None:
            result["content"] = self.content
        if self.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.loads(tc.arguments) if isinstance(tc.arguments, str) else tc.arguments,
                    },
                }
                for tc in self.tool_calls
            ]
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create from OpenAI-style message dict."""
        tool_calls = None
        if "tool_calls" in data:
            tool_calls = [
                ToolCall(
                    id=tc.get("id", ""),
                    name=tc["function"]["name"],
                    arguments=tc["function"]["arguments"],
                )
                for tc in data["tool_calls"]
            ]
        return cls(
            role=data["role"],
            content=data.get("content"),
            tool_calls=tool_calls,
            tool_call_id=data.get("tool_call_id"),
        )


@dataclass
class UserState:
    """State maintained by the user simulator during a session.

    Attributes:
        system_messages: System messages defining user behavior
        messages: Conversation history from user's perspective
        metadata: Additional state information
    """
    system_messages: List[Message] = field(default_factory=list)
    messages: List[Message] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_message(self, message: Message) -> "UserState":
        """Return new state with message added."""
        return UserState(
            system_messages=self.system_messages,
            messages=self.messages + [message],
            metadata=self.metadata.copy(),
        )

    def flip_roles(self) -> List[Message]:
        """Flip roles for user simulator perspective.

        From user simulator's view, the agent is "user" and user is "assistant".
        This is needed because the user simulator generates "assistant" responses
        but they should be treated as "user" messages in the actual conversation.
        """
        flipped = []
        for msg in self.messages:
            new_role = msg.role
            if msg.role == "assistant":
                new_role = "user"
            elif msg.role == "user":
                new_role = "assistant"
            flipped.append(Message(
                role=new_role,
                content=msg.content,
                tool_calls=msg.tool_calls,
                tool_call_id=msg.tool_call_id,
            ))
        return flipped


@dataclass
class AgentState:
    """State maintained by the agent during a session.

    Attributes:
        system_messages: System messages defining agent behavior
        messages: Conversation history from agent's perspective
        metadata: Additional state information
    """
    system_messages: List[Message] = field(default_factory=list)
    messages: List[Message] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_message(self, message: Message) -> "AgentState":
        """Return new state with message added."""
        return AgentState(
            system_messages=self.system_messages,
            messages=self.messages + [message],
            metadata=self.metadata.copy(),
        )


@dataclass
class UserSimConfig:
    """Configuration for user simulator training.

    Attributes:
        temperature: Sampling temperature for generation
        max_tokens: Maximum tokens per generation
        max_turns: Maximum conversation turns
        max_context_length: Maximum context length in characters
        trainable_role: Which role(s) to train
        stop_tokens: Tokens that signal conversation end
    """
    temperature: float = 0.7
    max_tokens: int = 2048
    max_turns: int = 30
    max_context_length: int = 16384
    trainable_role: TrainableRole = TrainableRole.USER
    stop_tokens: List[str] = field(default_factory=lambda: ["###STOP###", "###TRANSFER###"])


def get_token_delta(
    tokenizer: Any,
    messages: List[Dict[str, Any]],
    tools_schema: Optional[List[Dict[str, Any]]] = None,
    postprocess: Optional[Any] = None,
) -> Tuple[List[int], List[int]]:
    """Compute incremental tokens contributed by the last message.

    Canonical multi-turn token delta helper. This is the SINGLE source of truth
    for per-turn delta computation across orchestrators. Do not duplicate this
    logic elsewhere.

    The critical invariant: when the last message is a user/tool/system message,
    the returned delta MUST include the ``<|im_start|>assistant\\n`` generation
    prompt that precedes the next assistant turn. Otherwise the next call to the
    inference server is fed a prompt with no assistant cue and the model goes
    out of distribution, hallucinating a fake role prefix that then gets
    appended to the training buffer with ``loss_mask=1`` and reinforced every
    RL step.

    Correct toggling:

    - Last msg is ``assistant``: ``curr`` has no generation prompt, ``prev``
      has one (because the inference server already consumed the assistant
      prompt when generating).
    - Last msg is ``user``/``tool``/``system``: ``curr`` has the generation
      prompt (so the next generate call has the right input), ``prev`` does
      not.

    Args:
        tokenizer: HuggingFace tokenizer with ``apply_chat_template``.
        messages: Current conversation messages INCLUDING the new message.
        tools_schema: Optional tool schemas passed through to the chat template.
        postprocess: Optional text transform applied to both ``curr`` and
            ``prev`` before computing the delta. Used by tau2-bench to
            reformulate the default multi-tool instruction.

    Returns:
        Tuple of ``(new_tokens, loss_mask)`` for the last message only.
        ``loss_mask`` is 1 for assistant content and 0 for all other roles
        (including the assistant generation prompt glue that precedes a
        user-turn's reply, which is template structure, not model output).
    """
    is_assistant = messages[-1].get("role") == "assistant"

    template_kwargs: Dict[str, Any] = {}
    if tools_schema:
        template_kwargs["tools"] = tools_schema

    curr = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=not is_assistant,
        **template_kwargs,
    )
    prev_messages = messages[:-1]
    if prev_messages:
        prev = tokenizer.apply_chat_template(
            prev_messages,
            tokenize=False,
            add_generation_prompt=is_assistant,
            **template_kwargs,
        )
    else:
        prev = ""

    if postprocess is not None:
        curr = postprocess(curr)
        prev = postprocess(prev)

    delta_text = curr[len(prev):]
    new_tokens = tokenizer.encode(delta_text, add_special_tokens=False)
    loss_mask = [1 if is_assistant else 0] * len(new_tokens)
    return new_tokens, loss_mask


def probe_inter_message_glue(tokenizer: Any) -> List[int]:
    """Probe the tokens the chat template emits AFTER an assistant EOS.

    Most chat templates (Qwen, Llama 3, Mistral, Gemma, etc.) insert a fixed
    suffix after each assistant message's EOS token, typically a single
    newline ``\\n`` (token 198 on Qwen). Inference servers like SGLang stop
    generation at the EOS token and do NOT return this trailing glue, which
    creates off-by-one drift in the accumulated token buffer unless we append
    the glue ourselves after each completed assistant turn.

    Call this ONCE at orchestrator init and cache the result. The probe does
    not depend on the conversation content so this is O(1) per episode.

    Args:
        tokenizer: HuggingFace tokenizer with ``apply_chat_template``.

    Returns:
        List of token IDs to append after each assistant response that ended
        naturally at EOS. Empty list if no glue is detected (should be rare).
    """
    probe = tokenizer.apply_chat_template(
        [{"role": "user", "content": "x"},
         {"role": "assistant", "content": "y"}],
        tokenize=True,
        add_generation_prompt=False,
    )
    if not isinstance(probe, list):
        probe = list(probe)
    eos = tokenizer.eos_token_id
    if eos is None:
        return []
    # Find the LAST EOS token and return everything after it.
    for i in range(len(probe) - 1, -1, -1):
        if probe[i] == eos:
            return probe[i + 1:]
    return []


# Backward-compatible alias. New code should import ``get_token_delta``.
# Note: the legacy signature did not accept tools_schema/postprocess; callers
# that pass only (tokenizer, messages) continue to work unchanged.
def compute_token_delta(
    tokenizer: Any,
    messages: List[Dict[str, Any]],
) -> Tuple[List[int], List[int]]:
    """Deprecated alias for ``get_token_delta``.

    Kept only for backward compatibility with test_types.py. All active rollout
    code should import and call ``get_token_delta`` directly.
    """
    return get_token_delta(tokenizer, messages)


@dataclass
class Trajectory:
    """Complete trajectory from a session for training.

    This is the main output of the orchestrator, containing all information
    needed for RL training. Compatible with both Slime and Tinker backends.

    The loss_mask and rollout_log_probs exclude prompt tokens (match spare convention).
    The offset ``base_offset = len(tokens) - len(loss_mask)`` gives the prompt length.

    Attributes:
        index: Unique index for batching
        prompt: Initial prompt text
        tokens: All token IDs (prompt + all responses)
        loss_mask: Per-response-token mask (1=train, 0=skip), excludes prompt tokens
        rollout_log_probs: Log probs for response tokens, same length as loss_mask
        response: Concatenated response text
        response_length: Number of response tokens (sum of loss_mask)
        reward: Final reward from the session
        status: Trajectory execution status
        messages: Full message history
        turn_count: Number of turns in the session
        metadata: Additional trajectory info (task_id, domain, etc.)
    """

    # Core identification
    index: int = 0

    # Prompt data
    prompt: str = ""
    messages: List[Dict[str, Any]] = field(default_factory=list)

    # Token data (accumulated during rollout)
    tokens: List[int] = field(default_factory=list)
    loss_mask: List[int] = field(default_factory=list)

    # Response data
    response: str = ""
    response_length: int = 0

    # Training data
    rollout_log_probs: List[float] = field(default_factory=list)

    # Reward data
    reward: float = 0.0

    # Status
    status: TrajectoryStatus = TrajectoryStatus.PENDING

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    turn_count: int = 0
