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


def compute_token_delta(
    tokenizer: Any,
    messages: List[Dict[str, Any]],
) -> Tuple[List[int], List[int]]:
    """Compute incremental tokens from adding last message to conversation.

    This function implements token-in-token-out tracking.
    It calculates the token delta for the most recent message, enabling
    synchronization between generation and training.

    Args:
        tokenizer: HuggingFace tokenizer with apply_chat_template method
        messages: Current conversation messages including the new message

    Returns:
        Tuple of (new_tokens, loss_mask) for the last message only
    """
    curr = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    # Get previous state (before last message)
    prev_messages = messages[:-1]

    # Determine if we need generation prompt for previous state
    if messages[-1]["role"] == "assistant":
        prev = tokenizer.apply_chat_template(
            prev_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prev = tokenizer.apply_chat_template(
            prev_messages,
            tokenize=False,
            add_generation_prompt=False,
        )

    # Extract delta (text added by new message)
    delta_text = curr[len(prev):]
    new_tokens = tokenizer.encode(delta_text, add_special_tokens=False)

    # Set loss mask based on role: 1 for assistant tokens, 0 for others
    if messages[-1]["role"] == "assistant":
        loss_mask = [1] * len(new_tokens)  # Train on assistant tokens
    else:
        loss_mask = [0] * len(new_tokens)  # Don't train on user/tool/env tokens

    return new_tokens, loss_mask


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
