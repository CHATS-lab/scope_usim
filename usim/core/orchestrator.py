"""Turn-based orchestrator for Agent-User Simulator sessions.

This module implements the main orchestration logic for two-agent rollouts,
following the tau2-bench communication protocol.

The orchestrator calls model.generate_async() directly instead of delegating
to agent/user_sim generate methods. Agent and user_sim provide prompt building
(build_messages) and response parsing (parse_response) while the orchestrator
controls generation and token tracking.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from usim.core.model_adapter import ModelAdapter
from usim.core.types import (
    AgentState,
    Message,
    Trajectory,
    TrajectoryStatus,
    TrainableRole,
    UserSimConfig,
    UserState,
)


logger = logging.getLogger(__name__)


class UserSimOrchestrator:
    """Orchestrator for Agent <-> User Simulator sessions.

    Manages turn-based conversation between an agent and a user simulator,
    with optional environment for tool execution. Tracks tokens, loss masks,
    and logprobs for training.

    Communication Flow:
        1. Agent -> User: Agent sends text response to user
        2. User -> Agent: User simulator responds
        3. Agent -> Env: Agent makes tool call(s) to environment
        4. Env -> Agent: Environment returns results
        5. [Repeat until stop signal or max_turns]

    Token Tracking (spare convention):
        - all_tokens: ALL tokens (prompt + all responses)
        - all_masks: loss masks for response tokens only (excludes prompt)
        - all_logprobs: log probs for response tokens only (same length as all_masks)
        - Invariant: len(all_masks) == len(all_logprobs)
        - Invariant: len(all_tokens) >= len(all_masks)
        - base_offset = len(all_tokens) - len(all_masks) gives prompt length
    """

    def __init__(
        self,
        agent_model: ModelAdapter,
        user_model: ModelAdapter,
        config: UserSimConfig,
        environment: Optional[Any] = None,
    ):
        """Initialize the orchestrator.

        Args:
            agent_model: ModelAdapter for agent generation
            user_model: ModelAdapter for user simulator generation
            config: User simulator configuration
            environment: Optional environment for tool execution
        """
        self.agent_model = agent_model
        self.user_model = user_model
        self.config = config
        self.env = environment

    async def run_session(
        self,
        task: Dict[str, Any],
        agent: Any,
        user_simulator: Any,
    ) -> Trajectory:
        """Run a complete Agent <-> User Simulator session.

        This is the main entry point for rollout. It orchestrates the
        conversation, tracks tokens, and produces a training trajectory.

        The orchestrator calls model.generate_async() directly, using
        agent/user_sim only for prompt building and response parsing.

        Args:
            task: Task specification with instructions, initial state, etc.
            agent: Agent instance (must implement BaseAgent protocol)
            user_simulator: User simulator instance (must implement BaseUserSimulator)

        Returns:
            Trajectory containing all tokens, loss masks, and logprobs for training
        """
        # Initialize states
        agent_state = agent.get_init_state()
        user_state = user_simulator.get_init_state()

        # Token tracking (spare convention: masks/logprobs exclude prompt)
        all_tokens: List[int] = []
        all_masks: List[int] = []
        all_logprobs: List[float] = []
        messages: List[Dict[str, Any]] = []

        # Get initial prompt tokens from agent's system messages
        initial_messages = self._build_initial_messages(agent_state, user_state)
        prompt_tokens = self._tokenize_messages(initial_messages, self.agent_model)
        all_tokens.extend(prompt_tokens)
        # Note: no masks or logprobs for prompt tokens (spare convention)

        # === AGENT TURN: Generate initial agent greeting ===
        agent_msgs = agent.build_messages(agent_state)
        results = await self.agent_model.generate_async(
            messages=agent_msgs,
            input_ids=all_tokens[:],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        if not results or "error" in results[0]:
            error_msg = results[0].get("error", "Unknown error") if results else "No response"
            logger.error(f"Agent generation failed: {error_msg}")
            agent_msg = Message(role="assistant", content="###STOP###")
            agent_state = agent_state.add_message(agent_msg)
        else:
            response_text = results[0]["text"]
            agent_msg, agent_state = agent.parse_response(response_text, agent_state)

        messages.append(agent_msg.to_dict())

        # Track agent tokens
        agent_delta_tokens, agent_delta_mask = self._get_token_delta(
            messages, self.agent_model, "assistant"
        )
        agent_logprobs = self._extract_logprobs(results, len(agent_delta_tokens))
        all_tokens.extend(agent_delta_tokens)
        all_masks.extend(agent_delta_mask)
        all_logprobs.extend(agent_logprobs)

        # Check for immediate stop
        if agent.is_stop(agent_msg):
            return self._build_trajectory(
                all_tokens, all_masks, all_logprobs, messages, 0, task,
                TrajectoryStatus.COMPLETED,
            )

        # Main conversation loop
        turn_count = 0
        status = TrajectoryStatus.TRUNCATED  # Default if we hit max_turns

        for turn in range(self.config.max_turns):
            turn_count = turn + 1

            # Handle agent message destination
            if agent_msg.is_tool_call() and self.env is not None:
                # === AGENT -> ENVIRONMENT ===
                tool_msgs = await self._execute_tool_calls(agent_msg)
                for tool_msg in tool_msgs:
                    messages.append(tool_msg.to_dict())
                    # Track environment tokens (never trained, no logprobs)
                    env_tokens, env_mask = self._get_token_delta(
                        messages, self.agent_model, "tool"
                    )
                    all_tokens.extend(env_tokens)
                    all_masks.extend(env_mask)
                    all_logprobs.extend([0.0] * len(env_tokens))

                # Update agent state with tool messages
                for tool_msg in tool_msgs:
                    agent_state = agent_state.add_message(tool_msg)

                # === ENVIRONMENT -> AGENT: generate next agent response ===
                agent_msgs = agent.build_messages(agent_state)
                results = await self.agent_model.generate_async(
                    messages=agent_msgs,
                    input_ids=all_tokens[:],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )

                if not results or "error" in results[0]:
                    agent_msg = Message(role="assistant", content="###STOP###")
                    agent_state = agent_state.add_message(agent_msg)
                else:
                    response_text = results[0]["text"]
                    agent_msg, agent_state = agent.parse_response(
                        response_text, agent_state
                    )

                messages.append(agent_msg.to_dict())

                # Track agent tokens
                agent_delta_tokens, agent_delta_mask = self._get_token_delta(
                    messages, self.agent_model, "assistant"
                )
                agent_logprobs = self._extract_logprobs(results, len(agent_delta_tokens))
                all_tokens.extend(agent_delta_tokens)
                all_masks.extend(agent_delta_mask)
                all_logprobs.extend(agent_logprobs)

                if agent.is_stop(agent_msg):
                    status = TrajectoryStatus.COMPLETED
                    break

            else:
                # === AGENT -> USER: get user simulator response ===
                # Add the agent message to user state
                user_state = user_state.add_message(agent_msg)

                # Build user sim messages (with flipped roles)
                user_msgs = user_simulator.build_messages(user_state)

                # Generate user response
                user_results = await self.user_model.generate_async(
                    messages=user_msgs,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )

                if not user_results or "error" in user_results[0]:
                    user_msg = Message(role="user", content="###STOP###")
                    user_state = user_state.add_message(user_msg)
                else:
                    response_text = user_results[0]["text"]
                    user_msg, user_state = user_simulator.parse_response(
                        response_text, user_state
                    )

                messages.append(user_msg.to_dict())

                # Track user tokens using compute_token_delta
                user_delta_tokens, user_delta_mask = self._get_token_delta(
                    messages, self.user_model, "user"
                )
                user_logprobs = self._extract_logprobs(user_results, len(user_delta_tokens))
                all_tokens.extend(user_delta_tokens)
                all_masks.extend(user_delta_mask)
                all_logprobs.extend(user_logprobs)

                if user_simulator.is_stop(user_msg):
                    status = TrajectoryStatus.COMPLETED
                    break

                # Handle user tool calls
                if user_msg.is_tool_call() and self.env is not None:
                    tool_msgs = await self._execute_tool_calls(user_msg)
                    for tool_msg in tool_msgs:
                        messages.append(tool_msg.to_dict())
                        env_tokens, env_mask = self._get_token_delta(
                            messages, self.user_model, "tool"
                        )
                        all_tokens.extend(env_tokens)
                        all_masks.extend(env_mask)
                        all_logprobs.extend([0.0] * len(env_tokens))

                    # Update user state with tool messages
                    for tool_msg in tool_msgs:
                        user_state = user_state.add_message(tool_msg)

                # === USER -> AGENT: generate next agent response ===
                # Add user message to agent state
                agent_state = agent_state.add_message(user_msg)

                agent_msgs = agent.build_messages(agent_state)
                results = await self.agent_model.generate_async(
                    messages=agent_msgs,
                    input_ids=all_tokens[:],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )

                if not results or "error" in results[0]:
                    agent_msg = Message(role="assistant", content="###STOP###")
                    agent_state = agent_state.add_message(agent_msg)
                else:
                    response_text = results[0]["text"]
                    agent_msg, agent_state = agent.parse_response(
                        response_text, agent_state
                    )

                messages.append(agent_msg.to_dict())

                # Track agent tokens
                agent_delta_tokens, agent_delta_mask = self._get_token_delta(
                    messages, self.agent_model, "assistant"
                )
                agent_logprobs = self._extract_logprobs(results, len(agent_delta_tokens))
                all_tokens.extend(agent_delta_tokens)
                all_masks.extend(agent_delta_mask)
                all_logprobs.extend(agent_logprobs)

                if agent.is_stop(agent_msg):
                    status = TrajectoryStatus.COMPLETED
                    break

        # Signal session end to agent and user
        agent.stop(agent_msg if messages else None, agent_state)
        user_simulator.stop(
            Message.from_dict(messages[-1]) if messages else None,
            user_state,
        )

        return self._build_trajectory(
            all_tokens, all_masks, all_logprobs, messages, turn_count, task, status,
        )

    def _compute_loss_mask_for_role(self, role: str) -> int:
        """Determine if tokens from this role should be trained.

        Args:
            role: The message role ("assistant", "user", "tool", "system")

        Returns:
            1 if this role should be trained, 0 otherwise
        """
        if role == "tool" or role == "system":
            return 0

        trainable = self.config.trainable_role

        if trainable == TrainableRole.BOTH:
            return 1 if role in ("assistant", "user") else 0
        elif trainable == TrainableRole.AGENT:
            return 1 if role == "assistant" else 0
        elif trainable == TrainableRole.USER:
            return 1 if role == "user" else 0

        return 0

    def _get_token_delta(
        self,
        messages: List[Dict[str, Any]],
        model: ModelAdapter,
        role: str,
    ) -> Tuple[List[int], List[int]]:
        """Calculate token delta for the last message added.

        Uses incremental tokenization to track tokens added by each message.

        Args:
            messages: Current conversation (including new message)
            model: ModelAdapter with tokenizer
            role: Role of the new message

        Returns:
            Tuple of (new_token_ids, loss_mask)
        """
        if not messages:
            return [], []

        tokenizer = model.tokenizer

        # Current state with new message
        curr_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        # Previous state (before last message)
        prev_messages = messages[:-1]

        if messages[-1].get("role") == "assistant":
            # For assistant messages, previous had generation prompt
            prev_text = tokenizer.apply_chat_template(
                prev_messages,
                tokenize=False,
                add_generation_prompt=True,
            ) if prev_messages else ""
        else:
            prev_text = tokenizer.apply_chat_template(
                prev_messages,
                tokenize=False,
                add_generation_prompt=False,
            ) if prev_messages else ""

        # Get delta text and tokenize
        delta_text = curr_text[len(prev_text):]
        new_tokens = tokenizer.encode(delta_text, add_special_tokens=False)

        # Generate loss mask based on role and config
        mask_value = self._compute_loss_mask_for_role(role)
        loss_mask = [mask_value] * len(new_tokens)

        return new_tokens, loss_mask

    def _extract_logprobs(
        self,
        results: List[Dict[str, Any]],
        expected_length: int,
    ) -> List[float]:
        """Extract logprobs from generation results, padded to expected length.

        Args:
            results: Generation results from model
            expected_length: Expected number of logprob values

        Returns:
            List of logprob values, padded with 0.0 if needed
        """
        if not results or "error" in results[0]:
            return [0.0] * expected_length

        logprobs = results[0].get("logprobs", [])

        if len(logprobs) == expected_length:
            return list(logprobs)
        elif len(logprobs) < expected_length:
            # Pad with 0.0
            return list(logprobs) + [0.0] * (expected_length - len(logprobs))
        else:
            # Truncate
            return list(logprobs[:expected_length])

    def _tokenize_messages(
        self,
        messages: List[Dict[str, Any]],
        model: ModelAdapter,
    ) -> List[int]:
        """Tokenize a list of messages.

        Args:
            messages: Messages to tokenize
            model: ModelAdapter with tokenizer

        Returns:
            Token IDs
        """
        if not messages:
            return []

        text = model.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        return model.tokenizer.encode(text, add_special_tokens=False)

    def _build_initial_messages(
        self,
        agent_state: AgentState,
        user_state: UserState,
    ) -> List[Dict[str, Any]]:
        """Build initial messages from agent and user system prompts.

        Args:
            agent_state: Agent's initial state
            user_state: User's initial state

        Returns:
            List of initial system messages
        """
        messages = []

        # Add agent system messages
        for msg in agent_state.system_messages:
            messages.append(msg.to_dict())

        return messages

    async def _execute_tool_calls(self, message: Message) -> List[Message]:
        """Execute tool calls from a message.

        Args:
            message: Message containing tool calls

        Returns:
            List of tool response messages
        """
        if self.env is None or not message.tool_calls:
            return []

        tool_responses = []
        for tool_call in message.tool_calls:
            response = await self.env.execute_tool_async(tool_call)
            tool_responses.append(response)

        return tool_responses

    def _build_trajectory(
        self,
        tokens: List[int],
        loss_mask: List[int],
        logprobs: List[float],
        messages: List[Dict[str, Any]],
        turn_count: int,
        task: Dict[str, Any],
        status: TrajectoryStatus = TrajectoryStatus.COMPLETED,
    ) -> Trajectory:
        """Build final trajectory from collected data.

        Args:
            tokens: All token IDs (prompt + response)
            loss_mask: Per-response-token loss mask (excludes prompt)
            logprobs: Per-response-token logprobs (same length as loss_mask)
            messages: Message history
            turn_count: Number of turns completed
            task: Original task specification
            status: Trajectory status

        Returns:
            Complete Trajectory for training
        """
        # Build response text from messages
        response_parts = []
        for msg in messages:
            if msg.get("role") == "assistant" and msg.get("content"):
                response_parts.append(msg["content"])

        return Trajectory(
            index=0,
            prompt=task.get("instructions", ""),
            tokens=tokens,
            loss_mask=loss_mask,
            rollout_log_probs=logprobs,
            response="\n".join(response_parts),
            response_length=sum(loss_mask),
            reward=0.0,  # Reward computed externally
            status=status,
            messages=messages,
            turn_count=turn_count,
            metadata={
                "task_id": task.get("id", ""),
                "domain": task.get("domain", ""),
                "trainable_role": self.config.trainable_role.value,
            },
        )


def run_session_sync(
    orchestrator: UserSimOrchestrator,
    task: Dict[str, Any],
    agent: Any,
    user_simulator: Any,
) -> Trajectory:
    """Synchronous wrapper for run_session.

    Args:
        orchestrator: UserSimOrchestrator instance
        task: Task specification
        agent: Agent instance
        user_simulator: User simulator instance

    Returns:
        Trajectory from the session
    """
    import asyncio

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(
        orchestrator.run_session(task, agent, user_simulator)
    )
