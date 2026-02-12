"""Orchestrator for Agent <-> Environment coding loops.

Unlike UserSimOrchestrator (agent <-> user_sim conversation), this orchestrator
manages agent <-> environment loops where every agent response contains a bash
command, and the environment executes it and returns an observation.

Used for CooperBench coding tasks where the agent implements features
in a sandbox environment.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from usim.core.model_adapter import ModelAdapter
from usim.core.types import (
    AgentState,
    Message,
    Trajectory,
    TrajectoryStatus,
    TrainableRole,
    UserSimConfig,
)

logger = logging.getLogger(__name__)

# Regex patterns for parsing agent responses
BASH_BLOCK_RE = re.compile(r"```bash\s*\n(.*?)\n```", re.DOTALL)
SEND_MESSAGE_RE = re.compile(r'send_message\s+(\w+)\s+"([^"]*)"')
STOP_SIGNAL = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"


class CodingAgentOrchestrator:
    """Orchestrator for Agent <-> Environment coding sessions.

    Key differences from UserSimOrchestrator:
    - No user simulator — only agent + environment (bash sandbox)
    - Every agent response contains a bash command (parsed via regex)
    - Environment executes command, returns observation as "user" message
    - Incoming partner messages injected as additional context
    - send_message calls parsed and delivered via messaging connector
    - Stop condition: COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT in command text

    Token Tracking (spare convention):
        - all_tokens: ALL tokens (prompt + all responses)
        - all_masks: loss masks for response tokens only (excludes prompt)
        - all_logprobs: log probs for response tokens only
        - Agent (assistant) tokens get mask=1, environment (user) tokens get mask=0
    """

    def __init__(
        self,
        agent_model: ModelAdapter,
        config: UserSimConfig,
        environment: Any,
        messaging: Optional[Any] = None,
    ):
        """Initialize the coding orchestrator.

        Args:
            agent_model: ModelAdapter for agent generation (trainable, SGLang)
            config: Configuration (max_turns used as max_steps)
            environment: CooperBenchEnvironment with execute_command()
            messaging: Optional MessagingConnector for inter-agent communication
        """
        self.agent_model = agent_model
        self.config = config
        self.env = environment
        self.messaging = messaging

    async def run_session(
        self,
        task: Dict[str, Any],
        agent: Any,
    ) -> Trajectory:
        """Run a complete agent <-> environment coding session.

        Args:
            task: Task specification with instructions, feature description, etc.
            agent: Agent instance implementing BaseAgent protocol

        Returns:
            Trajectory with all tokens, loss masks, and logprobs for training
        """
        # Initialize agent state
        agent_state = agent.get_init_state()

        # Token tracking (spare convention: masks/logprobs exclude prompt)
        all_tokens: List[int] = []
        all_masks: List[int] = []
        all_logprobs: List[float] = []
        messages: List[Dict[str, Any]] = []

        # Get initial prompt tokens from agent's system + task messages
        initial_messages = [msg.to_dict() for msg in agent_state.system_messages]
        prompt_tokens = self._tokenize_messages(initial_messages, self.agent_model)
        all_tokens.extend(prompt_tokens)

        # Add task description as first user message
        task_content = task.get("instructions", "")
        task_msg = Message(role="user", content=task_content)
        agent_state = agent_state.add_message(task_msg)
        messages.append(task_msg.to_dict())

        # Track task message tokens (not trainable)
        task_delta, task_mask = self._get_token_delta(messages, self.agent_model, "user")
        all_tokens.extend(task_delta)
        all_masks.extend(task_mask)
        all_logprobs.extend([0.0] * len(task_delta))

        # Main coding loop
        step_count = 0
        max_steps = self.config.max_turns  # Reuse max_turns as max_steps
        status = TrajectoryStatus.TRUNCATED

        for step in range(max_steps):
            step_count = step + 1

            # === AGENT TURN: generate response ===
            agent_msgs = agent.build_messages(agent_state)
            results = await self.agent_model.generate_async(
                messages=agent_msgs,
                input_ids=all_tokens[:],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

            if not results or "error" in results[0]:
                error_msg = results[0].get("error", "Unknown") if results else "No response"
                logger.error(f"Agent generation failed at step {step_count}: {error_msg}")
                status = TrajectoryStatus.FAILED
                break

            response_text = results[0]["text"]
            agent_msg, agent_state = agent.parse_response(response_text, agent_state)
            messages.append(agent_msg.to_dict())

            # Track agent tokens (mask=1, trainable)
            agent_delta, agent_mask = self._get_token_delta(
                messages, self.agent_model, "assistant"
            )
            agent_logprobs = self._extract_logprobs(results, len(agent_delta))
            all_tokens.extend(agent_delta)
            all_masks.extend(agent_mask)
            all_logprobs.extend(agent_logprobs)

            # Parse bash command from response
            bash_match = BASH_BLOCK_RE.search(response_text)
            if not bash_match:
                # Format error — no bash block found
                error_content = (
                    "Please always provide EXACTLY ONE action in triple backticks.\n"
                    "If you want to end the task, please issue the following command: "
                    "`echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`\n"
                )
                error_msg = Message(role="user", content=error_content)
                agent_state = agent_state.add_message(error_msg)
                messages.append(error_msg.to_dict())

                error_delta, error_mask = self._get_token_delta(
                    messages, self.agent_model, "user"
                )
                all_tokens.extend(error_delta)
                all_masks.extend(error_mask)
                all_logprobs.extend([0.0] * len(error_delta))
                continue

            command = bash_match.group(1).strip()

            # Parse and deliver send_message calls
            if self.messaging:
                for match in SEND_MESSAGE_RE.finditer(command):
                    recipient, content = match.groups()
                    self.messaging.send(recipient, content)
                    logger.debug(f"Sent message to {recipient}: {content[:50]}...")
                # Strip send_message calls from command before execution
                command = SEND_MESSAGE_RE.sub("", command).strip()

            # Check for stop signal
            if STOP_SIGNAL in command:
                status = TrajectoryStatus.COMPLETED
                break

            # === ENVIRONMENT TURN: execute command ===
            exec_result = await self.env.execute_command(command)

            # Check for incoming messages from partner
            incoming_msgs = []
            if self.messaging:
                incoming_msgs = self.messaging.receive()

            # Format observation
            observation = _format_observation(
                exec_result["output"],
                exec_result["returncode"],
                incoming_msgs,
            )

            obs_msg = Message(role="user", content=observation)
            agent_state = agent_state.add_message(obs_msg)
            messages.append(obs_msg.to_dict())

            # Track environment tokens (mask=0, not trainable)
            obs_delta, obs_mask = self._get_token_delta(
                messages, self.agent_model, "user"
            )
            all_tokens.extend(obs_delta)
            all_masks.extend(obs_mask)
            all_logprobs.extend([0.0] * len(obs_delta))

        # Signal session end
        agent.stop(
            Message.from_dict(messages[-1]) if messages else None,
            agent_state,
        )

        return self._build_trajectory(
            all_tokens, all_masks, all_logprobs, messages, step_count, task, status,
        )

    # === Helper methods (shared patterns from UserSimOrchestrator) ===

    def _compute_loss_mask_for_role(self, role: str) -> int:
        """Determine if tokens from this role should be trained."""
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
        """Calculate token delta for the last message added."""
        if not messages:
            return [], []

        tokenizer = model.tokenizer
        curr_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        prev_messages = messages[:-1]

        if messages[-1].get("role") == "assistant":
            prev_text = tokenizer.apply_chat_template(
                prev_messages, tokenize=False, add_generation_prompt=True,
            ) if prev_messages else ""
        else:
            prev_text = tokenizer.apply_chat_template(
                prev_messages, tokenize=False, add_generation_prompt=False,
            ) if prev_messages else ""

        delta_text = curr_text[len(prev_text):]
        new_tokens = tokenizer.encode(delta_text, add_special_tokens=False)
        mask_value = self._compute_loss_mask_for_role(role)
        loss_mask = [mask_value] * len(new_tokens)

        return new_tokens, loss_mask

    def _extract_logprobs(
        self, results: List[Dict[str, Any]], expected_length: int,
    ) -> List[float]:
        """Extract logprobs from generation results, padded to expected length."""
        if not results or "error" in results[0]:
            return [0.0] * expected_length

        logprobs = results[0].get("logprobs", [])
        if len(logprobs) == expected_length:
            return list(logprobs)
        elif len(logprobs) < expected_length:
            return list(logprobs) + [0.0] * (expected_length - len(logprobs))
        else:
            return list(logprobs[:expected_length])

    def _tokenize_messages(
        self, messages: List[Dict[str, Any]], model: ModelAdapter,
    ) -> List[int]:
        """Tokenize a list of messages."""
        if not messages:
            return []
        text = model.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        return model.tokenizer.encode(text, add_special_tokens=False)

    def _build_trajectory(
        self,
        tokens: List[int],
        loss_mask: List[int],
        logprobs: List[float],
        messages: List[Dict[str, Any]],
        step_count: int,
        task: Dict[str, Any],
        status: TrajectoryStatus = TrajectoryStatus.COMPLETED,
    ) -> Trajectory:
        """Build final trajectory from collected data."""
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
            reward=0.0,  # Reward computed externally (merge test result)
            status=status,
            messages=messages,
            turn_count=step_count,
            metadata={
                "task_id": task.get("id", ""),
                "repo": task.get("repo", ""),
                "feature_id": task.get("feature_id", ""),
                "trainable_role": self.config.trainable_role.value,
            },
        )


def _format_observation(
    output: str,
    returncode: int,
    incoming_messages: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Format command output + incoming messages as observation.

    Follows CooperBench's action_observation_template format.
    """
    max_output_len = 10000

    parts = [f"<returncode>{returncode}</returncode>"]

    if len(output) < max_output_len:
        parts.append(f"<output>\n{output}\n</output>")
    else:
        elided = len(output) - max_output_len
        parts.append(
            "<warning>\n"
            "The output of your last command was too long.\n"
            "Please try a different command that produces less output.\n"
            "</warning>"
        )
        parts.append(f"<output_head>\n{output[:5000]}\n</output_head>")
        parts.append(f"<elided_chars>\n{elided} characters elided\n</elided_chars>")
        parts.append(f"<output_tail>\n{output[-5000:]}\n</output_tail>")

    # Append incoming messages from partner agent(s)
    if incoming_messages:
        for msg in incoming_messages:
            sender = msg.get("from", "unknown")
            content = msg.get("content", "")
            parts.append(f"\n[Message from {sender}]: {content}")

    return "\n".join(parts)
