"""Orchestrator for Agent <-> Environment coding loops using tool calling.

Unlike UserSimOrchestrator (agent <-> user_sim conversation), this orchestrator
manages agent <-> environment loops following mini-swe-agent-v2 protocol:
every agent response contains structured tool calls (bash + optional send_message),
and the environment executes them and returns JSON-formatted observations.

Used for CooperBench coding tasks where the agent implements features
in a sandbox environment.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.managers.io_struct import Function, Tool

from usim.core.model_adapter import ModelAdapter
from usim.core.types import (
    Message,
    ToolCall,
    Trajectory,
    TrajectoryStatus,
    TrainableRole,
    UserSimConfig,
)

logger = logging.getLogger(__name__)

STOP_SIGNAL = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"


def _parse_tool_calls(
    response_text: str,
    tools_schema: List[Dict[str, Any]],
    tool_call_parser: str = "glm",
) -> Dict[str, Any]:
    """Parse tool calls from agent response using sglang's FunctionCallParser.

    Supports any model format via tool_call_parser ("qwen", "hermes", "llama3", etc.).
    Returns {"normal_text": str, "calls": list} where each call is
    {"name": str, "arguments": dict, "id": str}.
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
        parser = FunctionCallParser(tools=tools_list, tool_call_parser=tool_call_parser)
        normal_text, calls = parser.parse_non_stream(response_text)

        normalized = []
        for call in calls:
            call_dict = call.model_dump()
            args = call_dict.get("parameters", call_dict.get("arguments", {}))
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, TypeError):
                    args = {}
            normalized.append({
                "name": call_dict.get("name", ""),
                "arguments": args,
                "id": call_dict.get("id", ""),
            })

        return {"normal_text": normal_text, "calls": normalized}
    except Exception as e:
        logger.warning(f"Tool call parsing failed: {e}")
        return {"normal_text": response_text, "calls": []}


def _format_tool_observation(
    output: str,
    returncode: int,
    incoming_messages: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Format command output as JSON tool result (v2 observation_template format)."""
    max_output_len = 10000

    if len(output) < max_output_len:
        obs: Dict[str, Any] = {"returncode": returncode, "output": output}
    else:
        obs = {
            "returncode": returncode,
            "output_head": output[:5000],
            "output_tail": output[-5000:],
            "elided_chars": len(output) - max_output_len,
            "warning": "Output too long.",
        }

    if incoming_messages:
        obs["partner_messages"] = [
            {"from": m.get("from", "partner"), "content": m.get("content", "")}
            for m in incoming_messages
        ]

    return json.dumps(obs)


class CodingAgentOrchestrator:
    """Orchestrator for Agent <-> Environment coding sessions using tool calling.

    Key differences from UserSimOrchestrator:
    - No user simulator — only agent + environment (bash sandbox)
    - Agent responses parsed for structured tool calls (Qwen3 <tool_call> format)
    - Environment executes bash commands, returns JSON observations as role="tool" messages
    - send_message tool calls delivered via messaging connector
    - Stop condition: COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT in bash command

    Tool call parsing uses sglang's FunctionCallParser with a configurable
    tool_call_parser name ("qwen", "hermes", "llama3", etc.) — model-agnostic.

    Token Tracking (spare convention):
        - all_tokens: ALL tokens (prompt + all responses)
        - all_masks: loss masks for response tokens only (excludes prompt)
        - all_logprobs: log probs for response tokens only
        - Agent (assistant) tokens get mask=1, environment (tool) tokens get mask=0
    """

    def __init__(
        self,
        agent_model: ModelAdapter,
        config: UserSimConfig,
        environment: Any,
        messaging: Optional[Any] = None,
        tool_call_parser: str = "glm",
    ):
        """Initialize the coding orchestrator.

        Args:
            agent_model: ModelAdapter for agent generation (trainable, SGLang)
            config: Configuration (max_turns used as max_steps)
            environment: CooperBenchEnvironment with execute_command() and get_tools()
            messaging: Optional MessagingConnector for inter-agent communication
            tool_call_parser: sglang parser name for the model's tool call format.
                "qwen" for Qwen3, "hermes" for Hermes/Mistral, "llama3" for Llama 3.1+.
        """
        self.agent_model = agent_model
        self.config = config
        self.env = environment
        self.messaging = messaging
        self.tool_call_parser = tool_call_parser
        self._tools_schema = environment.get_tools()

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
        agent_state = agent.get_init_state()

        # Token tracking (spare convention: masks/logprobs exclude prompt)
        all_tokens: List[int] = []
        all_masks: List[int] = []
        all_logprobs: List[float] = []
        messages: List[Dict[str, Any]] = []

        # Get initial prompt tokens from agent's system message (with tool definitions)
        initial_messages = [msg.to_dict() for msg in agent_state.system_messages]
        prompt_tokens = self._tokenize_messages(initial_messages)
        all_tokens.extend(prompt_tokens)

        # Add task description as first user message
        task_content = task.get("instructions", "")
        task_msg = Message(role="user", content=task_content)
        agent_state = agent_state.add_message(task_msg)
        messages.append(task_msg.to_dict())

        # Track task message tokens (not trainable)
        task_delta, task_mask = self._get_token_delta(messages, "user")
        all_tokens.extend(task_delta)
        all_masks.extend(task_mask)
        all_logprobs.extend([0.0] * len(task_delta))

        # Main coding loop
        step_count = 0
        max_steps = self.config.max_turns
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
            parsed = _parse_tool_calls(response_text, self._tools_schema, self.tool_call_parser)

            if not parsed["calls"]:
                # Format error — no tool call found; add plain assistant message + error
                asst_msg = Message(role="assistant", content=response_text)
                agent_state = agent_state.add_message(asst_msg)
                messages.append(asst_msg.to_dict())

                asst_delta, asst_mask = self._get_token_delta(messages, "assistant")
                agent_logprobs = self._extract_logprobs(results, len(asst_delta))
                all_tokens.extend(asst_delta)
                all_masks.extend(asst_mask)
                all_logprobs.extend(agent_logprobs)

                error_content = (
                    "Tool call error:\n\n<error>\n"
                    "No tool calls found in the response. Every response MUST include at least one tool call.\n"
                    "</error>\n\n"
                    "Here is general guidance on how to submit correct toolcalls:\n\n"
                    "Every response needs to use the 'bash' tool at least once to execute commands.\n\n"
                    "Call the bash tool with your command as the argument:\n"
                    "- Tool: bash\n"
                    '- Arguments: {"command": "your_command_here"}\n\n'
                    f"If you want to end the task, please issue the following command: `echo {STOP_SIGNAL}`\n"
                    "without any other command."
                )
                err_msg = Message(role="user", content=error_content)
                agent_state = agent_state.add_message(err_msg)
                messages.append(err_msg.to_dict())

                err_delta, err_mask = self._get_token_delta(messages, "user")
                all_tokens.extend(err_delta)
                all_masks.extend(err_mask)
                all_logprobs.extend([0.0] * len(err_delta))
                continue

            # Build assistant message with tool_calls
            tool_calls = [
                ToolCall(
                    id=call.get("id", f"call_{step}_{i}"),
                    name=call["name"],
                    arguments=call["arguments"],
                )
                for i, call in enumerate(parsed["calls"])
            ]
            normal_text = parsed["normal_text"] or None
            asst_msg = Message(role="assistant", content=normal_text, tool_calls=tool_calls)
            agent_state = agent_state.add_message(asst_msg)
            messages.append(asst_msg.to_dict())

            # Track agent tokens (mask=1, trainable)
            agent_delta, agent_mask = self._get_token_delta(messages, "assistant")
            agent_logprobs = self._extract_logprobs(results, len(agent_delta))
            all_tokens.extend(agent_delta)
            all_masks.extend(agent_mask)
            all_logprobs.extend(agent_logprobs)

            # === ENVIRONMENT TURN: execute tool calls ===
            stop = False
            for i, call in enumerate(parsed["calls"]):
                name = call["name"]
                args = call["arguments"]
                call_id = tool_calls[i].id

                if name == "bash":
                    command = args.get("command", "").strip() if isinstance(args, dict) else ""
                    if STOP_SIGNAL in command:
                        stop = True
                        tool_content = json.dumps({"returncode": 0, "output": STOP_SIGNAL})
                    else:
                        exec_result = await self.env.execute_command(command)
                        incoming_msgs = self.messaging.receive() if self.messaging else []
                        tool_content = _format_tool_observation(
                            exec_result["output"],
                            exec_result["returncode"],
                            incoming_msgs if incoming_msgs else None,
                        )

                elif name == "send_message":
                    recipient = args.get("recipient", "") if isinstance(args, dict) else ""
                    content = args.get("content", "") if isinstance(args, dict) else ""
                    if self.messaging:
                        self.messaging.send(recipient, content)
                        logger.debug(f"Sent message to {recipient}: {content[:50]}...")
                    tool_content = json.dumps({"status": "sent"})

                else:
                    tool_content = json.dumps({"error": f"Unknown tool: {name}"})

                tool_msg = Message(role="tool", tool_call_id=call_id, content=tool_content)
                agent_state = agent_state.add_message(tool_msg)
                messages.append(tool_msg.to_dict())

                # Track tool tokens (mask=0, not trainable)
                tool_delta, tool_mask_vals = self._get_token_delta(messages, "tool")
                all_tokens.extend(tool_delta)
                all_masks.extend(tool_mask_vals)
                all_logprobs.extend([0.0] * len(tool_delta))

                if stop:
                    break

            if stop:
                status = TrajectoryStatus.COMPLETED
                break

        # Signal session end
        agent.stop(
            Message.from_dict(messages[-1]) if messages else None,
            agent_state,
        )

        return self._build_trajectory(
            all_tokens, all_masks, all_logprobs, messages, step_count, task, status,
        )

    # === Helper methods ===

    def _compute_loss_mask_for_role(self, role: str) -> int:
        """Determine if tokens from this role should be trained."""
        if role in ("tool", "system"):
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
        role: str,
    ) -> Tuple[List[int], List[int]]:
        """Calculate token delta for the last message added."""
        if not messages:
            return [], []

        tokenizer = self.agent_model.tokenizer
        curr_text = tokenizer.apply_chat_template(
            messages,
            tools=self._tools_schema,
            tokenize=False,
            add_generation_prompt=False,
        )
        prev_messages = messages[:-1]

        if messages[-1].get("role") == "assistant":
            prev_text = (
                tokenizer.apply_chat_template(
                    prev_messages,
                    tools=self._tools_schema,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                if prev_messages
                else ""
            )
        else:
            prev_text = (
                tokenizer.apply_chat_template(
                    prev_messages,
                    tools=self._tools_schema,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                if prev_messages
                else ""
            )

        delta_text = curr_text[len(prev_text):]
        new_tokens = tokenizer.encode(delta_text, add_special_tokens=False)
        mask_value = self._compute_loss_mask_for_role(role)
        return new_tokens, [mask_value] * len(new_tokens)

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
        return list(logprobs[:expected_length])

    def _tokenize_messages(self, messages: List[Dict[str, Any]]) -> List[int]:
        """Tokenize a list of messages (with tool definitions in system prompt)."""
        if not messages:
            return []
        text = self.agent_model.tokenizer.apply_chat_template(
            messages,
            tools=self._tools_schema,
            tokenize=False,
            add_generation_prompt=True,
        )
        return self.agent_model.tokenizer.encode(text, add_special_tokens=False)

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
