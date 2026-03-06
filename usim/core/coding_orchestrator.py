"""Orchestrator for Agent <-> Environment coding loops using tool calling.

Unlike UserSimOrchestrator (agent <-> user_sim conversation), this orchestrator
manages agent <-> environment loops following mini-swe-agent-v2 protocol:
every agent response contains structured tool calls (bash + optional send_message),
and the environment executes them and returns JSON-formatted observations.

Used for CooperBench coding tasks where the agent implements features
in a sandbox environment.
"""

import html
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

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

# Regex patterns for Qwen3.5 XML-style tool calls:
#   <tool_call>
#   <function=name>
#   <parameter=key>value</parameter>
#   </function>
#   </tool_call>
_TOOL_CALL_BLOCK_RE = re.compile(
    r"<tool_call>(.*?)</tool_call>", re.DOTALL
)
_FUNCTION_RE = re.compile(
    r"<function=(\w+)>(.*?)</function>", re.DOTALL
)
_PARAMETER_RE = re.compile(
    r"<parameter=(\w+)>(.*?)</parameter>", re.DOTALL
)

# Also support JSON-style tool calls (Qwen2.5 format):
#   <tool_call>
#   {"name": "bash", "arguments": {"command": "ls"}}
#   </tool_call>
_JSON_TOOL_CALL_RE = re.compile(
    r'\{\s*"name"\s*:\s*"(\w+)"\s*,\s*"arguments"\s*:\s*(\{.*?\})\s*\}', re.DOTALL
)


def _parse_xml_block(block_content: str) -> Optional[Dict[str, Any]]:
    """Parse a single <tool_call> block (XML format: Qwen3.5/qwen3_coder)."""
    func_match = _FUNCTION_RE.search(block_content)
    if func_match:
        func_name = func_match.group(1)
        func_body = func_match.group(2)
        params = {}
        for param_match in _PARAMETER_RE.finditer(func_body):
            key = param_match.group(1)
            val = html.unescape(param_match.group(2).strip())
            # Try to parse as JSON (for numeric/bool values)
            try:
                val = json.loads(val)
            except (json.JSONDecodeError, ValueError):
                pass
            params[key] = val
        return {"name": func_name, "arguments": params, "id": ""}
    return None


def _parse_json_block(block_content: str) -> Optional[Dict[str, Any]]:
    """Parse a single <tool_call> block (JSON format: Qwen2.5/qwen25)."""
    match = _JSON_TOOL_CALL_RE.search(block_content)
    if match:
        func_name = match.group(1)
        try:
            args = json.loads(match.group(2))
        except (json.JSONDecodeError, ValueError):
            args = {}
        return {"name": func_name, "arguments": args, "id": ""}
    # Also try raw JSON parse
    try:
        data = json.loads(block_content.strip())
        if isinstance(data, dict) and "name" in data:
            args = data.get("arguments", data.get("parameters", {}))
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, ValueError):
                    args = {}
            return {"name": data["name"], "arguments": args, "id": ""}
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def _parse_tool_calls(
    response_text: str,
    tools_schema: List[Dict[str, Any]],
    tool_call_parser: str = "qwen3_coder",
) -> Dict[str, Any]:
    """Parse tool calls from agent response text.

    Self-contained parser — no sglang dependency. Supports both XML format
    (Qwen3.5/qwen3_coder) and JSON format (Qwen2.5/qwen25) inside <tool_call> blocks.

    Returns {"normal_text": str, "calls": list} where each call is
    {"name": str, "arguments": dict, "id": str}.
    """
    if response_text.endswith("<|im_end|>"):
        response_text = response_text[:-10]
    response_text = response_text.strip()

    if not tools_schema:
        return {"normal_text": response_text, "calls": []}

    calls = []
    normal_parts = []
    cursor = 0

    for match in _TOOL_CALL_BLOCK_RE.finditer(response_text):
        # Collect text before this block
        normal_parts.append(response_text[cursor:match.start()])
        cursor = match.end()

        block_content = match.group(1)
        # Try XML format first (Qwen3.5), then JSON (Qwen2.5)
        parsed = _parse_xml_block(block_content)
        if parsed is None:
            parsed = _parse_json_block(block_content)
        if parsed is not None:
            calls.append(parsed)
        else:
            logger.warning(
                f"Could not parse tool_call block: {block_content[:200]!r}"
            )

    # Remaining text after last block
    normal_parts.append(response_text[cursor:])
    normal_text = "".join(normal_parts).strip()

    if not calls:
        logger.debug(
            f"No <tool_call> blocks found (parser={tool_call_parser}). "
            f"Response preview: {response_text[:300]!r}"
        )

    return {"normal_text": normal_text, "calls": calls}


def _format_tool_observation(
    output: str,
    returncode: int,
    incoming_messages: Optional[List[Dict[str, Any]]] = None,
    max_output_len: int = 4000,
) -> str:
    """Format command output as JSON tool result (v2 observation_template format)."""
    if len(output) < max_output_len:
        obs: Dict[str, Any] = {"returncode": returncode, "output": output}
    else:
        half = max_output_len // 2
        obs = {
            "returncode": returncode,
            "output_head": output[:half],
            "output_tail": output[-half:],
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
        tool_call_parser: str = "qwen3_coder",
        chat_template_kwargs: Optional[Dict[str, Any]] = None,
        max_tool_output_chars: int = 4000,
    ):
        """Initialize the coding orchestrator.

        Args:
            agent_model: ModelAdapter for agent generation (trainable, SGLang)
            config: Configuration (max_turns used as max_steps)
            environment: CooperBenchEnvironment with execute_command() and get_tools()
            messaging: Optional MessagingConnector for inter-agent communication
            tool_call_parser: sglang parser name for the model's tool call format.
                "qwen" for Qwen3, "hermes" for Hermes/Mistral, "llama3" for Llama 3.1+.
            chat_template_kwargs: Extra kwargs for apply_chat_template (e.g. enable_thinking=False)
            max_tool_output_chars: Max chars in tool output before truncation (default: 4000)
        """
        self.agent_model = agent_model
        self.config = config
        self.env = environment
        self.messaging = messaging
        self.tool_call_parser = tool_call_parser
        self._tools_schema = environment.get_tools()
        self._chat_template_kwargs = chat_template_kwargs or {}
        self._max_tool_output_chars = max_tool_output_chars

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

        # Add task description as first user message
        task_content = task.get("instructions", "")
        task_msg = Message(role="user", content=task_content)
        agent_state = agent_state.add_message(task_msg)
        messages.append(task_msg.to_dict())

        # Tokenize system + task message together as prompt
        # (Qwen3.5 chat template requires a user message to be present)
        initial_messages = [msg.to_dict() for msg in agent_state.system_messages] + [
            task_msg.to_dict()
        ]
        prompt_tokens = self._tokenize_messages(initial_messages)
        all_tokens.extend(prompt_tokens)

        # Main coding loop
        step_count = 0
        max_steps = self.config.max_turns
        max_context = getattr(self.config, "max_context_length", 0) or 32768
        status = TrajectoryStatus.TRUNCATED
        has_written_code = False
        tool_call_success_count = 0
        tool_call_fail_count = 0
        first_failed_response = ""

        for step in range(max_steps):
            step_count = step + 1

            # Check context length — truncate if approaching limit
            if len(all_tokens) + self.config.max_tokens > max_context:
                logger.warning(
                    f"Context length ({len(all_tokens)} tokens) approaching limit "
                    f"({max_context}), truncating at step {step_count}"
                )
                status = TrajectoryStatus.TRUNCATED
                break

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
                tool_call_fail_count += 1
                # Save first failed response for debugging (visible in wandb metadata)
                if not first_failed_response:
                    first_failed_response = response_text[:500]
                logger.warning(
                    f"[step {step_count}] No tool calls parsed (parser={self.tool_call_parser}). "
                    f"Response preview: {response_text[:300]!r}"
                )
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
            tool_call_success_count += 1
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
                    is_write = any(
                        k in command for k in (
                            "cat >", "cat <<", "sed -i", "echo >", "tee ",
                            "patch ", "> ", ">>", "printf ", "cp ", "mv ",
                        )
                    )
                    if is_write:
                        has_written_code = True
                    logger.info(
                        f"[step {step_count}] bash({'W' if is_write else 'R'}): "
                        f"{command[:200]}{'...' if len(command) > 200 else ''}"
                    )
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
                            max_output_len=self._max_tool_output_chars,
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

        logger.info(
            f"Session done: steps={step_count}, status={status.value}, "
            f"tool_ok={tool_call_success_count}, tool_fail={tool_call_fail_count}, "
            f"wrote_code={has_written_code}, tokens={len(all_tokens)}"
        )

        traj = self._build_trajectory(
            all_tokens, all_masks, all_logprobs, messages, step_count, task, status,
        )
        traj.metadata["tool_call_success"] = tool_call_success_count
        traj.metadata["tool_call_fail"] = tool_call_fail_count
        if first_failed_response:
            traj.metadata["first_failed_response"] = first_failed_response
        traj.metadata["has_written_code"] = has_written_code
        return traj

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

    @staticmethod
    def _ensure_dict_arguments(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Ensure tool_calls arguments are dicts (not JSON strings) for apply_chat_template.

        Qwen3.5's Jinja2 chat template calls .items() on arguments, which fails
        if arguments are JSON strings. This converts them to dicts.
        """
        result = []
        for msg in messages:
            if msg.get("tool_calls"):
                msg = dict(msg)
                msg["tool_calls"] = [
                    {
                        **tc,
                        "function": {
                            **tc["function"],
                            "arguments": (
                                json.loads(tc["function"]["arguments"])
                                if isinstance(tc["function"]["arguments"], str)
                                else tc["function"]["arguments"]
                            ),
                        },
                    }
                    for tc in msg["tool_calls"]
                ]
            result.append(msg)
        return result

    def _get_token_delta(
        self,
        messages: List[Dict[str, Any]],
        role: str,
    ) -> Tuple[List[int], List[int]]:
        """Calculate token delta for the last message added."""
        if not messages:
            return [], []

        # Preprocess: ensure tool_calls have dict arguments for apply_chat_template
        messages = self._ensure_dict_arguments(messages)

        tokenizer = self.agent_model.tokenizer
        tmpl_kwargs = self._chat_template_kwargs
        curr_text = tokenizer.apply_chat_template(
            messages,
            tools=self._tools_schema,
            tokenize=False,
            add_generation_prompt=False,
            **tmpl_kwargs,
        )
        prev_messages = messages[:-1]

        if messages[-1].get("role") == "assistant":
            prev_text = (
                tokenizer.apply_chat_template(
                    prev_messages,
                    tools=self._tools_schema,
                    tokenize=False,
                    add_generation_prompt=True,
                    **tmpl_kwargs,
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
                    **tmpl_kwargs,
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
            **self._chat_template_kwargs,
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
            response_length=len(loss_mask),
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
