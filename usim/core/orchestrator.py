"""Gym-based orchestrator for agent-environment rollouts.

Takes an environment (BaseEnvironment protocol) and a generate function,
runs the full rollout loop, and returns a Trajectory for RL training.

The orchestrator is environment-agnostic — all env-specific logic (observation
conversion, tool parsing, prompt postprocessing) lives in the environment.
Works with any Gym env: tau2-bench, Persuasion for Good, CooperBench, etc.

Token Tracking (spare convention):
    - all_tokens: starts with prompt tokens, grows with each turn
    - all_masks: 1 for assistant tokens, 0 for everything else
    - all_logprobs: model logprobs for assistant tokens, 0.0 for env tokens
    - Invariant: len(all_masks) == len(all_logprobs)
    - Invariant: len(all_tokens) >= len(all_masks)
    - base_offset = len(all_tokens) - len(all_masks) = prompt length
    - For assistant tokens: token_ids and logprobs from model output DIRECTLY
    - For env/user/tool tokens: computed via _get_token_delta, logprobs=0.0
"""

import json
import logging
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

from usim.core.types import (
    Trajectory,
    TrajectoryStatus,
    UserSimConfig,
)


logger = logging.getLogger(__name__)


class UserSimOrchestrator:
    """Gym-based orchestrator for agent-environment rollouts.

    The environment handles user simulation, tool execution, and turn management
    internally via step(). The orchestrator just:
    1. Resets the environment to get initial state
    2. Generates agent actions via LLM (passing input_ids for TITO)
    3. Steps the environment (async)
    4. Tracks tokens, loss masks, and logprobs for training
    """

    def __init__(
        self,
        tokenizer: Any,
        config: UserSimConfig,
    ):
        """Initialize the orchestrator.

        Args:
            tokenizer: HuggingFace tokenizer for token tracking
            config: Training configuration (max_turns, temperature, etc.)
        """
        self.tokenizer = tokenizer
        self.config = config

    async def rollout(
        self,
        env: Any,
        generate_fn: Callable,
        sampling_params: Dict[str, Any],
    ) -> Trajectory:
        """Run a complete agent-environment rollout.

        1. env.reset() → initial_messages, tools_schema, task_info
        2. Loop: generate(input_ids) → parse → track assistant tokens → step env → track env tokens
        3. Return Trajectory with accumulated tokens, masks, logprobs

        Args:
            env: Environment implementing BaseEnvironment protocol.
                Must have: reset(), step(), parse_response(), prompt_postprocess_fn
            generate_fn: async (input_ids: List[int], sampling_params: Dict) -> Dict
                Must return {"text", "token_ids", "logprobs", "meta_info"}.
                token_ids and logprobs from sglang's output_token_logprobs.
            sampling_params: LLM sampling parameters

        Returns:
            Trajectory with tokens, loss_mask, rollout_log_probs, reward, etc.
        """
        # Reset environment
        initial_messages, tools_schema, task_info = await env.reset()
        postprocess = env.prompt_postprocess_fn or (lambda x: x)

        messages = list(initial_messages)

        # Tokenize initial prompt
        all_tokens = self._tokenize_prompt(messages, tools_schema, postprocess)
        all_masks: List[int] = []
        all_logprobs: List[float] = []

        total_reward = 0.0
        status = TrajectoryStatus.TRUNCATED
        num_turns = 0
        env_info: Dict[str, Any] = {}

        for turn in range(self.config.max_turns):
            # Token limit check before generation
            if len(all_tokens) + self.config.max_tokens > self.config.max_context_length:
                logger.warning(
                    f"Token limit at turn {turn}: {len(all_tokens)} + "
                    f"{self.config.max_tokens} > {self.config.max_context_length}"
                )
                status = TrajectoryStatus.TRUNCATED
                break

            # 1. Generate via LLM (TITO: pass accumulated input_ids)
            try:
                output = await generate_fn(all_tokens, sampling_params)

                meta = output.get("meta_info", {})
                finish_reason = meta.get("finish_reason", {})
                if isinstance(finish_reason, dict) and finish_reason.get("type") == "abort":
                    status = TrajectoryStatus.FAILED
                    break

                response_text = output["text"]
            except Exception as e:
                logger.error(f"Generation failed: {e}")
                status = TrajectoryStatus.FAILED
                break

            # 2. Parse response (tool calls or plain text)
            parsed = env.parse_response(response_text)
            if parsed is None:
                logger.warning(f"Response parsing failed: {response_text[:200]}...")
                status = TrajectoryStatus.FAILED
                break

            # 3. Track assistant tokens from model output DIRECTLY
            #    Using model's token_ids and logprobs ensures exact match
            #    with what the model actually generated (critical for GRPO).
            messages.append({"role": "assistant", "content": response_text})
            response_tokens = output["token_ids"]
            response_logprobs = output["logprobs"]
            all_tokens.extend(response_tokens)
            all_masks.extend([1] * len(response_tokens))
            all_logprobs.extend(response_logprobs)

            # 4. Build action for environment
            calls = parsed["calls"]
            if calls:
                tool_call_id = calls[0].get("id") or str(uuid.uuid4())
                action = json.dumps({
                    "name": calls[0]["name"],
                    "arguments": calls[0]["arguments"],
                    "id": tool_call_id,
                })
                tool_called = True
            else:
                tool_call_id = None
                action = parsed["normal_text"]
                tool_called = False

            # 5. Step environment (async)
            try:
                obs, reward, terminated, truncated, step_info = await env.step(action)
                total_reward = reward
            except Exception as e:
                logger.warning(f"Environment step failed: {e}")
                status = TrajectoryStatus.FAILED
                break

            # 6. Track environment response tokens (loss_mask=0, logprobs=0.0)
            if tool_called:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": obs,
                })
            else:
                messages.append({"role": "user", "content": obs})

            delta, mask = self._get_token_delta(messages, tools_schema, postprocess)
            all_tokens.extend(delta)
            all_masks.extend(mask)
            all_logprobs.extend([0.0] * len(delta))

            env_info = {**env_info, **step_info}
            num_turns = turn + 1

            if terminated:
                status = TrajectoryStatus.COMPLETED
                break
            if truncated:
                status = TrajectoryStatus.TRUNCATED
                break

        return self._build_trajectory(
            all_tokens, all_masks, all_logprobs,
            messages, num_turns, task_info, status, total_reward, env_info,
        )

    def _tokenize_prompt(
        self,
        messages: List[Dict[str, Any]],
        tools_schema: Optional[List[Dict[str, Any]]],
        postprocess: Callable[[str], str],
    ) -> List[int]:
        """Tokenize initial prompt messages."""
        kwargs: Dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
        if tools_schema:
            kwargs["tools"] = tools_schema
        text = self.tokenizer.apply_chat_template(messages, **kwargs)
        text = postprocess(text)
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def _get_token_delta(
        self,
        messages: List[Dict[str, Any]],
        tools_schema: Optional[List[Dict[str, Any]]],
        postprocess: Callable[[str], str],
    ) -> Tuple[List[int], List[int]]:
        """Calculate token delta for the last message added.

        Used for user/tool/environment tokens (NOT assistant tokens, which
        come directly from model output for TITO accuracy).

        Returns (token_ids, loss_mask) where loss_mask=0 for env tokens.
        """
        template_kwargs: Dict[str, Any] = {}
        if tools_schema:
            template_kwargs["tools"] = tools_schema

        curr = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, **template_kwargs,
        )
        curr = postprocess(curr)

        prev_messages = messages[:-1]
        prev = self.tokenizer.apply_chat_template(
            prev_messages, tokenize=False, add_generation_prompt=False, **template_kwargs,
        )
        prev = postprocess(prev)

        delta_text = curr[len(prev):]
        new_tokens = self.tokenizer.encode(delta_text, add_special_tokens=False)
        loss_mask = [0] * len(new_tokens)

        return new_tokens, loss_mask

    def _build_trajectory(
        self,
        all_tokens: List[int],
        all_masks: List[int],
        all_logprobs: List[float],
        messages: List[Dict[str, Any]],
        num_turns: int,
        task_info: Dict[str, Any],
        status: TrajectoryStatus,
        reward: float,
        info: Dict[str, Any],
    ) -> Trajectory:
        """Build Trajectory from accumulated data."""
        response_parts = []
        for msg in messages:
            if msg.get("role") == "assistant" and msg.get("content"):
                response_parts.append(msg["content"])

        return Trajectory(
            index=0,
            prompt=task_info.get("instructions", ""),
            tokens=all_tokens,
            loss_mask=all_masks,
            rollout_log_probs=all_logprobs,
            response="".join(response_parts),
            response_length=len(all_masks),
            reward=reward,
            status=status,
            messages=messages,
            turn_count=num_turns,
            metadata={
                "task_id": task_info.get("id", ""),
                "domain": task_info.get("domain", ""),
                **info,
            },
        )
