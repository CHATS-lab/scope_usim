"""Co-training orchestrator for dual-trajectory rollouts.

Extends UserSimOrchestrator to produce TWO Trajectory objects from a single
conversation — one for the agent (persuader) and one for the opponent
(persuadee). Both trajectories share the same token sequence but differ
in loss_mask and logprobs:

  agent_traj:    loss_mask=1 on agent tokens,    =0 on opponent tokens
  opponent_traj: loss_mask=1 on opponent tokens,  =0 on agent tokens

The environment must expose:
  - last_opponent_output: Dict with token_ids, logprobs from opponent generate_fn
  - persuadee_messages: conversation history from opponent's perspective
  - persuadee_system_prompt: opponent's system prompt

Used by cotrain and selfplay modes where both models are SGLang-served.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from usim.core.types import (
    Trajectory,
    TrajectoryStatus,
    UserSimConfig,
    get_token_delta,
    probe_inter_message_glue,
)


logger = logging.getLogger(__name__)


class CoTrainingOrchestrator:
    """Dual-trajectory orchestrator for co-training / self-play.

    Runs the same rollout loop as UserSimOrchestrator but tracks two
    trajectories simultaneously — one per model. The agent's token_ids
    and logprobs come from the agent generate_fn; the opponent's come
    from env.last_opponent_output (set by the environment after each step).
    """

    def __init__(
        self,
        agent_tokenizer: Any,
        opponent_tokenizer: Any,
        config: UserSimConfig,
        cooperative: bool = False,
    ):
        """Initialize the co-training orchestrator.

        Args:
            agent_tokenizer: HuggingFace tokenizer for the agent (persuader)
            opponent_tokenizer: HuggingFace tokenizer for the opponent (persuadee)
            config: Training configuration (max_turns, temperature, etc.)
            cooperative: If True, opponent gets the same reward as agent.
                If False (default), opponent_reward = 1.0 - agent_reward (adversarial).
        """
        self.agent_tokenizer = agent_tokenizer
        self.opponent_tokenizer = opponent_tokenizer
        self.config = config
        self.cooperative = cooperative
        # Probe inter-message glue per tokenizer (Fix 2). See
        # ``probe_inter_message_glue`` in ``usim.core.types`` for details.
        self._agent_inter_message_glue: List[int] = probe_inter_message_glue(agent_tokenizer)
        self._opponent_inter_message_glue: List[int] = probe_inter_message_glue(opponent_tokenizer)
        self._agent_eos_token_id: Optional[int] = getattr(agent_tokenizer, "eos_token_id", None)
        self._opponent_eos_token_id: Optional[int] = getattr(opponent_tokenizer, "eos_token_id", None)

    async def rollout(
        self,
        env: Any,
        agent_generate_fn: Callable,
        sampling_params: Dict[str, Any],
    ) -> Tuple[Trajectory, Trajectory]:
        """Run a co-training rollout producing dual trajectories.

        The flow follows UserSimOrchestrator but additionally:
        1. Maintains a parallel opponent token sequence
        2. After each env.step(), reads env.last_opponent_output for
           the opponent's token_ids and logprobs
        3. Builds two Trajectory objects with complementary loss_masks

        Args:
            env: P4G environment with opponent_generate_fn set.
                Must expose persuadee_messages, persuadee_system_prompt,
                and last_opponent_output.
            agent_generate_fn: Agent's LLM generate function
            sampling_params: LLM sampling parameters

        Returns:
            (agent_trajectory, opponent_trajectory)
        """
        # Reset environment
        initial_messages, tools_schema, task_info = await env.reset()
        postprocess = env.prompt_postprocess_fn or (lambda x: x)

        # Agent conversation state
        agent_messages = list(initial_messages)
        agent_tokens = self._tokenize_prompt(
            self.agent_tokenizer, agent_messages, tools_schema, postprocess
        )
        agent_masks: List[int] = []
        agent_logprobs: List[float] = []

        # Opponent conversation state
        # Opponent sees the persuadee's system prompt, tokenized independently
        opponent_tokens = self._tokenize_prompt(
            self.opponent_tokenizer,
            env.persuadee_messages,
            None,
            lambda x: x,
        )
        opponent_masks: List[int] = []
        opponent_logprobs: List[float] = []

        total_reward = 0.0
        status = TrajectoryStatus.TRUNCATED
        num_turns = 0
        env_info: Dict[str, Any] = {}

        for turn in range(self.config.max_turns):
            # Token limit check
            if len(agent_tokens) + self.config.max_tokens > self.config.max_context_length:
                logger.warning(
                    f"Agent token limit at turn {turn}: {len(agent_tokens)} + "
                    f"{self.config.max_tokens} > {self.config.max_context_length}"
                )
                status = TrajectoryStatus.TRUNCATED
                break

            # 1. Generate agent (persuader) action
            try:
                output = await agent_generate_fn(agent_tokens, sampling_params)

                # Fix 4: inference-server error detection.
                if output.get("error"):
                    logger.warning(f"Agent inference error at turn {turn}: {output['error']}")
                    status = TrajectoryStatus.FAILED
                    break

                meta = output.get("meta_info", {})
                finish_reason = meta.get("finish_reason", {})
                if isinstance(finish_reason, dict) and finish_reason.get("type") == "abort":
                    status = TrajectoryStatus.FAILED
                    break

                response_text = output["text"]
            except Exception as e:
                logger.error(f"Agent generation failed: {e}")
                status = TrajectoryStatus.FAILED
                break

            # 2. Parse response (P4G is plain text, no tools)
            parsed = env.parse_response(response_text)
            if parsed is None:
                logger.warning(f"Response parsing failed: {response_text[:200]}...")
                status = TrajectoryStatus.FAILED
                break

            # 3. Track agent tokens from model output DIRECTLY
            agent_messages.append({"role": "assistant", "content": response_text})
            response_tokens = output["token_ids"]
            response_logprobs = output["logprobs"]
            agent_tokens.extend(response_tokens)
            agent_masks.extend([1] * len(response_tokens))  # agent trained
            agent_logprobs.extend(response_logprobs)

            # Fix 3: max_tokens truncation guard for agent.
            agent_hit_eos = (
                bool(response_tokens)
                and self._agent_eos_token_id is not None
                and response_tokens[-1] == self._agent_eos_token_id
            )
            if not agent_hit_eos:
                status = TrajectoryStatus.TRUNCATED
                num_turns = turn + 1
                break

            # Fix 2: append inter-message glue after a completed agent response.
            if self._agent_inter_message_glue:
                agent_tokens.extend(self._agent_inter_message_glue)
                agent_masks.extend([0] * len(self._agent_inter_message_glue))
                agent_logprobs.extend([0.0] * len(self._agent_inter_message_glue))

            # 4. Snapshot opponent messages before env.step() mutates them
            opp_messages_before_step = list(env.persuadee_messages)

            # 5. Build action and step environment
            action = parsed["normal_text"]
            try:
                obs, reward, terminated, truncated, step_info = await env.step(action)
                total_reward = reward
            except Exception as e:
                logger.warning(f"Environment step failed: {e}")
                status = TrajectoryStatus.FAILED
                break

            # 6. After env.step(), persuadee_messages has two new entries:
            #    [... existing ..., {"role": "user", ...}, {"role": "assistant", ...}]
            #    We compute token deltas for each separately.

            # 6a. Agent's message as "user" in opponent's token stream (loss_mask=0).
            # Fix 1: delta correctly includes the opponent's next assistant
            # generation prompt via the canonical helper.
            opp_messages_with_user = opp_messages_before_step + [
                {"role": "user", "content": action}
            ]
            opp_user_delta, _ = get_token_delta(
                self.opponent_tokenizer, opp_messages_with_user,
            )
            opponent_tokens.extend(opp_user_delta)
            opponent_masks.extend([0] * len(opp_user_delta))
            opponent_logprobs.extend([0.0] * len(opp_user_delta))

            # 6b. Opponent's response tokens from last_opponent_output (loss_mask=1)
            opponent_output = env.last_opponent_output
            if opponent_output is not None:
                opp_response_tokens = opponent_output["token_ids"]
                opp_response_logprobs = opponent_output["logprobs"]
                opponent_tokens.extend(opp_response_tokens)
                opponent_masks.extend([1] * len(opp_response_tokens))
                opponent_logprobs.extend(opp_response_logprobs)

                # Fix 2: append inter-message glue after the opponent's EOS.
                opp_hit_eos = (
                    bool(opp_response_tokens)
                    and self._opponent_eos_token_id is not None
                    and opp_response_tokens[-1] == self._opponent_eos_token_id
                )
                if opp_hit_eos and self._opponent_inter_message_glue:
                    opponent_tokens.extend(self._opponent_inter_message_glue)
                    opponent_masks.extend([0] * len(self._opponent_inter_message_glue))
                    opponent_logprobs.extend(
                        [0.0] * len(self._opponent_inter_message_glue)
                    )
            else:
                # API-based opponent — no token-level tracking, use re-tokenization
                opp_resp_delta, _ = get_token_delta(
                    self.opponent_tokenizer, env.persuadee_messages,
                )
                opponent_tokens.extend(opp_resp_delta)
                opponent_masks.extend([0] * len(opp_resp_delta))
                opponent_logprobs.extend([0.0] * len(opp_resp_delta))

            # Track opponent's response in agent's token stream (loss_mask=0).
            # Fix 1: delta correctly includes the agent's next assistant
            # generation prompt via the canonical helper.
            agent_messages.append({"role": "user", "content": obs})
            agent_delta, _ = get_token_delta(
                self.agent_tokenizer, agent_messages,
                tools_schema=tools_schema, postprocess=postprocess,
            )
            agent_tokens.extend(agent_delta)
            agent_masks.extend([0] * len(agent_delta))  # not trained
            agent_logprobs.extend([0.0] * len(agent_delta))

            env_info = {**env_info, **step_info}
            num_turns = turn + 1

            if terminated:
                status = TrajectoryStatus.COMPLETED
                break
            if truncated:
                status = TrajectoryStatus.TRUNCATED
                break

        # Build both trajectories
        if self.cooperative:
            opponent_reward = total_reward
        else:
            # Adversarial: agent wants high reward, opponent resists
            opponent_reward = 1.0 - total_reward
        agent_traj = self._build_trajectory(
            agent_tokens, agent_masks, agent_logprobs,
            agent_messages, num_turns, task_info, status, total_reward, env_info,
        )
        opponent_traj = self._build_trajectory(
            opponent_tokens, opponent_masks, opponent_logprobs,
            env.persuadee_messages, num_turns, task_info, status, opponent_reward, env_info,
        )

        return agent_traj, opponent_traj

    def _tokenize_prompt(
        self,
        tokenizer: Any,
        messages: List[Dict[str, Any]],
        tools_schema: Optional[List[Dict[str, Any]]],
        postprocess: Callable[[str], str],
    ) -> List[int]:
        """Tokenize initial prompt messages."""
        kwargs: Dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
        if tools_schema:
            kwargs["tools"] = tools_schema
        text = tokenizer.apply_chat_template(messages, **kwargs)
        text = postprocess(text)
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    def _get_token_delta_for_messages(
        self,
        tokenizer: Any,
        messages: List[Dict[str, Any]],
        tools_schema: Optional[List[Dict[str, Any]]],
        postprocess: Callable[[str], str],
    ) -> Tuple[List[int], List[int]]:
        """Thin wrapper around the canonical ``get_token_delta`` helper.

        Kept for backward-compatibility with existing callers. All rollout
        logic in this class now routes through ``get_token_delta`` from
        ``usim.core.types`` directly.
        """
        return get_token_delta(
            tokenizer, messages, tools_schema=tools_schema, postprocess=postprocess,
        )

    def _get_token_delta_simple(
        self,
        tokenizer: Any,
        messages: List[Dict[str, Any]],
        tools_schema: Optional[List[Dict[str, Any]]],
        postprocess: Callable[[str], str],
    ) -> List[int]:
        """Get token delta for the last message (returns only token IDs)."""
        tokens, _ = get_token_delta(
            tokenizer, messages, tools_schema=tools_schema, postprocess=postprocess,
        )
        return tokens

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
