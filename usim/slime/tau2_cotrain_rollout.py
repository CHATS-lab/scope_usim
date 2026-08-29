"""tau2-bench co-training rollout function for Slime.

Uses the standard tau2 rollout path for the agent trajectory, but replaces
the internal litellm user simulator with an SGLang-backed one. This lets
both the agent and user simulator train via separate training groups.

Agent trajectory: built by UserSimOrchestrator (same as single-model tau2)
Opponent trajectory: built from captured SGLang user sim outputs

The opponent (user simulator) gets the same reward as the agent because
tau2-bench is a collaborative task.
"""

import asyncio
import logging
import math
import os
from argparse import Namespace
from typing import Any, Dict, List

import weave
from slime.rollout.base_types import RolloutFnTrainOutput
from slime.utils.http_utils import post
from slime.utils.processing_utils import load_tokenizer
from slime.utils.types import Sample

from usim.core.environment.tau2 import Tau2Environment
from usim.core.orchestrator import UserSimOrchestrator
from usim.core.rollout_metrics import compute_rollout_metrics
from usim.core.trajectory_recorder import TrajectoryRecorder
from usim.core.types import Trajectory, TrajectoryStatus, TrainableRole, UserSimConfig
from usim.slime.sglang_user_simulator import SGLangUserSimulator
from usim.slime.trajectory_converter import trajectory_to_slime_sample
from usim.utils.docent import get_docent_uploader

from tau2.gym import AgentGymEnv


logger = logging.getLogger(__name__)


def _get_model_url(args, model_name: str, endpoint: str) -> str:
    """Get URL for a named model's SGLang endpoint.

    For cotrain, sglang-config defines multiple models (actor, opponent).
    Each has its own router port. This reads model configs from args.
    Falls back to the default router URL for single-model setups.
    """
    # Try sglang-config based multi-model setup
    sglang_config = getattr(args, "sglang_config_data", None)
    if sglang_config and "models" in sglang_config:
        for model in sglang_config["models"]:
            if model.get("name") == model_name:
                host = getattr(args, "sglang_router_ip", "127.0.0.1")
                port = model.get("router_port", getattr(args, "sglang_router_port", 4702))
                return f"http://{host}:{port}{endpoint}"

    # Fallback: use default router
    host = getattr(args, "sglang_router_ip", "127.0.0.1")
    port = getattr(args, "sglang_router_port", 4702)
    return f"http://{host}:{port}{endpoint}"


class _SGLangAgentGymEnv(AgentGymEnv):
    """AgentGymEnv subclass that uses SGLang user simulator instead of litellm."""

    def __init__(self, sglang_url: str, tokenizer: Any, sampling_params: Dict, **kwargs):
        self._sglang_url = sglang_url
        self._sglang_tokenizer = tokenizer
        self._sglang_sampling_params = sampling_params
        self.sglang_user_sim = None  # Set after _get_user() is called
        super().__init__(**kwargs)

    def _get_user(self):
        environment = self._get_environment()
        task = self._get_task()
        try:
            user_tools = environment.get_user_tools()
        except ValueError:
            user_tools = None

        self.sglang_user_sim = SGLangUserSimulator(
            sglang_url=self._sglang_url,
            tokenizer=self._sglang_tokenizer,
            sampling_params=self._sglang_sampling_params,
            tools=user_tools,
            instructions=task.user_scenario,
            llm=self.user_llm,
            llm_args=self.user_llm_args,
        )
        return self.sglang_user_sim


class Tau2CoTrainEnvironment:
    """tau2-bench environment for co-training with SGLang opponent.

    Wraps _SGLangAgentGymEnv and exposes the user simulator's captured
    outputs for opponent trajectory construction.
    """

    def __init__(
        self,
        domain: str,
        task_id: str,
        max_turns: int,
        sglang_url: str,
        tokenizer: Any,
        sampling_params: Dict,
    ):
        from tau2.agent.llm_agent import AGENT_INSTRUCTION, SYSTEM_PROMPT

        self._domain = domain
        self._task_id = task_id
        self._system_prompt_template = SYSTEM_PROMPT
        self._agent_instruction = AGENT_INSTRUCTION

        self._env = _SGLangAgentGymEnv(
            sglang_url=sglang_url,
            tokenizer=tokenizer,
            sampling_params=sampling_params,
            domain=domain,
            task_id=task_id,
            max_steps=max_turns,
            solo_mode=False,
            user_llm="sglang-local",  # Placeholder, overridden by _get_user
        )
        self._tools_schema = None

    async def reset(self):
        from tau2.agent.llm_agent import AGENT_INSTRUCTION, SYSTEM_PROMPT
        from usim.core.environment.tau2.environment import _observation_to_messages

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

    async def step(self, action):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._env.step, action)

    def parse_response(self, response_text):
        from usim.core.environment.tau2.environment import _parse_tool_calls
        return _parse_tool_calls(response_text, self._tools_schema)

    @property
    def prompt_postprocess_fn(self):
        from usim.core.environment.tau2.environment import _tau2_prompt_postprocess
        return _tau2_prompt_postprocess

    @property
    def sglang_user_sim(self):
        return self._env.sglang_user_sim


def _build_opponent_trajectory(
    agent_traj: Trajectory,
    user_sim: SGLangUserSimulator,
    tokenizer: Any,
) -> Trajectory:
    """Build opponent trajectory from captured SGLang user sim outputs.

    The opponent trajectory has:
    - loss_mask=1 on user simulator tokens (trainable)
    - loss_mask=0 on agent tokens (not trained)
    - Same reward as agent (cooperative task)
    """
    all_tokens = []
    all_masks = []
    all_logprobs = []

    # Build the opponent's view: system prompt + conversation
    # Use the user sim's captured outputs for token_ids/logprobs
    gen_idx = 0
    for output in user_sim.generation_outputs:
        # Opponent's own response tokens (trainable)
        all_tokens.extend(output["token_ids"])
        all_masks.extend([1] * len(output["token_ids"]))
        all_logprobs.extend(output["logprobs"])
        gen_idx += 1

    # Cooperative: opponent gets same reward as agent
    return Trajectory(
        index=agent_traj.index,
        prompt=agent_traj.prompt,
        tokens=all_tokens,
        loss_mask=all_masks,
        rollout_log_probs=all_logprobs,
        response="",
        response_length=len(all_masks),
        reward=agent_traj.reward,
        status=agent_traj.status,
        messages=[],
        turn_count=agent_traj.turn_count,
        metadata={**agent_traj.metadata, "role": "opponent"},
    )


async def _tau2_cotrain_generate_single(
    args: Any,
    sample: Any,
    sampling_params: Dict[str, Any],
) -> tuple:
    """Per-sample tau2 co-training rollout."""
    try:
        metadata = getattr(sample, "metadata", {}) or {}

        domain = getattr(args, "usim_domain", "retail")
        max_turns = getattr(args, "max_turns", 30)

        # Build generate functions
        agent_url = _get_model_url(args, "actor", "/generate")
        opponent_url = _get_model_url(args, "opponent", "/generate")

        agent_hf = getattr(args, "hf_checkpoint", None)
        opponent_hf = getattr(args, "opponent_hf_checkpoint", None) or agent_hf
        agent_tokenizer = load_tokenizer(agent_hf, trust_remote_code=True)
        opponent_tokenizer = load_tokenizer(opponent_hf, trust_remote_code=True)

        # Agent generate function (TITO via SGLang)
        @weave.op
        async def agent_generate_fn(input_ids, samp_params):
            output = await post(agent_url, {
                "input_ids": input_ids,
                "sampling_params": samp_params,
                "return_logprob": True,
            })
            meta = output.get("meta_info", {})
            token_logprobs = meta.get("output_token_logprobs", [])
            return {
                "text": output["text"],
                "token_ids": [item[1] for item in token_logprobs],
                "logprobs": [item[0] for item in token_logprobs],
                "meta_info": meta,
            }

        opp_sampling_params = {
            "temperature": sampling_params.get("temperature", 0.7),
            "top_p": sampling_params.get("top_p", 0.95),
            "max_new_tokens": 1024,
        }

        task_ids = metadata.get("task_ids", [])
        task_id = task_ids[sample.index % len(task_ids)] if task_ids else str(sample.index)

        env = Tau2CoTrainEnvironment(
            domain=domain,
            task_id=task_id,
            max_turns=max_turns,
            sglang_url=opponent_url,
            tokenizer=opponent_tokenizer,
            sampling_params=opp_sampling_params,
        )

        config = UserSimConfig(
            trainable_role=TrainableRole(getattr(args, "trainable_role", "agent")),
            max_turns=max_turns,
            max_tokens=getattr(args, "max_tokens", 8192),
            max_context_length=getattr(args, "rollout_max_response_len", 32768),
            temperature=sampling_params.get("temperature", 0.7),
        )

        orchestrator = UserSimOrchestrator(tokenizer=agent_tokenizer, config=config)
        agent_traj = await orchestrator.rollout(env, agent_generate_fn, sampling_params)

        logger.info(
            f"tau2 cotrain rollout {sample.index}: {agent_traj.turn_count} turns, "
            f"reward={agent_traj.reward:.3f}, status={agent_traj.status}"
        )

        # Build opponent trajectory from captured user sim outputs
        opponent_traj = _build_opponent_trajectory(
            agent_traj, env.sglang_user_sim, opponent_tokenizer
        )

        agent_sample = trajectory_to_slime_sample(agent_traj, sample.index, role="actor")
        opponent_sample = trajectory_to_slime_sample(opponent_traj, sample.index, role="opponent")
        return agent_sample, opponent_sample

    except Exception as e:
        logger.error(f"tau2 cotrain rollout failed (sample {sample.index}): {e}", exc_info=True)
        error_sample = Sample(
            index=sample.index,
            prompt=getattr(sample, "prompt", ""),
            tokens=[],
            response="",
            reward=0.0,
            loss_mask=[],
            response_length=0,
            metadata={"error": str(e)},
        )
        return error_sample, error_sample


async def _run_tau2_cotrain_batch_async(
    args: Namespace,
    samples: List[List[Sample]],
) -> tuple:
    """Run tau2 co-training batch."""
    sampling_params = dict(
        temperature=args.rollout_temperature,
        top_p=getattr(args, "rollout_top_p", 0.95),
        top_k=getattr(args, "rollout_top_k", -1),
        max_new_tokens=8192,
    )

    tasks = []
    for group in samples:
        for sample in group:
            tasks.append(_tau2_cotrain_generate_single(args, sample, sampling_params))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    agent_grouped = []
    opponent_grouped = []
    idx = 0
    for group in samples:
        agent_group = []
        opponent_group = []
        for _ in group:
            result = results[idx]
            if isinstance(result, Exception):
                logger.error(f"Sample failed: {result}")
                error_sample = Sample(
                    index=group[0].index,
                    tokens=[],
                    response="",
                    reward=0.0,
                    loss_mask=[],
                    response_length=0,
                    metadata={"error": str(result)},
                )
                agent_group.append(error_sample)
                opponent_group.append(error_sample)
            else:
                agent_sample, opponent_sample = result
                agent_group.append(agent_sample)
                opponent_group.append(opponent_sample)
            idx += 1
        agent_grouped.append(agent_group)
        opponent_grouped.append(opponent_group)

    return agent_grouped, opponent_grouped


def _variance_curriculum_reward(var: float, center: float = 0.25, sigma2: float = 0.02) -> float:
    """Gaussian curriculum signal: r = exp(-(var - center)^2 / sigma2).

    Peaks at 1.0 when the agent's group reward variance equals `center` (0.25
    is the max achievable Bernoulli variance, i.e. a 50/50 pass rate). Rewards
    the user simulator for producing task difficulty that splits agent
    outcomes, following the SPICE paper (arXiv:2510.24684).
    """
    diff = var - center
    return math.exp(-(diff * diff) / sigma2)


def _apply_user_curriculum_reward(
    agent_grouped: List[List[Sample]],
    opponent_grouped: List[List[Sample]],
    tool_error_penalty: float,
) -> Dict[str, float]:
    """Post-process opponent rewards with the per-prompt variance curriculum.

    Modifies opponent samples in place. Returns a dict of metrics for logging.
    """
    import numpy as np

    group_variances: List[float] = []
    group_curriculum_rewards: List[float] = []
    tool_error_count = 0

    for agent_group, opponent_group in zip(agent_grouped, opponent_grouped):
        agent_rewards = np.array(
            [float(getattr(s, "reward", 0.0) or 0.0) for s in agent_group],
            dtype=np.float64,
        )
        # Biased variance (ddof=0): matches SPICE's 1/K weighting and is
        # stable even when the group size is 1.
        var = float(np.var(agent_rewards)) if agent_rewards.size > 1 else 0.0
        curriculum_r = _variance_curriculum_reward(var)
        group_variances.append(var)
        group_curriculum_rewards.append(curriculum_r)

        for opp_sample in opponent_group:
            original = float(getattr(opp_sample, "reward", 0.0) or 0.0)
            new_r = curriculum_r
            meta = dict(getattr(opp_sample, "metadata", {}) or {})
            if meta.get("error") or meta.get("tau2_tool_error"):
                new_r = new_r + tool_error_penalty
                tool_error_count += 1
            meta["agent_group_variance"] = var
            meta["user_curriculum_reward"] = curriculum_r
            meta["user_reward_cooperative_original"] = original
            opp_sample.metadata = meta
            opp_sample.reward = new_r

    return {
        "user_curriculum/mean_variance": float(np.mean(group_variances)) if group_variances else 0.0,
        "user_curriculum/mean_reward": float(np.mean(group_curriculum_rewards)) if group_curriculum_rewards else 0.0,
        "user_curriculum/tool_error_count": float(tool_error_count),
        "user_curriculum/n_groups": float(len(group_variances)),
    }


def tau2_cotrain_generate_rollout(
    args: Namespace,
    rollout_id: int,
    data_source: Any,
    evaluation: bool = False,
) -> RolloutFnTrainOutput:
    """Batch-level tau2 co-training rollout function."""
    if evaluation:
        # Delegate to standard tau2 eval
        from usim.slime.rollout import _eval_rollout
        return _eval_rollout(args, rollout_id, data_source)

    samples = data_source.get_samples(args.rollout_batch_size)
    agent_grouped, opponent_grouped = asyncio.run(
        _run_tau2_cotrain_batch_async(args, samples)
    )

    metrics = compute_rollout_metrics(agent_grouped, rollout_id, prefix="TAU2_COTRAIN")

    # Optional: replace the opponent's cooperative reward with the per-prompt
    # variance curriculum (SPICE-style). Enabled by --tau2-user-reward-mode=curriculum.
    user_reward_mode = getattr(args, "tau2_user_reward_mode", "cooperative")
    if user_reward_mode == "curriculum":
        tool_error_penalty = float(getattr(args, "tau2_tool_error_penalty", -0.1))
        curriculum_metrics = _apply_user_curriculum_reward(
            agent_grouped, opponent_grouped, tool_error_penalty=tool_error_penalty
        )
        metrics = {**metrics, **curriculum_metrics}

    all_agent_samples = [s for group in agent_grouped for s in group]
    output_dir = getattr(args, "trajectory_output_dir", None)
    if output_dir:
        recorder = TrajectoryRecorder(output_dir)
        recorder.record_batch(all_agent_samples, rollout_id)

    docent_uploader = get_docent_uploader(args)
    if docent_uploader:
        docent_uploader.upload_batch(all_agent_samples, rollout_id)

    all_samples = agent_grouped + opponent_grouped
    return RolloutFnTrainOutput(samples=all_samples, metrics=metrics)
