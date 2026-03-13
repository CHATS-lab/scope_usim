"""Co-training rollout function for Slime.

Routes generation requests to named SGLang models via get_model_url():
  - "actor" → agent (persuader) SGLang engine
  - "opponent" → opponent (persuadee) SGLang engine

Produces dual samples (agent + opponent) from each conversation.
"""

import asyncio
import logging
from argparse import Namespace
from typing import Any, Dict, List

import weave
from slime.rollout.base_types import RolloutFnTrainOutput
from slime.rollout.sglang_rollout import get_model_url
from slime.utils.http_utils import post
from slime.utils.processing_utils import load_tokenizer
from slime.utils.types import Sample

from usim.core.co_training_orchestrator import CoTrainingOrchestrator
from usim.core.environment.p4g import P4gEnvironment
from usim.core.rollout_metrics import compute_rollout_metrics
from usim.core.trajectory_recorder import TrajectoryRecorder
from usim.core.types import TrainableRole, UserSimConfig
from usim.p4g.persona import PersonaLoader
from usim.slime.trajectory_converter import trajectory_to_slime_sample
from usim.utils.docent import get_docent_uploader


logger = logging.getLogger(__name__)

_persona_loader_cache: Dict[str, PersonaLoader] = {}


def _get_persona_loader(corpus_path: str) -> PersonaLoader:
    if corpus_path not in _persona_loader_cache:
        _persona_loader_cache[corpus_path] = PersonaLoader(corpus_path)
    return _persona_loader_cache[corpus_path]


def _make_sglang_generate_fn(url: str):
    """Create an async generate function that calls an SGLang endpoint."""

    @weave.op
    async def generate_fn(
        input_ids: list, samp_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        output = await post(url, {
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

    return generate_fn


def _make_opponent_generate_fn(url: str, tokenizer: Any):
    """Create opponent generate function that tokenizes messages and calls SGLang.

    The P4G environment passes persuadee_messages (OpenAI format) to this fn.
    We tokenize them, call SGLang, and return the output.
    """

    async def opponent_generate_fn(
        messages: List[Dict[str, Any]], samp_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        # Tokenize messages for SGLang (TITO)
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = tokenizer(text, add_special_tokens=False)["input_ids"]

        output = await post(url, {
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

    return opponent_generate_fn


async def _cotrain_generate_single(
    args: Any,
    sample: Any,
    sampling_params: Dict[str, Any],
) -> tuple:
    """Per-sample co-training rollout. Returns (agent_sample, opponent_sample)."""
    try:
        metadata = getattr(sample, "metadata", {}) or {}

        num_turns = getattr(args, "p4g_num_turns", 10)
        word_limit = getattr(args, "p4g_word_limit", 50)
        num_exchanges = num_turns // 2

        corpus_path = metadata.get(
            "corpus_path", getattr(args, "p4g_corpus_path", "")
        )
        persuader_speaker_id = metadata["persuader_speaker_id"]
        persuadee_speaker_id = metadata["persuadee_speaker_id"]

        persona_loader = _get_persona_loader(corpus_path)
        persuader_persona = persona_loader.get_persona_text(persuader_speaker_id)
        persuadee_persona = persona_loader.get_persona_text(persuadee_speaker_id)

        conversation_id = metadata.get("conversation_id", str(sample.index))

        # Build generate functions for both models via sglang-config
        agent_url = get_model_url(args, "actor", "/generate")
        opponent_url = get_model_url(args, "opponent", "/generate")

        agent_generate_fn = _make_sglang_generate_fn(agent_url)

        # Load tokenizers for both models
        agent_hf = getattr(args, "hf_checkpoint", None)
        opponent_hf = getattr(args, "opponent_hf_checkpoint", None) or agent_hf
        agent_tokenizer = load_tokenizer(agent_hf, trust_remote_code=True)
        opponent_tokenizer = load_tokenizer(opponent_hf, trust_remote_code=True)

        opponent_generate_fn = _make_opponent_generate_fn(opponent_url, opponent_tokenizer)

        raw_prefix = getattr(args, "p4g_persuadee_prompt_prefix", []) or []
        persuadee_prompt_prefix = " ".join(raw_prefix) if isinstance(raw_prefix, list) else (raw_prefix or "")

        env = P4gEnvironment(
            persuader_persona=persuader_persona,
            persuadee_persona=persuadee_persona,
            word_limit=word_limit,
            num_turns=num_turns,
            conversation_id=conversation_id,
            persuadee_prompt_prefix=persuadee_prompt_prefix,
            opponent_generate_fn=opponent_generate_fn,
        )

        config = UserSimConfig(
            trainable_role=TrainableRole(getattr(args, "trainable_role", "agent")),
            max_turns=num_exchanges,
            max_tokens=getattr(args, "max_tokens", 8192),
            max_context_length=getattr(args, "rollout_max_response_len", 32768),
            temperature=sampling_params.get("temperature", 0.7),
        )

        orchestrator = CoTrainingOrchestrator(
            agent_tokenizer=agent_tokenizer,
            opponent_tokenizer=opponent_tokenizer,
            config=config,
        )
        agent_traj, opponent_traj = await orchestrator.rollout(
            env, agent_generate_fn, sampling_params
        )

        logger.info(
            f"Cotrain rollout {sample.index}: {agent_traj.turn_count} turns, "
            f"reward={agent_traj.reward:.2f}, status={agent_traj.status}"
        )

        agent_sample = trajectory_to_slime_sample(agent_traj, sample.index, role="actor")
        opponent_sample = trajectory_to_slime_sample(opponent_traj, sample.index, role="opponent")
        return agent_sample, opponent_sample

    except Exception as e:
        logger.error(f"Cotrain rollout failed (sample {sample.index}): {e}", exc_info=True)
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


async def _run_cotrain_batch_async(
    args: Namespace,
    samples: List[List[Sample]],
) -> tuple:
    """Run co-training batch. Returns (agent_grouped, opponent_grouped)."""
    sampling_params = dict(
        temperature=args.rollout_temperature,
        top_p=getattr(args, "rollout_top_p", 0.95),
        top_k=getattr(args, "rollout_top_k", -1),
        max_new_tokens=8192,
    )

    tasks = []
    for group in samples:
        for sample in group:
            tasks.append(_cotrain_generate_single(args, sample, sampling_params))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Re-group results into agent and opponent sample groups
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


def cotrain_generate_rollout(
    args: Namespace,
    rollout_id: int,
    data_source: Any,
    evaluation: bool = False,
) -> RolloutFnTrainOutput:
    """Batch-level co-training rollout function.

    Returns RolloutFnTrainOutput where samples contain both agent and
    opponent data. The cotrain_loop separates them for training.
    """
    if evaluation:
        # For eval, only agent trajectory matters
        return RolloutFnTrainOutput(samples=[], metrics={})

    samples = data_source.get_samples(args.rollout_batch_size)
    agent_grouped, opponent_grouped = asyncio.run(
        _run_cotrain_batch_async(args, samples)
    )

    # Compute metrics from agent samples
    metrics = compute_rollout_metrics(agent_grouped, rollout_id, prefix="COTRAIN")

    # Record trajectories
    all_agent_samples = [s for group in agent_grouped for s in group]
    output_dir = getattr(args, "trajectory_output_dir", None)
    if output_dir:
        recorder = TrajectoryRecorder(output_dir)
        recorder.record_batch(all_agent_samples, rollout_id)

    docent_uploader = get_docent_uploader(args)
    if docent_uploader:
        docent_uploader.upload_batch(all_agent_samples, rollout_id)

    # Combine agent and opponent samples into one training batch.
    # Each set keeps its own prompt groups (GRPO computes advantages per group).
    # Agent groups come first, then opponent groups.
    all_samples = agent_grouped + opponent_grouped

    return RolloutFnTrainOutput(samples=all_samples, metrics=metrics)
