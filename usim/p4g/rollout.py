"""Slime rollout entry point for Persuasion for Good training.

Thin glue layer: creates the P4G environment, sglang generate function,
and calls the generic orchestrator. Converts the resulting Trajectory
to Slime's Sample format.

All P4G-specific logic (persuadee API calls, persona loading, reward
computation) lives in usim.core.environment.p4g.
"""

import logging
import os
from typing import Any, Dict, List

from slime.data.types import Sample
from slime.utils.http_utils import post
from slime.utils.processing_utils import load_tokenizer

from usim.core.environment.p4g import P4gEnvironment
from usim.core.orchestrator import UserSimOrchestrator
from usim.core.types import TrainableRole, UserSimConfig
from usim.p4g.persona import PersonaLoader
from usim.slime.trajectory_converter import trajectory_to_slime_sample


logger = logging.getLogger(__name__)

# Module-level cache for PersonaLoader (loaded once per worker process)
_persona_loader_cache: Dict[str, PersonaLoader] = {}


def _get_persona_loader(corpus_path: str) -> PersonaLoader:
    """Get or create a cached PersonaLoader."""
    if corpus_path not in _persona_loader_cache:
        _persona_loader_cache[corpus_path] = PersonaLoader(corpus_path)
    return _persona_loader_cache[corpus_path]


def _get_persuadee_config(args: Any, sample: Any) -> tuple:
    """Get persuadee model configuration from metadata overrides or CLI args.

    Returns:
        (model_name, base_url, api_key) tuple
    """
    metadata = getattr(sample, "metadata", {}) or {}

    # Check metadata overrides first (for eval)
    fixed_model = metadata.get("usim_fixed_opponent_model")
    base_url = metadata.get(
        "usim_fixed_opponent_base_url",
        getattr(args, "usim_fixed_opponent_base_url", "https://api.openai.com/v1"),
    )
    api_key_var = metadata.get(
        "usim_fixed_opponent_api_key_var",
        getattr(args, "usim_fixed_opponent_api_key_var", "OPENAI_API_KEY"),
    )

    # Fall back to CLI args for training
    if not fixed_model:
        fixed_model = getattr(args, "usim_fixed_opponent_model", None)

    if not fixed_model:
        return None, base_url, None

    # Model rotation: comma-separated list
    model_list = [m.strip() for m in fixed_model.split(",") if m.strip()]
    model_idx = sample.index % len(model_list)
    model_name = model_list[model_idx]

    # Per-model base URL and API key (comma-separated, maps 1:1)
    base_url_list = [u.strip() for u in base_url.split(",") if u.strip()]
    api_key_var_list = [k.strip() for k in api_key_var.split(",") if k.strip()]
    model_base_url = base_url_list[model_idx % len(base_url_list)]
    model_api_key_var = api_key_var_list[model_idx % len(api_key_var_list)]
    api_key = os.environ.get(model_api_key_var)

    logger.info(
        f"P4G persuadee: {model_name} (sample {sample.index}, "
        f"{len(model_list)} model(s) in rotation)"
    )
    return model_name, model_base_url, api_key


async def p4g_generate_rollout(
    args: Any,
    sample: Any,
    sampling_params: Dict[str, Any],
) -> List[Any]:
    """Generate a P4G rollout for Slime training.

    Args:
        args: Slime training arguments namespace
        sample: Slime Sample with P4G task metadata
        sampling_params: Sampling parameters for generation

    Returns:
        List of Slime Sample objects with trajectories
    """
    try:
        metadata = getattr(sample, "metadata", {}) or {}

        # --- P4G config ---
        num_turns = getattr(args, "p4g_num_turns", 10)
        word_limit = getattr(args, "p4g_word_limit", 50)
        num_exchanges = num_turns // 2

        # --- Load personas ---
        corpus_path = metadata.get(
            "corpus_path", getattr(args, "p4g_corpus_path", "")
        )
        persuader_speaker_id = metadata["persuader_speaker_id"]
        persuadee_speaker_id = metadata["persuadee_speaker_id"]

        persona_loader = _get_persona_loader(corpus_path)
        persuader_persona = persona_loader.get_persona_text(persuader_speaker_id)
        persuadee_persona = persona_loader.get_persona_text(persuadee_speaker_id)

        # --- Persuadee model config ---
        persuadee_model, persuadee_base_url, persuadee_api_key = (
            _get_persuadee_config(args, sample)
        )
        if not persuadee_model:
            raise ValueError(
                "P4G requires --usim-fixed-opponent-model for the persuadee"
            )

        # --- Create environment ---
        conversation_id = metadata.get("conversation_id", str(sample.index))
        env = P4gEnvironment(
            persuader_persona=persuader_persona,
            persuadee_persona=persuadee_persona,
            word_limit=word_limit,
            num_turns=num_turns,
            persuadee_model=persuadee_model,
            persuadee_base_url=persuadee_base_url,
            persuadee_api_key=persuadee_api_key,
            conversation_id=conversation_id,
        )

        # --- Create generate function (TITO: input_ids → token_ids + logprobs) ---
        sglang_url = (
            f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
        )

        async def generate_fn(
            input_ids: list, samp_params: Dict[str, Any]
        ) -> Dict[str, Any]:
            output = await post(sglang_url, {
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

        # --- Load tokenizer ---
        tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)

        # --- Create orchestrator and run ---
        config = UserSimConfig(
            trainable_role=TrainableRole(getattr(args, "trainable_role", "agent")),
            max_turns=num_exchanges,
            max_tokens=getattr(args, "max_tokens", 2048),
            max_context_length=getattr(args, "rollout_max_response_len", 4096),
            temperature=sampling_params.get("temperature", 0.7),
        )

        orchestrator = UserSimOrchestrator(tokenizer=tokenizer, config=config)
        trajectory = await orchestrator.rollout(env, generate_fn, sampling_params)

        logger.info(
            f"P4G rollout {sample.index}: {trajectory.turn_count} turns, "
            f"reward={trajectory.reward:.2f}, status={trajectory.status}"
        )

        # Convert to Slime sample
        output_sample = trajectory_to_slime_sample(trajectory, sample.index)
        return [output_sample]

    except Exception as e:
        logger.error(f"P4G rollout failed: {e}", exc_info=True)
        return [
            Sample(
                index=sample.index,
                prompt=sample.prompt,
                tokens=[],
                response="",
                reward=0.0,
                loss_mask=[],
                response_length=0,
                metadata={"error": str(e)},
            )
        ]


def add_p4g_arguments(parser: Any) -> None:
    """Add P4G-specific arguments to argparse parser."""
    group = parser.add_argument_group("p4g", "Persuasion for Good arguments")

    group.add_argument(
        "--p4g-corpus-path",
        type=str,
        default="persuasion_simulation/src/datasets/donation/persuasionforgood_corpus",
        help="Path to convokit Corpus for persona loading",
    )

    group.add_argument(
        "--p4g-dataset-dir",
        type=str,
        default="persuasion_simulation/src/datasets/donation/persuasionforgood_train",
        help="Path to directory with dialogue JSONL files",
    )

    group.add_argument(
        "--p4g-word-limit",
        type=int,
        default=50,
        help="Max words per response in prompts (default: 50)",
    )

    group.add_argument(
        "--p4g-num-turns",
        type=int,
        default=10,
        help="Total conversation turns, each side gets num_turns//2 (default: 10)",
    )
