"""Rollout function for P4G with Slime backend.

Reuses UserSimOrchestrator — P4G is a pure conversation with no tools/environment.
"""

import logging
import os
from typing import Any, Dict, List

from usim.core.orchestrator import UserSimOrchestrator
from usim.core.types import TrainableRole, UserSimConfig
from usim.p4g.agent import PersuaderAgent
from usim.p4g.persona import PersonaLoader
from usim.p4g.reward import compute_p4g_reward
from usim.p4g.user_simulator import PersuadeeUserSimulator
from usim.slime.model_adapter import create_slime_model_adapter
from usim.slime.trajectory_converter import trajectory_to_slime_sample

logger = logging.getLogger(__name__)

# Module-level cache for PersonaLoader (loaded once per worker process)
_persona_loader_cache: Dict[str, PersonaLoader] = {}


def _get_persona_loader(corpus_path: str) -> PersonaLoader:
    """Get or create a cached PersonaLoader."""
    if corpus_path not in _persona_loader_cache:
        _persona_loader_cache[corpus_path] = PersonaLoader(corpus_path)
    return _persona_loader_cache[corpus_path]


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
        # Agent model = trainable, served by SGLang
        agent_model = create_slime_model_adapter(args)

        # User sim model = fixed opponent via API
        fixed_usim_models = getattr(args, "usim_fixed_opponent_model", None)
        if fixed_usim_models:
            from usim.core.api_model_adapter import create_openai_model_adapter

            base_url = getattr(args, "usim_fixed_opponent_base_url", "https://api.openai.com/v1")
            api_key_var = getattr(args, "usim_fixed_opponent_api_key_var", "OPENAI_API_KEY")
            api_key = os.environ.get(api_key_var)

            model_list = [m.strip() for m in fixed_usim_models.split(",") if m.strip()]
            model_name = model_list[sample.index % len(model_list)]

            user_model = create_openai_model_adapter(
                model_name=model_name,
                tokenizer=agent_model.tokenizer,
                base_url=base_url,
                api_key=api_key,
            )
            logger.info(
                f"P4G fixed persuadee: {model_name} "
                f"(sample {sample.index}, {len(model_list)} model(s) in rotation)"
            )
        else:
            user_model = create_slime_model_adapter(args)

        # P4G config from args
        trainable_role = TrainableRole(getattr(args, "trainable_role", "agent"))
        max_turns = getattr(args, "p4g_num_turns", 10)
        max_tokens = getattr(args, "max_tokens", 2048)
        word_limit = getattr(args, "p4g_word_limit", 50)

        config = UserSimConfig(
            trainable_role=trainable_role,
            max_turns=max_turns,
            max_tokens=max_tokens,
            temperature=sampling_params.get("temperature", 0.7),
        )

        # Load personas
        metadata = getattr(sample, "metadata", {}) or {}
        corpus_path = metadata.get("corpus_path", getattr(args, "p4g_corpus_path", ""))
        persuader_speaker_id = metadata["persuader_speaker_id"]
        persuadee_speaker_id = metadata["persuadee_speaker_id"]

        persona_loader = _get_persona_loader(corpus_path)
        persuader_persona = persona_loader.get_persona_text(persuader_speaker_id)
        persuadee_persona = persona_loader.get_persona_text(persuadee_speaker_id)

        # Create agent and user simulator
        agent = PersuaderAgent(
            persona_text=persuader_persona,
            word_limit=word_limit,
            num_turns=max_turns,
        )
        user_simulator = PersuadeeUserSimulator(
            persona_text=persuadee_persona,
            word_limit=word_limit,
        )

        # Create orchestrator — reuses existing UserSimOrchestrator
        orchestrator = UserSimOrchestrator(
            agent_model=agent_model,
            user_model=user_model,
            config=config,
        )

        # Build task dict
        task = {
            "id": metadata.get("conversation_id", str(sample.index)),
            "domain": "p4g",
            "instructions": "",
            "domain_policy": "",
        }

        # Run session
        trajectory = await orchestrator.run_session(task, agent, user_simulator)

        # Compute reward from donation amount
        reward = compute_p4g_reward(trajectory.messages)
        trajectory.reward = reward

        logger.info(
            f"P4G rollout {sample.index}: {trajectory.turn_count} turns, "
            f"reward={reward:.2f}, status={trajectory.status}"
        )

        # Convert to Slime sample
        output_sample = trajectory_to_slime_sample(trajectory, sample.index)
        return [output_sample]

    except Exception as e:
        logger.error(f"P4G rollout failed: {e}")
        try:
            from slime.data.types import Sample

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
        except ImportError:
            raise


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
        help="Number of conversation turns (default: 10)",
    )
