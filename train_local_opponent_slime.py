"""Training entry point with locally-served opponent via dual SGLang servers.

Starts two SGLang servers on separate GPU groups:
- Agent server: trainable model (GRPO training)
- Opponent server: fixed model (inference-only)

The opponent's SGLang endpoint replaces the external API (OpenAI/OpenRouter),
removing API latency and cost while enabling future co-training.

Supports both tau2-bench and P4G — select via --rollout-function-path and
--data-source-path (or use the defaults from train_usim_slime / train_p4g_slime).
"""

import argparse
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def add_local_opponent_arguments(parser: argparse.ArgumentParser) -> None:
    """Add local opponent and USIM arguments to the parser."""
    # --- Opponent server args ---
    opp_group = parser.add_argument_group("opponent", "Local opponent server arguments")

    opp_group.add_argument(
        "--opponent-model-path",
        type=str,
        required=True,
        help="HF checkpoint path for the opponent model",
    )

    opp_group.add_argument(
        "--opponent-num-gpus",
        type=int,
        default=2,
        help="Number of GPUs for the opponent SGLang server (default: 2)",
    )

    opp_group.add_argument(
        "--opponent-sglang-tp-size",
        type=int,
        default=1,
        help="Tensor parallel size for opponent SGLang (default: 1)",
    )

    opp_group.add_argument(
        "--opponent-sglang-mem-fraction",
        type=float,
        default=None,
        help="Memory fraction for opponent SGLang (default: same as agent)",
    )

    # --- USIM common args ---
    usim_group = parser.add_argument_group("usim", "User Simulator arguments")

    usim_group.add_argument(
        "--trainable-role",
        type=str,
        default="agent",
        choices=["agent", "user", "both"],
        help="Which role(s) to train (default: agent)",
    )

    usim_group.add_argument(
        "--max-turns",
        type=int,
        default=30,
        help="Maximum conversation turns",
    )

    usim_group.add_argument(
        "--usim-fixed-opponent-model",
        type=str,
        default=None,
        help=(
            "Model name for the opponent (used in litellm calls). "
            "Defaults to the HF checkpoint name from --opponent-model-path"
        ),
    )

    usim_group.add_argument(
        "--usim-fixed-opponent-base-url",
        type=str,
        default=None,
        help="Overridden automatically by the local opponent server",
    )

    usim_group.add_argument(
        "--usim-fixed-opponent-api-key-var",
        type=str,
        default=None,
        help="Not needed for local opponent (set automatically)",
    )

    usim_group.add_argument(
        "--usim-domain",
        type=str,
        default="retail",
        choices=["retail", "airline", "telecom"],
        help="Domain for tau2-bench (default: retail)",
    )

    usim_group.add_argument(
        "--trajectory-output-dir",
        type=str,
        default=None,
        help="Directory to save trajectory JSONL files (disabled if not set)",
    )
    usim_group.add_argument(
        "--docent-collection",
        type=str,
        default=None,
        help="Docent collection name; enables upload when set (requires docent-python)",
    )
    usim_group.add_argument(
        "--docent-upload-interval",
        type=int,
        default=1,
        help="Upload to Docent every N rollouts (default: 1)",
    )

    # --- P4G args (optional, only used when rollout function is P4G) ---
    from usim.p4g.rollout import add_p4g_arguments

    add_p4g_arguments(parser)



def main() -> None:
    """Main entry point for training with a locally-served opponent."""
    try:
        from slime.utils.arguments import parse_args
    except ImportError:
        logger.error("slime package not found. Install with: pip install slime")
        sys.exit(1)

    args = parse_args(add_custom_arguments=add_local_opponent_arguments)

    # Default opponent model name to the HF checkpoint basename
    if not args.usim_fixed_opponent_model:
        args.usim_fixed_opponent_model = os.path.basename(
            args.opponent_model_path.rstrip("/")
        )

    # Default rollout/data source paths (tau2 by default, override for P4G)
    args.rollout_function_path = (
        args.rollout_function_path or "usim.slime.rollout.usim_generate_rollout"
    )
    args.data_source_path = (
        args.data_source_path or "usim.slime.data_source.get_tau2_data_source"
    )

    # Override max_turns with p4g_num_turns if set (P4G convention)
    if hasattr(args, "p4g_num_turns"):
        args.max_turns = args.p4g_num_turns

    logger.info("=" * 60)
    logger.info("Local Opponent Training Configuration")
    logger.info("=" * 60)
    logger.info(f"Agent model: {args.hf_checkpoint}")
    logger.info(f"Opponent model: {args.opponent_model_path}")
    logger.info(f"Opponent model name: {args.usim_fixed_opponent_model}")
    logger.info(f"Agent rollout GPUs: {args.rollout_num_gpus}")
    logger.info(f"Opponent GPUs: {args.opponent_num_gpus}")
    logger.info(f"Opponent TP size: {args.opponent_sglang_tp_size}")
    logger.info(f"Trainable role: {args.trainable_role}")
    logger.info(f"Max turns: {args.max_turns}")
    logger.info(f"Rollout function: {args.rollout_function_path}")
    logger.info(f"Data source: {args.data_source_path}")
    logger.info("=" * 60)

    from usim.slime.dual_server_loop import train_with_local_opponent

    train_with_local_opponent(args)


if __name__ == "__main__":
    main()
