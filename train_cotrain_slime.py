"""Training entry point for multi-agent co-training and self-play with Slime.

Supports five training modes:
  single        - 1 trainable model + fixed API opponent (delegates to Slime)
  cotrain       - 1 training group, dual-trajectory rollout (both sides trained)
  selfplay      - 1 training group + opponent checkpoint pool
  dual_cotrain  - 2 training groups, colocated NCCL/IPC weight sync
  dual_selfplay - 2 training groups + opponent checkpoint pool

Single-model modes (cotrain/selfplay) use one training group for both roles.
Dual-model modes use two independent training groups with FilteredRolloutProxy
for per-model NCCL/IPC weight sync (requires slime_per_server_engines.patch).

Usage:
  # Single training group co-training
  python train_cotrain_slime.py --training-mode cotrain \
    --sglang-config configs/sglang/cotrain_2plus2.yaml ...

  # True dual-model co-training (colocated, 4+4 GPUs)
  python train_cotrain_slime.py --training-mode dual_cotrain \
    --sglang-config configs/sglang/cotrain_4plus4.yaml \
    --actor-num-gpus-per-node 4 ...

  # Dual-model self-play with checkpoint pool
  python train_cotrain_slime.py --training-mode dual_selfplay \
    --sglang-config configs/sglang/cotrain_4plus4.yaml \
    --pool-dir /checkpoints/selfplay_pool ...
"""

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def add_cotrain_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add co-training / self-play arguments to the parser."""
    # Training mode
    mode_group = parser.add_argument_group("cotrain", "Co-training / self-play arguments")

    mode_group.add_argument(
        "--training-mode",
        type=str,
        default="cotrain",
        choices=["single", "cotrain", "selfplay", "dual_cotrain", "dual_selfplay"],
        help=(
            "Training mode: single (1 model + API), "
            "cotrain (1 training group, dual trajectories), "
            "selfplay (1 training group + pool), "
            "dual_cotrain (2 training groups, colocated NCCL), "
            "dual_selfplay (2 training groups + pool)"
        ),
    )

    mode_group.add_argument(
        "--opponent-hf-checkpoint",
        type=str,
        default=None,
        help="HuggingFace checkpoint for opponent model (defaults to --hf-checkpoint)",
    )

    mode_group.add_argument(
        "--opponent-num-gpus",
        type=int,
        default=None,
        help="Number of GPUs for opponent (defaults to actor GPU count)",
    )

    mode_group.add_argument(
        "--opponent-num-nodes",
        type=int,
        default=1,
        help="Number of nodes for opponent training group",
    )

    mode_group.add_argument(
        "--opponent-num-gpus-per-node",
        type=int,
        default=None,
        help="GPUs per node for opponent training group",
    )

    # Self-play checkpoint pool
    mode_group.add_argument(
        "--pool-dir",
        type=str,
        default=None,
        help="Directory for self-play checkpoint pool (required for selfplay mode)",
    )

    mode_group.add_argument(
        "--pool-size",
        type=int,
        default=10,
        help="Maximum number of checkpoints in the pool (default: 10)",
    )

    mode_group.add_argument(
        "--pool-save-interval",
        type=int,
        default=16,
        help="Save opponent checkpoint to pool every N rollouts (default: 16)",
    )

    mode_group.add_argument(
        "--pool-selection",
        type=str,
        default="random",
        choices=["random", "latest"],
        help="How to select opponent from pool: random or latest (default: random)",
    )

    # USIM common args
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
        default=10,
        help="Maximum conversation turns (default: 10)",
    )

    usim_group.add_argument(
        "--usim-fixed-opponent-model",
        type=str,
        default=None,
        help="Fixed opponent model(s) via API (for single mode only)",
    )

    usim_group.add_argument(
        "--usim-fixed-opponent-base-url",
        type=str,
        default="https://api.openai.com/v1",
        help="Base URL for fixed opponent API",
    )

    usim_group.add_argument(
        "--usim-fixed-opponent-api-key-var",
        type=str,
        default="OPENAI_API_KEY",
        help="Environment variable name for API key",
    )

    # P4G-specific args (environment config)
    from usim.p4g.rollout import add_p4g_arguments

    add_p4g_arguments(parser)

    return parser


def main() -> None:
    """Main entry point for co-training / self-play."""
    try:
        from slime.utils.arguments import parse_args
    except ImportError:
        logger.error("slime package not found. Install with: pip install slime")
        sys.exit(1)

    args = parse_args(add_custom_arguments=add_cotrain_arguments)

    training_mode = getattr(args, "training_mode", "cotrain")

    # Set defaults based on mode
    if training_mode == "single":
        # Delegate to standard single-model training
        args.rollout_function_path = (
            args.rollout_function_path or "usim.p4g.rollout.p4g_generate_rollout"
        )
        args.data_source_path = (
            args.data_source_path or "usim.p4g.data_source.get_p4g_data_source"
        )

        from slime.train_async import train

        logger.info("[COTRAIN] Mode: single — delegating to standard training loop")
        train(args)
        return

    # Co-training or self-play
    args.rollout_function_path = (
        args.rollout_function_path or "usim.slime.cotrain_rollout.cotrain_generate_rollout"
    )
    args.data_source_path = (
        args.data_source_path or "usim.p4g.data_source.get_p4g_data_source"
    )

    # Log configuration
    logger.info("=" * 60)
    logger.info(f"Multi-Agent Training Configuration (mode={training_mode})")
    logger.info("=" * 60)
    logger.info(f"Model: {args.hf_checkpoint}")
    logger.info(f"Training GPUs: {args.actor_num_gpus_per_node * args.actor_num_nodes}")
    logger.info(f"Rollout GPUs: {args.rollout_num_gpus}")
    logger.info(f"Rollout function: {args.rollout_function_path}")
    logger.info(f"Data source: {args.data_source_path}")

    if training_mode in ("dual_cotrain", "dual_selfplay"):
        opp_hf = getattr(args, "opponent_hf_checkpoint", None) or args.hf_checkpoint
        logger.info(f"Opponent model: {opp_hf}")
        logger.info(f"Opponent GPUs/node: {getattr(args, 'opponent_num_gpus_per_node', args.actor_num_gpus_per_node)}")

    if training_mode in ("selfplay", "dual_selfplay"):
        pool_dir = getattr(args, "pool_dir", None)
        if not pool_dir:
            logger.error(f"{training_mode} mode requires --pool-dir")
            sys.exit(1)
        logger.info(f"Pool dir: {pool_dir}")
        logger.info(f"Pool size: {getattr(args, 'pool_size', 10)}")
        logger.info(f"Pool save interval: {getattr(args, 'pool_save_interval', 16)}")
        logger.info(f"Pool selection: {getattr(args, 'pool_selection', 'random')}")

    logger.info("=" * 60)

    from usim.slime.cotrain_loop import (
        cotrain,
        dual_cotrain,
        dual_selfplay,
        selfplay,
    )

    dispatch = {
        "cotrain": cotrain,
        "selfplay": selfplay,
        "dual_cotrain": dual_cotrain,
        "dual_selfplay": dual_selfplay,
    }

    fn = dispatch.get(training_mode)
    if fn is None:
        logger.error(f"Unknown training mode: {training_mode}")
        sys.exit(1)
    fn(args)


if __name__ == "__main__":
    main()
