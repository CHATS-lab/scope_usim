"""Training entry point for CooperBench with Slime backend.

Trains coding agents via RL (GRPO) using CooperBench benchmark.
Supports three settings:
- baseline: 1 agent, 1 feature
- solo: 1 agent, 2 features
- coop: 2 agents (trainable + fixed partner), 1 feature each
"""

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def add_cooperbench_arguments(parser: argparse.ArgumentParser) -> None:
    """Add CooperBench-specific arguments to the parser."""
    group = parser.add_argument_group("cooperbench", "CooperBench arguments")

    group.add_argument(
        "--trainable-role",
        type=str,
        default="agent",
        choices=["agent", "user", "both"],
        help="Which role(s) to train (default: agent)",
    )

    group.add_argument(
        "--cooperbench-setting",
        type=str,
        default="solo",
        choices=["baseline", "solo", "coop"],
        help="Training setting: baseline (1 agent, 1 feature), "
             "solo (1 agent, 2 features), coop (2 agents) (default: solo)",
    )

    group.add_argument(
        "--cooperbench-data-dir",
        type=str,
        default=None,
        help="Path to usim/data/cooperbench/ directory with JSON files",
    )

    group.add_argument(
        "--cooperbench-backend",
        type=str,
        default="modal",
        help="CooperBench sandbox backend: modal, docker (default: modal)",
    )

    group.add_argument(
        "--cooperbench-partner-model",
        type=str,
        default="gpt-5-mini",
        help="LLM model for the partner agent in coop mode (default: gpt-5-mini)",
    )

    group.add_argument(
        "--cooperbench-max-steps",
        type=int,
        default=50,
        help="Maximum steps per agent (default: 50)",
    )

    group.add_argument(
        "--cooperbench-redis-url",
        type=str,
        default="redis://localhost:6379",
        help="Redis URL for inter-agent messaging in coop mode",
    )

    group.add_argument(
        "--cooperbench-dataset-dir",
        type=str,
        default=None,
        help="Path to CooperBench repo (with dataset/ subdirectory) for eval",
    )

    group.add_argument(
        "--cooperbench-partial-reward",
        action="store_true",
        default=False,
        help="Give 0.5 reward when agent's feature passes but merge fails (coop only)",
    )


def main() -> None:
    """Main entry point for CooperBench training with Slime."""
    try:
        from slime.train_async import parse_args, train
    except ImportError:
        logger.error("slime package not found. Install with: pip install slime")
        sys.exit(1)

    # Parse arguments with CooperBench additions
    args = parse_args(add_custom_arguments=add_cooperbench_arguments)

    # Configure CooperBench rollout and data source
    args.rollout_function_path = (
        args.rollout_function_path
        or "usim.cooperbench.rollout.cooperbench_generate_rollout"
    )
    args.data_source_path = (
        args.data_source_path
        or "usim.cooperbench.data_source.get_cooperbench_data_source"
    )

    setting = getattr(args, "cooperbench_setting", "solo")

    logger.info("=" * 60)
    logger.info("CooperBench Training Configuration")
    logger.info("=" * 60)
    logger.info(f"Setting: {setting}")
    logger.info(f"Trainable role: {args.trainable_role}")
    logger.info(f"Data dir: {getattr(args, 'cooperbench_data_dir', None)}")
    logger.info(f"Backend: {getattr(args, 'cooperbench_backend', 'modal')}")
    logger.info(f"Max steps: {getattr(args, 'cooperbench_max_steps', 50)}")
    if setting == "coop":
        logger.info(f"Partner model: {getattr(args, 'cooperbench_partner_model', 'gpt-5-mini')}")
        logger.info(f"Redis URL: {getattr(args, 'cooperbench_redis_url', 'redis://localhost:6379')}")
        logger.info(f"Partial reward: {getattr(args, 'cooperbench_partial_reward', False)}")
    logger.info(f"Rollout function: {args.rollout_function_path}")
    logger.info(f"Data source: {args.data_source_path}")
    logger.info("=" * 60)

    # Start training
    train(args)


if __name__ == "__main__":
    main()
