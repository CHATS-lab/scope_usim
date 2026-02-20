"""Training entry point for CooperBench with Slime backend.

Trains coding agents to collaborate via RL using CooperBench benchmark.
Agent (trainable, SGLang) learns to implement features while coordinating
with a partner agent (fixed, API) to avoid merge conflicts.
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
        "--cooperbench-subset",
        type=str,
        default=None,
        help="CooperBench subset: lite, flash, or None for all",
    )

    group.add_argument(
        "--cooperbench-repo",
        type=str,
        default=None,
        help="Filter by repository name",
    )

    group.add_argument(
        "--cooperbench-partner-model",
        type=str,
        default="gpt-5-mini",
        help="LLM model for the partner agent (default: gpt-5-mini)",
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
        help="Redis URL for inter-agent messaging",
    )

    group.add_argument(
        "--cooperbench-dataset-dir",
        type=str,
        default=None,
        help="Path to CooperBench directory (with dataset/ subdirectory)",
    )

    group.add_argument(
        "--cooperbench-partial-reward",
        action="store_true",
        default=False,
        help="Give 0.5 reward when agent's feature passes but merge fails",
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
        args.rollout_function_path or "usim.cooperbench.rollout.cooperbench_generate_rollout"
    )
    args.data_source_path = (
        args.data_source_path or "usim.cooperbench.data_source.get_cooperbench_samples"
    )

    logger.info("=" * 60)
    logger.info("CooperBench Training Configuration")
    logger.info("=" * 60)
    logger.info(f"Trainable role: {args.trainable_role}")
    logger.info(f"Subset: {getattr(args, 'cooperbench_subset', None)}")
    logger.info(f"Repo filter: {getattr(args, 'cooperbench_repo', None)}")
    logger.info(f"Partner model: {getattr(args, 'cooperbench_partner_model', 'gpt-5-mini')}")
    logger.info(f"Max steps: {getattr(args, 'cooperbench_max_steps', 50)}")
    logger.info(f"Redis URL: {getattr(args, 'cooperbench_redis_url', 'redis://localhost:6379')}")
    logger.info(f"Rollout function: {args.rollout_function_path}")
    logger.info(f"Data source: {args.data_source_path}")
    logger.info("Agent (trainable): SGLang model")
    logger.info(f"Partner (fixed): {getattr(args, 'cooperbench_partner_model', 'gpt-5-mini')}")
    logger.info("=" * 60)

    # Start training
    train(args)


if __name__ == "__main__":
    main()
