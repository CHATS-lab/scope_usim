"""Main training entry point for USIM with Slime backend.

This script integrates USIM's orchestrator with Slime's training loop.
"""

import argparse
import logging
import sys
from typing import Any, Dict, List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def add_usim_arguments(parser: argparse.ArgumentParser) -> None:
    """Add USIM-specific arguments to the parser.

    Args:
        parser: Argument parser to add arguments to
    """
    group = parser.add_argument_group("usim", "User Simulator arguments")

    group.add_argument(
        "--trainable-role",
        type=str,
        default="user",
        choices=["agent", "user", "both"],
        help="Which role(s) to train: agent, user, or both",
    )

    group.add_argument(
        "--max-turns",
        type=int,
        default=30,
        help="Maximum conversation turns",
    )

    group.add_argument(
        "--usim-domain",
        type=str,
        default="retail",
        choices=["retail", "airline", "telecom"],
        help="Domain for user simulation",
    )

    group.add_argument(
        "--user-model",
        type=str,
        default=None,
        help="Optional separate model for user simulator (uses agent model if not set)",
    )

    group.add_argument(
        "--usim-temperature",
        type=float,
        default=0.7,
        help="Temperature for user simulator generation",
    )

    group.add_argument(
        "--usim-max-tokens",
        type=int,
        default=2048,
        help="Max tokens per user simulator generation",
    )


def main() -> None:
    """Main entry point for USIM training with Slime."""
    try:
        from slime.train_async import parse_args, train
    except ImportError:
        logger.error("slime package not found. Install with: pip install slime")
        sys.exit(1)

    # Parse arguments with USIM additions
    args = parse_args(add_custom_arguments=add_usim_arguments)

    # Configure USIM-specific settings
    args.rollout_function_path = args.rollout_function_path or "usim.slime.rollout.usim_generate_rollout"
    args.data_source_path = args.data_source_path or "usim.slime.data_source.get_tau2_samples"

    logger.info("=" * 60)
    logger.info("USIM Training Configuration")
    logger.info("=" * 60)
    logger.info(f"Trainable role: {args.trainable_role}")
    logger.info(f"Max turns: {args.max_turns}")
    logger.info(f"Domain: {args.usim_domain}")
    logger.info(f"Rollout function: {args.rollout_function_path}")
    logger.info(f"Data source: {args.data_source_path}")
    logger.info("=" * 60)

    # Start training
    train(args)


if __name__ == "__main__":
    main()
