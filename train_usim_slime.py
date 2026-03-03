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
        default="agent",
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

    group.add_argument(
        "--usim-fixed-opponent-model",
        type=str,
        default=None,
        help=(
            "Fixed user sim model(s) via OpenAI-compatible API. "
            "Comma-separated for rotation (e.g., 'gpt-5-mini' or "
            "'gpt-5-mini,gpt-4o-mini,deepseek-v3.2')"
        ),
    )

    group.add_argument(
        "--usim-fixed-opponent-base-url",
        type=str,
        default="https://api.openai.com/v1",
        help="Base URL for fixed user sim API",
    )

    group.add_argument(
        "--usim-fixed-opponent-api-key-var",
        type=str,
        default="OPENAI_API_KEY",
        help="Environment variable name for fixed user sim API key",
    )

    group.add_argument(
        "--trajectory-output-dir",
        type=str,
        default=None,
        help="Directory to save trajectory JSONL files (disabled if not set)",
    )
    group.add_argument(
        "--docent-collection",
        type=str,
        default=None,
        help="Docent collection name; enables upload when set (requires docent-python)",
    )
    group.add_argument(
        "--docent-upload-interval",
        type=int,
        default=1,
        help="Upload to Docent every N rollouts (default: 1)",
    )

    return parser


def main() -> None:
    """Main entry point for USIM training with Slime."""
    try:
        from slime.train import train
        from slime.utils.arguments import parse_args
    except ImportError as e:
        logger.error(f"slime import failed: {e}", exc_info=True)
        sys.exit(1)

    # Parse arguments with USIM additions
    args = parse_args(add_custom_arguments=add_usim_arguments)

    # Configure USIM-specific settings
    args.rollout_function_path = args.rollout_function_path or "usim.slime.rollout.usim_generate_rollout"
    args.data_source_path = args.data_source_path or "usim.slime.data_source.get_tau2_data_source"

    logger.info("=" * 60)
    logger.info("USIM Training Configuration")
    logger.info("=" * 60)
    logger.info(f"Trainable role: {args.trainable_role}")
    logger.info(f"Max turns: {args.max_turns}")
    logger.info(f"Domain: {args.usim_domain}")
    logger.info(f"Rollout function: {args.rollout_function_path}")
    logger.info(f"Data source: {args.data_source_path}")
    logger.info("Agent (trainable): SGLang model")
    fixed_opponent = getattr(args, "usim_fixed_opponent_model", None)
    if fixed_opponent:
        models = [m.strip() for m in fixed_opponent.split(",") if m.strip()]
        logger.info(f"User sim (fixed): {models} via {args.usim_fixed_opponent_base_url}")
    else:
        logger.info("User sim: same SGLang model (no fixed opponent)")
    logger.info("=" * 60)

    # Start training
    train(args)


if __name__ == "__main__":
    main()
