"""Training entry point for Persuasion for Good with Slime backend.

Trains a persuader agent to convince people to donate to charity using RL.
Agent (trainable, SGLang) learns persuasion skills while the persuadee
(fixed, API) simulates a realistic human persona.
"""

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def add_p4g_train_arguments(parser: argparse.ArgumentParser) -> None:
    """Add P4G and USIM arguments to the parser."""
    # USIM common args (fixed opponent config)
    usim_group = parser.add_argument_group("usim", "User Simulator arguments")

    usim_group.add_argument(
        "--trainable-role",
        type=str,
        default="agent",
        choices=["agent", "user", "both"],
        help="Which role(s) to train (default: agent = persuader)",
    )

    usim_group.add_argument(
        "--max-turns",
        type=int,
        default=10,
        help="Maximum conversation turns (default: 10 for P4G)",
    )

    usim_group.add_argument(
        "--usim-fixed-opponent-model",
        type=str,
        default=None,
        help=(
            "Fixed persuadee model(s) via OpenAI-compatible API. "
            "Comma-separated for rotation (e.g., 'gpt-5-mini')"
        ),
    )

    usim_group.add_argument(
        "--usim-fixed-opponent-base-url",
        type=str,
        default="https://api.openai.com/v1",
        help="Base URL for fixed persuadee API",
    )

    usim_group.add_argument(
        "--usim-fixed-opponent-api-key-var",
        type=str,
        default="OPENAI_API_KEY",
        help="Environment variable name for API key",
    )

    # P4G-specific args
    from usim.p4g.rollout import add_p4g_arguments

    add_p4g_arguments(parser)

    return parser


def main() -> None:
    """Main entry point for P4G training with Slime."""
    try:
        from slime.train import parse_args, train
    except ImportError:
        logger.error("slime package not found. Install with: pip install slime")
        sys.exit(1)

    # Parse arguments with P4G additions
    args = parse_args(add_custom_arguments=add_p4g_train_arguments)

    # Configure P4G rollout and data source
    args.rollout_function_path = (
        args.rollout_function_path or "usim.p4g.rollout.p4g_generate_rollout"
    )
    args.data_source_path = args.data_source_path or "usim.p4g.data_source.get_p4g_data_source"

    # Override max_turns with p4g_num_turns if set
    if hasattr(args, "p4g_num_turns"):
        args.max_turns = args.p4g_num_turns

    logger.info("=" * 60)
    logger.info("P4G Training Configuration")
    logger.info("=" * 60)
    logger.info(f"Trainable role: {args.trainable_role}")
    logger.info(f"Max turns: {args.max_turns}")
    logger.info(f"Word limit: {getattr(args, 'p4g_word_limit', 50)}")
    logger.info(f"Corpus path: {getattr(args, 'p4g_corpus_path', '')}")
    logger.info(f"Dataset dir: {getattr(args, 'p4g_dataset_dir', '')}")
    logger.info(f"Rollout function: {args.rollout_function_path}")
    logger.info(f"Data source: {args.data_source_path}")
    logger.info("Agent (trainable): SGLang model (persuader)")
    fixed_opponent = getattr(args, "usim_fixed_opponent_model", None)
    if fixed_opponent:
        models = [m.strip() for m in fixed_opponent.split(",") if m.strip()]
        logger.info(f"Persuadee (fixed): {models} via {args.usim_fixed_opponent_base_url}")
    else:
        logger.info("Persuadee: same SGLang model (no fixed opponent)")
    logger.info("=" * 60)

    # Start training
    train(args)


if __name__ == "__main__":
    main()
