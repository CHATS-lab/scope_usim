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


def add_cooperbench_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
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

    group.add_argument(
        "--cooperbench-tool-call-parser",
        type=str,
        default="qwen3_coder",
        help="sglang tool call parser for the training model's format: "
             "qwen3_coder (Qwen3.5), qwen25 (Qwen2.5), llama3, mistral (default: qwen3_coder)",
    )

    group.add_argument(
        "--cooperbench-sample-timeout",
        type=int,
        default=5400,
        help="Per-sample timeout in seconds (default: 5400 = sandbox 3600 + reward eval buffer)",
    )

    group.add_argument(
        "--cooperbench-max-context-length",
        type=int,
        default=65536,
        help="Max context length for coding agent (default: 65536)",
    )

    group.add_argument(
        "--cooperbench-enable-thinking",
        action="store_true",
        default=False,
        help="Enable Qwen3.5 thinking mode (default: disabled for tool calling)",
    )

    group.add_argument(
        "--cooperbench-max-concurrent",
        type=int,
        default=16,
        help="Max concurrent rollout sessions to avoid SGLang OOM (default: 16)",
    )

    group.add_argument(
        "--cooperbench-max-tokens-per-turn",
        type=int,
        default=4096,
        help="Max tokens per agent generation turn (default: 4096)",
    )

    group.add_argument(
        "--cooperbench-max-tool-output-chars",
        type=int,
        default=4000,
        help="Max chars in tool output before truncation (default: 4000)",
    )

    group.add_argument(
        "--cooperbench-patch-bonus",
        type=float,
        default=0.1,
        help="Reward bonus for non-empty patch when tests fail (default: 0.1)",
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
    """Main entry point for CooperBench training with Slime."""
    try:
        from slime.train import train
        from slime.utils.arguments import parse_args
    except ImportError as e:
        logger.error(f"slime import failed: {e}", exc_info=True)
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
