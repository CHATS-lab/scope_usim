"""Training loop with two SGLang servers: agent (trainable) + opponent (inference-only).

The agent trains with GRPO while the opponent serves a fixed model locally via SGLang.
This replaces the external API (OpenAI/OpenRouter) with a local SGLang server, enabling:
- Faster rollouts (no API latency)
- Cost-free opponent inference
- Foundation for co-training (Branch 2) and population-based training (Branch 3)

Architecture:
    1. Create separate placement groups for agent and opponent GPUs
    2. Start opponent SGLang server (inference-only, no training)
    3. Start agent SGLang server + training actors (standard Slime flow)
    4. Wire opponent's router URL into args as usim_fixed_opponent_base_url
    5. Run standard Slime training loop (only agent trains)
"""

import logging
from argparse import Namespace
from copy import deepcopy

import ray

from slime.ray.placement_group import (
    _create_placement_group,
    create_rollout_manager,
    create_training_models,
)
from slime.ray.rollout import RolloutServer, start_rollout_servers
from slime.utils.http_utils import init_http_client
from slime.utils.logging_utils import configure_logger, init_tracking
from slime.utils.misc import should_run_periodic_action

logger = logging.getLogger(__name__)


@ray.remote(num_cpus=1, num_gpus=0)
class OpponentServer:
    """Lightweight Ray actor that runs an SGLang server for the opponent model.

    Unlike RolloutManager, this actor only serves inference — no training,
    no data source, no rollout functions. It exposes the router endpoint
    so the agent's environments can call the opponent via litellm.
    """

    def __init__(self, args: Namespace, pg: tuple):
        configure_logger()
        init_http_client(args)
        self.servers: dict[str, RolloutServer] = start_rollout_servers(args, pg)
        srv = next(iter(self.servers.values()))
        self._router_ip = srv.router_ip
        self._router_port = srv.router_port
        logger.info(
            f"Opponent SGLang server started at {self._router_ip}:{self._router_port}"
        )

    def get_router_info(self) -> tuple:
        """Return (router_ip, router_port) for the opponent's SGLang server."""
        return self._router_ip, self._router_port

    def dispose(self) -> None:
        """Clean up resources."""
        pass


def _make_opponent_args(args: Namespace) -> Namespace:
    """Build args namespace for the opponent's SGLang server.

    Copies relevant SGLang settings from agent args and overrides
    model-specific fields for the opponent.
    """
    opp = deepcopy(args)

    # Override model path
    opp.hf_checkpoint = args.opponent_model_path

    # Override GPU settings for the opponent
    opp.rollout_num_gpus = args.opponent_num_gpus
    opp.rollout_num_gpus_per_engine = args.opponent_sglang_tp_size

    # SGLang settings (inherit from agent unless overridden)
    opp.sglang_mem_fraction_static = getattr(
        args, "opponent_sglang_mem_fraction", args.sglang_mem_fraction_static
    )

    # Force fresh router (don't reuse agent's router)
    opp.sglang_router_ip = None
    opp.sglang_router_port = None
    opp.sglang_config = None

    # No PD disaggregation for opponent
    opp.prefill_num_servers = None

    # Opponent doesn't need training-specific settings
    opp.debug_train_only = False
    opp.debug_rollout_only = False

    # Disable fault tolerance for opponent (simpler lifecycle)
    opp.use_fault_tolerance = False

    return opp


def _build_agent_pgs(args: Namespace, opponent_num_gpus: int) -> dict:
    """Build placement groups for the agent (actor + rollout).

    Creates a single PG for the agent side. The opponent gets its own PG
    created separately, so the total cluster must have
    agent_gpus + opponent_gpus available.
    """
    # Standard agent PG: actor GPUs + rollout GPUs
    num_agent_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node
    if not args.colocate:
        num_agent_gpus += args.rollout_num_gpus
    if getattr(args, "use_critic", False):
        num_agent_gpus += args.critic_num_nodes * args.critic_num_gpus_per_node

    logger.info(
        f"GPU allocation: agent={num_agent_gpus}, opponent={opponent_num_gpus}, "
        f"total={num_agent_gpus + opponent_num_gpus}"
    )

    # Create agent PG using standard Slime logic
    # We import create_placement_groups and call it — it reads actor/rollout/critic
    # GPU counts from args and creates one PG with the right layout.
    from slime.ray.placement_group import create_placement_groups

    return create_placement_groups(args)


def train_with_local_opponent(args: Namespace) -> None:
    """Run training with a locally-served opponent model.

    Flow:
        1. Start opponent SGLang server on its own GPU group
        2. Wire opponent endpoint into args
        3. Start agent training with standard Slime loop
    """
    configure_logger()
    init_tracking(args)

    opponent_num_gpus = args.opponent_num_gpus

    # --- Step 1: Start opponent SGLang server ---
    logger.info("=" * 60)
    logger.info("Starting opponent SGLang server")
    logger.info(f"  Model: {args.opponent_model_path}")
    logger.info(f"  GPUs: {opponent_num_gpus}")
    logger.info(f"  TP size: {args.opponent_sglang_tp_size}")
    logger.info("=" * 60)

    # Create opponent PG (separate from agent PG)
    opponent_pg_tuple = _create_placement_group(opponent_num_gpus)
    opponent_args = _make_opponent_args(args)
    opponent_server = OpponentServer.remote(opponent_args, opponent_pg_tuple)

    # Get opponent endpoint
    router_ip, router_port = ray.get(opponent_server.get_router_info.remote())
    opponent_url = f"http://{router_ip}:{router_port}/v1"
    logger.info(f"Opponent serving at: {opponent_url}")

    # --- Step 2: Wire opponent endpoint into agent args ---
    # litellm needs "openai/" prefix for custom base_url with OpenAI-compatible API
    args.usim_fixed_opponent_base_url = opponent_url
    if not args.usim_fixed_opponent_model.startswith("openai/"):
        args.usim_fixed_opponent_model = f"openai/{args.usim_fixed_opponent_model}"

    # No API key needed for local server
    if not getattr(args, "usim_fixed_opponent_api_key_var", None):
        args.usim_fixed_opponent_api_key_var = "USIM_LOCAL_DUMMY_KEY"

    # --- Step 3: Standard Slime training loop for agent ---
    logger.info("=" * 60)
    logger.info("Starting agent training")
    logger.info(f"  Opponent URL: {opponent_url}")
    logger.info(f"  Opponent model: {args.usim_fixed_opponent_model}")
    logger.info("=" * 60)

    # Create agent placement groups
    pgs = _build_agent_pgs(args, opponent_num_gpus)

    # Create agent rollout manager (starts agent's SGLang server)
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    # Create agent training actors
    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)

    if getattr(args, "offload_rollout", False):
        ray.get(rollout_manager.onload_weights.remote())

    # Initial weight sync: agent training → agent rollout engines
    actor_model.update_weights()

    if getattr(args, "offload_rollout", False):
        ray.get(rollout_manager.onload_kv.remote())

    # Eval-only mode
    if args.num_rollout == 0 and args.eval_interval is not None:
        ray.get(rollout_manager.eval.remote(rollout_id=0))

    # --- Training loop (standard Slime pattern, agent only) ---
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        if (
            args.eval_interval is not None
            and rollout_id == 0
            and not args.skip_eval_before_train
        ):
            ray.get(rollout_manager.eval.remote(rollout_id))

        # Generate rollouts (agent interacts with environments that call opponent)
        rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))

        if getattr(args, "offload_rollout", False):
            ray.get(rollout_manager.offload.remote())

        # Train agent (opponent is inference-only, no training)
        if getattr(args, "use_critic", False):
            critic_train_handle = critic_model.async_train(rollout_id, rollout_data_ref)
            if rollout_id >= getattr(args, "num_critic_only_steps", 0):
                ray.get(actor_model.async_train(rollout_id, rollout_data_ref))
            ray.get(critic_train_handle)
        else:
            ray.get(actor_model.async_train(rollout_id, rollout_data_ref))

        # Save model
        if should_run_periodic_action(
            rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout
        ):
            actor_model.save_model(
                rollout_id,
                force_sync=rollout_id == args.num_rollout - 1,
            )
            if getattr(args, "rollout_global_dataset", False):
                ray.get(rollout_manager.save.remote(rollout_id))

        # Offload train / clear memory
        if getattr(args, "offload_train", False):
            actor_model.offload()
        else:
            actor_model.clear_memory()

        if getattr(args, "offload_rollout", False):
            ray.get(rollout_manager.onload_weights.remote())

        # Sync agent weights to agent rollout engines
        actor_model.update_weights()

        if getattr(args, "offload_rollout", False):
            ray.get(rollout_manager.onload_kv.remote())

        # Eval
        if should_run_periodic_action(
            rollout_id, args.eval_interval, num_rollout_per_epoch
        ):
            ray.get(rollout_manager.eval.remote(rollout_id))

    # --- Cleanup ---
    ray.get(rollout_manager.dispose.remote())
    ray.get(opponent_server.dispose.remote())
    logger.info("Training complete.")
