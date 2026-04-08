"""Co-training and self-play training loops for USIM.

Composes Slime's building blocks to implement multi-agent training:
  - cotrain(): Single training group, dual-trajectory rollout
  - selfplay(): Single training group + opponent checkpoint pool
  - dual_cotrain(): Two training groups, colocated NCCL/IPC weight sync
  - dual_selfplay(): Two training groups + opponent checkpoint pool

Single-model modes (cotrain/selfplay):
  ONE training group trains a single model using samples from BOTH sides.
  Actor engine: update_weights=true (NCCL). Opponent: update_weights=false (HTTP).

Dual-model modes (dual_cotrain/dual_selfplay):
  TWO training groups, each with their own SGLang engines on shared GPUs
  (colocated time-multiplexing). FilteredRolloutProxy enables per-model
  NCCL/IPC weight sync. DualRolloutSplitter separates combined rollout
  output into per-model training data.
  Requires: Slime patch patches/slime_per_server_engines.patch
"""

import copy
import logging

import ray

from slime.ray.placement_group import (
    _create_placement_group,
    allocate_train_group,
    create_rollout_manager,
)
from slime.utils.logging_utils import configure_logger, init_tracking
from slime.utils.misc import should_run_periodic_action

from usim.core.checkpoint_pool import CheckpointPool


def _get_model_url(args, model_name: str, endpoint: str) -> str:
    sglang_config = getattr(args, 'sglang_config_data', None)
    if sglang_config and 'models' in sglang_config:
        for model in sglang_config['models']:
            if model.get('name') == model_name:
                host = getattr(args, 'sglang_router_ip', '127.0.0.1')
                port = model.get('router_port', getattr(args, 'sglang_router_port', 4702))
                return f'http://{host}:{port}{endpoint}'
    host = getattr(args, 'sglang_router_ip', '127.0.0.1')
    port = getattr(args, 'sglang_router_port', 4702)
    return f'http://{host}:{port}{endpoint}' 


logger = logging.getLogger(__name__)


def _reload_opponent_weights(args, checkpoint_path: str):
    """Reload opponent SGLang engine weights from disk via HTTP API.

    Uses the model router URL for the "opponent" model from sglang-config,
    then calls /update_weights_from_disk on the router which forwards to
    all backend engines for that model.
    """
    import asyncio

    import aiohttp


    async def _do_reload():
        url = _get_model_url(args, "opponent", "/update_weights_from_disk")
        logger.info(f"Reloading opponent weights from {checkpoint_path} via {url}")
        async with aiohttp.ClientSession() as session:
            resp = await session.post(
                url,
                json={"model_path": checkpoint_path},
                timeout=aiohttp.ClientTimeout(total=300),
            )
            if resp.status != 200:
                body = await resp.text()
                logger.error(f"Opponent weight reload failed: {resp.status} {body}")
                raise RuntimeError(f"Opponent weight reload failed: {resp.status}")
            logger.info(f"Opponent weights reloaded from {checkpoint_path}")

    asyncio.run(_do_reload())


def cotrain(args):
    """Mode 2: Co-training loop with dual-trajectory rollout.

    Uses a single training group that trains on samples from both sides of
    the conversation. The rollout function produces agent and opponent
    trajectory samples via CoTrainingOrchestrator; all samples go into one
    training batch.

    Actor SGLang engine weights are synced via Slime's NCCL mechanism.
    Opponent SGLang engine weights are updated periodically from the actor's
    saved HF checkpoint (since both start from the same model).

    Requires sglang-config with "actor" (update_weights: true) and
    "opponent" (update_weights: false) models.
    """
    configure_logger()

    actor_num_gpus = args.actor_num_gpus_per_node * args.actor_num_nodes

    # GPU layout: [training] [rollout]
    # Training GPUs are separate from rollout GPUs (non-colocated, like train_async.py)
    total_gpus = actor_num_gpus + args.rollout_num_gpus

    logger.info(
        f"[COTRAIN] Creating placement group: {total_gpus} GPUs "
        f"(training={actor_num_gpus}, rollout={args.rollout_num_gpus})"
    )

    pg, pg_bundle_indices, pg_gpu_ids = _create_placement_group(total_gpus)

    init_tracking(args)

    # Split: training GPUs first, then rollout GPUs
    train_indices = pg_bundle_indices[:actor_num_gpus]
    train_gpu_ids = pg_gpu_ids[:actor_num_gpus]
    rollout_indices = pg_bundle_indices[actor_num_gpus:]
    rollout_gpu_ids = pg_gpu_ids[actor_num_gpus:]

    # Create rollout manager (manages both actor and opponent SGLang engines)
    rollout_pg = (pg, rollout_indices, rollout_gpu_ids)
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, rollout_pg)

    # Single training group — trains on dual-trajectory samples
    train_pg = (pg, train_indices, train_gpu_ids)
    actor_model = allocate_train_group(
        args=args,
        num_nodes=args.actor_num_nodes,
        num_gpus_per_node=args.actor_num_gpus_per_node,
        pg=train_pg,
    )

    # Initialize model
    with_ref = args.kl_coef != 0 or getattr(args, "use_kl_loss", False)
    ray.get(actor_model.async_init(args, role="actor", with_ref=with_ref))

    # Connect training group to rollout manager (required for NCCL weight sync)
    actor_model.set_rollout_manager(rollout_manager)

    # Sync weights to actor SGLang engines via NCCL
    actor_model.update_weights()
    # Opponent engines start with the HF checkpoint weights (loaded at engine start)

    # Training loop
    rollout_data_next = rollout_manager.generate.remote(args.start_rollout_id)
    rollout_data_ref = None
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        # Get current rollout data
        if rollout_data_next is not None:
            rollout_data_ref = ray.get(rollout_data_next)

        # Start next rollout early (async pipelining)
        if rollout_id + 1 < args.num_rollout:
            rollout_data_next = rollout_manager.generate.remote(rollout_id + 1)

        # Train on all samples (both agent and opponent trajectories)
        ray.get(actor_model.async_train(rollout_id, rollout_data_ref))

        # Save checkpoints
        if should_run_periodic_action(
            rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout
        ):
            actor_model.save_model(
                rollout_id,
                force_sync=rollout_id == args.num_rollout - 1,
            )

        # Update SGLang engine weights
        if (rollout_id + 1) % args.update_weights_interval == 0:
            # Sync generation before weight update
            if rollout_data_next is not None:
                rollout_data_ref = ray.get(rollout_data_next)
                rollout_data_next = None
            # Actor engines: fast NCCL sync
            actor_model.update_weights()
            # Opponent engines: reload from HF checkpoint on disk
            # (uses the original HF checkpoint path — same model as actor)
            try:
                _reload_opponent_weights(args, args.hf_checkpoint)
            except Exception as e:
                logger.warning(f"Opponent weight reload failed (non-fatal): {e}")

        # Eval
        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            ray.get(rollout_manager.eval.remote(rollout_id))

    ray.get(rollout_manager.dispose.remote())


def selfplay(args):
    """Mode 3: Self-play with checkpoint pool.

    Same architecture as cotrain() (single training group, dual trajectories)
    but adds a checkpoint pool for opponent diversity. Before each rollout,
    opponent engine weights are swapped to a historical checkpoint from the
    pool. GRPO clipping handles the off-policy correction.

    Pool flow:
      1. Sample opponent checkpoint from pool
      2. Reload opponent engine weights from sampled checkpoint
      3. Run rollout → dual trajectories
      4. Train model on all samples
      5. Periodically save actor checkpoint to pool
    """
    configure_logger()

    actor_num_gpus = args.actor_num_gpus_per_node * args.actor_num_nodes
    total_gpus = actor_num_gpus + args.rollout_num_gpus

    logger.info(
        f"[SELFPLAY] Creating placement group: {total_gpus} GPUs "
        f"(training={actor_num_gpus}, rollout={args.rollout_num_gpus})"
    )

    pg, pg_bundle_indices, pg_gpu_ids = _create_placement_group(total_gpus)

    init_tracking(args)

    train_indices = pg_bundle_indices[:actor_num_gpus]
    train_gpu_ids = pg_gpu_ids[:actor_num_gpus]
    rollout_indices = pg_bundle_indices[actor_num_gpus:]
    rollout_gpu_ids = pg_gpu_ids[actor_num_gpus:]

    rollout_pg = (pg, rollout_indices, rollout_gpu_ids)
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, rollout_pg)

    train_pg = (pg, train_indices, train_gpu_ids)
    actor_model = allocate_train_group(
        args=args,
        num_nodes=args.actor_num_nodes,
        num_gpus_per_node=args.actor_num_gpus_per_node,
        pg=train_pg,
    )

    with_ref = args.kl_coef != 0 or getattr(args, "use_kl_loss", False)
    ray.get(actor_model.async_init(args, role="actor", with_ref=with_ref))
    actor_model.set_rollout_manager(rollout_manager)
    actor_model.update_weights()

    # Initialize checkpoint pool
    pool_dir = getattr(args, "pool_dir", "/tmp/usim_checkpoint_pool")
    pool_size = getattr(args, "pool_size", 10)
    pool_save_interval = getattr(args, "pool_save_interval", 16)
    pool_selection = getattr(args, "pool_selection", "random")

    pool = CheckpointPool(pool_dir, max_size=pool_size)

    # Seed pool with initial checkpoint (HF format — compatible with SGLang)
    initial_ckpt = getattr(args, "opponent_hf_checkpoint", None) or args.hf_checkpoint
    if len(pool) == 0:
        pool.add(initial_ckpt, step=0)
        logger.info(f"[SELFPLAY] Seeded pool with: {initial_ckpt}")

    # Training loop
    # NOTE: No async pre-fetching in selfplay — each rollout must run AFTER
    # opponent weights are reloaded from the pool checkpoint.
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        # 1. Swap opponent to a pool checkpoint for rollout diversity
        opponent_ckpt = pool.sample(pool_selection)
        logger.info(
            f"[SELFPLAY] Rollout {rollout_id}: opponent checkpoint={opponent_ckpt}"
        )
        try:
            _reload_opponent_weights(args, opponent_ckpt)
        except Exception as e:
            logger.error(f"Failed to reload opponent weights: {e}")
            # Fall through — opponent keeps whatever weights it had

        # 2. Run rollout with pool-sampled opponent
        rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))

        # 3. Train on all samples (both agent and opponent trajectories)
        ray.get(actor_model.async_train(rollout_id, rollout_data_ref))

        # 4. Update actor engine weights via NCCL
        if (rollout_id + 1) % args.update_weights_interval == 0:
            actor_model.update_weights()

        # 5. Save checkpoints and update pool
        if should_run_periodic_action(
            rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout
        ):
            actor_model.save_model(
                rollout_id,
                force_sync=rollout_id == args.num_rollout - 1,
            )

        if should_run_periodic_action(
            rollout_id, pool_save_interval, num_rollout_per_epoch, args.num_rollout
        ):
            # Save HF checkpoint for the pool (SGLang needs HF format for disk reload)
            if getattr(args, "save_hf", None):
                pool.add(args.save_hf, step=rollout_id)
                logger.info(
                    f"[SELFPLAY] Added to pool: step={rollout_id}, path={args.save_hf}"
                )

        # 6. Eval
        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            ray.get(rollout_manager.eval.remote(rollout_id))

    ray.get(rollout_manager.dispose.remote())


def _make_opponent_args(args):
    """Create args namespace for opponent training group.

    Copies base args and overrides checkpoint paths and GPU config
    for the opponent model. The opponent training group needs separate
    save directories and potentially a different HF checkpoint.
    """
    opp = copy.deepcopy(args)
    if getattr(args, "opponent_hf_checkpoint", None):
        opp.hf_checkpoint = args.opponent_hf_checkpoint
    if getattr(args, "save_dir", None):
        opp.save_dir = args.save_dir + "_opponent"
    opp_gpus = getattr(args, "opponent_num_gpus_per_node", None)
    if opp_gpus:
        opp.actor_num_gpus_per_node = opp_gpus
    opp_nodes = getattr(args, "opponent_num_nodes", None)
    if opp_nodes:
        opp.actor_num_nodes = opp_nodes
    return opp


def _get_dp_size(model):
    """Get DP size from a training group's rank 0 parallel config."""
    config = ray.get(model._actor_handlers[0].get_train_parallel_config.remote())
    return config.get("dp_size", 1)


def _reload_model_weights(args, model_name: str, checkpoint_path: str):
    """Reload a named model's SGLang engine weights from disk via HTTP.

    Uses the model router URL from sglang-config, calling
    /update_weights_from_disk on the router which forwards to all
    backend engines for that model.
    """
    import asyncio

    import aiohttp

    async def _do_reload():
        url = _get_model_url(args, model_name, "/update_weights_from_disk")
        logger.info(f"Reloading {model_name} weights from {checkpoint_path} via {url}")
        async with aiohttp.ClientSession() as session:
            resp = await session.post(
                url,
                json={"model_path": checkpoint_path},
                timeout=aiohttp.ClientTimeout(total=300),
            )
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(
                    f"{model_name} weight reload failed: {resp.status} {body}"
                )
            logger.info(f"{model_name} weights reloaded from {checkpoint_path}")

    asyncio.run(_do_reload())


def dual_cotrain(args):
    """True dual-model colocated co-training.

    Two separate training groups (actor + opponent), each with their own
    SGLang engines, sharing GPUs via colocated time-multiplexing.

    GPU layout (8 GPUs):
      GPUs 0-3: actor training workers + actor SGLang engines
      GPUs 4-7: opponent training workers + opponent SGLang engines

    Each training group syncs weights to its own engines via NCCL/IPC
    through FilteredRolloutProxy. DualRolloutSplitter separates the
    combined rollout output into per-model training data.

    Requires:
      - sglang-config with "actor" and "opponent" models (both update_weights: true)
      - Slime patch: patches/slime_per_server_engines.patch
    """
    configure_logger()

    actor_gpus = args.actor_num_gpus_per_node * args.actor_num_nodes
    opponent_args = _make_opponent_args(args)
    opponent_gpus = opponent_args.actor_num_gpus_per_node * opponent_args.actor_num_nodes
    total_gpus = actor_gpus + opponent_gpus

    # Colocated mode is required for dual-model
    for flag in ("colocate", "offload_rollout", "offload_train"):
        if not getattr(args, flag, False):
            logger.warning(
                f"[DUAL_COTRAIN] Setting --{flag.replace('_', '-')} (required)"
            )
            setattr(args, flag, True)
            setattr(opponent_args, flag, True)

    logger.info(
        f"[DUAL_COTRAIN] GPU layout: {total_gpus} total "
        f"(actor={actor_gpus}, opponent={opponent_gpus})"
    )

    # ONE placement group for all GPUs
    pg, pg_bundle_indices, pg_gpu_ids = _create_placement_group(total_gpus)
    init_tracking(args)

    actor_indices = pg_bundle_indices[:actor_gpus]
    actor_gpu_ids = pg_gpu_ids[:actor_gpus]
    opponent_indices = pg_bundle_indices[actor_gpus:]
    opponent_gpu_ids = pg_gpu_ids[actor_gpus:]

    # ONE rollout manager (both actor + opponent engines, all GPUs)
    args.rollout_num_gpus = total_gpus
    rollout_pg = (pg, pg_bundle_indices, pg_gpu_ids)
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, rollout_pg)

    # TWO training groups on separate GPU subsets
    actor_pg = (pg, actor_indices, actor_gpu_ids)
    actor_model = allocate_train_group(
        args=args,
        num_nodes=args.actor_num_nodes,
        num_gpus_per_node=args.actor_num_gpus_per_node,
        pg=actor_pg,
    )

    opponent_pg = (pg, opponent_indices, opponent_gpu_ids)
    opponent_model = allocate_train_group(
        args=opponent_args,
        num_nodes=opponent_args.actor_num_nodes,
        num_gpus_per_node=opponent_args.actor_num_gpus_per_node,
        pg=opponent_pg,
    )

    # Initialize both training groups
    # Per-role KL: if --no-agent-kl, agent trains without KL; opponent keeps KL
    no_agent_kl = getattr(args, "no_agent_kl", False)
    actor_with_ref = (args.kl_coef != 0 or getattr(args, "use_kl_loss", False)) and not no_agent_kl
    opponent_with_ref = args.kl_coef != 0 or getattr(args, "use_kl_loss", False)

    if no_agent_kl:
        # Disable KL on actor args so the loss function doesn't compute it
        args.use_kl_loss = False
        args.kl_coef = 0
        logger.info("[DUAL_COTRAIN] Agent KL disabled (--no-agent-kl)")

    ray.get(actor_model.async_init(args, role="actor", with_ref=actor_with_ref))
    ray.get(opponent_model.async_init(opponent_args, role="actor", with_ref=opponent_with_ref))

    # TWO proxies for per-model NCCL/IPC weight updates
    from usim.slime.filtered_rollout_proxy import FilteredRolloutProxy

    actor_proxy = FilteredRolloutProxy.remote(
        rollout_manager, "actor", gpu_offset_base=0
    )
    opponent_proxy = FilteredRolloutProxy.remote(
        rollout_manager, "opponent", gpu_offset_base=actor_gpus
    )

    # Connect training groups to their proxies
    # (proxy forces dp_size=1 on RM for combined output)
    actor_model.set_rollout_manager(actor_proxy)
    opponent_model.set_rollout_manager(opponent_proxy)

    # Initial weight sync (colocated sequence)
    ray.get(rollout_manager.onload_weights.remote())
    actor_model.update_weights()
    opponent_model.update_weights()
    ray.get(rollout_manager.onload_kv.remote())

    # Data splitting params
    n_agent_samples = args.rollout_batch_size * getattr(args, "n_samples_per_prompt", 1)
    actor_dp = _get_dp_size(actor_model)
    opponent_dp = _get_dp_size(opponent_model)
    balance = getattr(args, "balance_data", False)

    from usim.slime.dual_rollout_splitter import split_combined_rollout

    # Training loop (follows Slime colocated pattern)
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        # 1. Generate (all engines active)
        rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))

        # 2. Split into per-model training data
        actor_data, opponent_data = split_combined_rollout(
            rollout_data_ref, n_agent_samples, actor_dp, opponent_dp, balance
        )

        # 3. Offload rollout engines
        ray.get(rollout_manager.offload.remote())

        # 4. Train both models in parallel (different GPU subsets)
        actor_train_refs = actor_model.async_train(rollout_id, actor_data)
        opponent_train_refs = opponent_model.async_train(rollout_id, opponent_data)
        ray.get(actor_train_refs)
        ray.get(opponent_train_refs)

        # 5. Save checkpoints (while training models are still loaded)
        if should_run_periodic_action(
            rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout
        ):
            actor_model.save_model(
                rollout_id, force_sync=rollout_id == args.num_rollout - 1
            )
            opponent_model.save_model(
                rollout_id, force_sync=rollout_id == args.num_rollout - 1
            )

        # 6. Offload training → onload weights → NCCL sync → onload KV
        actor_model.offload()
        opponent_model.offload()
        ray.get(rollout_manager.onload_weights.remote())
        actor_model.update_weights()
        opponent_model.update_weights()
        ray.get(rollout_manager.onload_kv.remote())

        # 7. Eval
        if should_run_periodic_action(
            rollout_id, args.eval_interval, num_rollout_per_epoch
        ):
            ray.get(rollout_manager.eval.remote(rollout_id))

    ray.get(rollout_manager.dispose.remote())


def dual_selfplay(args):
    """True dual-model self-play with checkpoint pool.

    Same architecture as dual_cotrain() (two training groups, colocated)
    plus a checkpoint pool for opponent diversity. Before each rollout,
    opponent engine weights are swapped to a historical checkpoint from
    the pool via HTTP. Both models train every step via NCCL/IPC.

    GRPO clipping handles the off-policy gap between pool weights and
    latest trained weights.
    """
    configure_logger()

    actor_gpus = args.actor_num_gpus_per_node * args.actor_num_nodes
    opponent_args = _make_opponent_args(args)
    opponent_gpus = opponent_args.actor_num_gpus_per_node * opponent_args.actor_num_nodes
    total_gpus = actor_gpus + opponent_gpus

    for flag in ("colocate", "offload_rollout", "offload_train"):
        if not getattr(args, flag, False):
            logger.warning(
                f"[DUAL_SELFPLAY] Setting --{flag.replace('_', '-')} (required)"
            )
            setattr(args, flag, True)
            setattr(opponent_args, flag, True)

    logger.info(
        f"[DUAL_SELFPLAY] GPU layout: {total_gpus} total "
        f"(actor={actor_gpus}, opponent={opponent_gpus})"
    )

    pg, pg_bundle_indices, pg_gpu_ids = _create_placement_group(total_gpus)
    init_tracking(args)

    actor_indices = pg_bundle_indices[:actor_gpus]
    actor_gpu_ids = pg_gpu_ids[:actor_gpus]
    opponent_indices = pg_bundle_indices[actor_gpus:]
    opponent_gpu_ids = pg_gpu_ids[actor_gpus:]

    args.rollout_num_gpus = total_gpus
    rollout_pg = (pg, pg_bundle_indices, pg_gpu_ids)
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, rollout_pg)

    actor_pg = (pg, actor_indices, actor_gpu_ids)
    actor_model = allocate_train_group(
        args=args,
        num_nodes=args.actor_num_nodes,
        num_gpus_per_node=args.actor_num_gpus_per_node,
        pg=actor_pg,
    )

    opponent_pg = (pg, opponent_indices, opponent_gpu_ids)
    opponent_model = allocate_train_group(
        args=opponent_args,
        num_nodes=opponent_args.actor_num_nodes,
        num_gpus_per_node=opponent_args.actor_num_gpus_per_node,
        pg=opponent_pg,
    )

    # Per-role KL: if --no-agent-kl, agent trains without KL; opponent keeps KL
    no_agent_kl = getattr(args, "no_agent_kl", False)
    actor_with_ref = (args.kl_coef != 0 or getattr(args, "use_kl_loss", False)) and not no_agent_kl
    opponent_with_ref = args.kl_coef != 0 or getattr(args, "use_kl_loss", False)

    if no_agent_kl:
        args.use_kl_loss = False
        args.kl_coef = 0
        logger.info("[DUAL_SELFPLAY] Agent KL disabled (--no-agent-kl)")

    ray.get(actor_model.async_init(args, role="actor", with_ref=actor_with_ref))
    ray.get(opponent_model.async_init(opponent_args, role="actor", with_ref=opponent_with_ref))

    from usim.slime.filtered_rollout_proxy import FilteredRolloutProxy

    actor_proxy = FilteredRolloutProxy.remote(
        rollout_manager, "actor", gpu_offset_base=0
    )
    opponent_proxy = FilteredRolloutProxy.remote(
        rollout_manager, "opponent", gpu_offset_base=actor_gpus
    )

    actor_model.set_rollout_manager(actor_proxy)
    opponent_model.set_rollout_manager(opponent_proxy)

    ray.get(rollout_manager.onload_weights.remote())
    actor_model.update_weights()
    opponent_model.update_weights()
    ray.get(rollout_manager.onload_kv.remote())

    # Initialize checkpoint pool
    pool_dir = getattr(args, "pool_dir", "/tmp/usim_checkpoint_pool")
    pool_size = getattr(args, "pool_size", 10)
    pool_save_interval = getattr(args, "pool_save_interval", 16)
    pool_selection = getattr(args, "pool_selection", "random")

    pool = CheckpointPool(pool_dir, max_size=pool_size)
    initial_ckpt = getattr(args, "opponent_hf_checkpoint", None) or args.hf_checkpoint
    if len(pool) == 0:
        pool.add(initial_ckpt, step=0)
        logger.info(f"[DUAL_SELFPLAY] Seeded pool with: {initial_ckpt}")

    n_agent_samples = args.rollout_batch_size * getattr(args, "n_samples_per_prompt", 1)
    actor_dp = _get_dp_size(actor_model)
    opponent_dp = _get_dp_size(opponent_model)
    balance = getattr(args, "balance_data", False)

    from usim.slime.dual_rollout_splitter import split_combined_rollout

    # Training loop (no async pipelining — opponent weights swap each rollout)
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        # 1. Swap opponent to pool checkpoint for diverse rollout
        opponent_ckpt = pool.sample(pool_selection)
        logger.info(
            f"[DUAL_SELFPLAY] Rollout {rollout_id}: "
            f"opponent checkpoint={opponent_ckpt}"
        )
        try:
            _reload_model_weights(args, "opponent", opponent_ckpt)
        except Exception as e:
            logger.error(f"Failed to reload opponent weights: {e}")

        # 2. Generate (actor latest, opponent pool weights)
        rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))

        # 3. Split
        actor_data, opponent_data = split_combined_rollout(
            rollout_data_ref, n_agent_samples, actor_dp, opponent_dp, balance
        )

        # 4. Offload engines → train both in parallel
        ray.get(rollout_manager.offload.remote())
        actor_train_refs = actor_model.async_train(rollout_id, actor_data)
        opponent_train_refs = opponent_model.async_train(rollout_id, opponent_data)
        ray.get(actor_train_refs)
        ray.get(opponent_train_refs)

        # 5. Save checkpoints
        if should_run_periodic_action(
            rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout
        ):
            actor_model.save_model(
                rollout_id, force_sync=rollout_id == args.num_rollout - 1
            )
            opponent_model.save_model(
                rollout_id, force_sync=rollout_id == args.num_rollout - 1
            )

        # 6. Offload training → onload weights → NCCL sync → onload KV
        actor_model.offload()
        opponent_model.offload()
        ray.get(rollout_manager.onload_weights.remote())
        actor_model.update_weights()
        opponent_model.update_weights()
        ray.get(rollout_manager.onload_kv.remote())

        # 7. Save opponent to pool (periodic)
        if should_run_periodic_action(
            rollout_id, pool_save_interval, num_rollout_per_epoch, args.num_rollout
        ):
            if getattr(opponent_args, "save_hf", None):
                pool.add(opponent_args.save_hf, step=rollout_id)
                logger.info(
                    f"[DUAL_SELFPLAY] Added to pool: step={rollout_id}, "
                    f"path={opponent_args.save_hf}"
                )

        # 8. Eval
        if should_run_periodic_action(
            rollout_id, args.eval_interval, num_rollout_per_epoch
        ):
            ray.get(rollout_manager.eval.remote(rollout_id))

    ray.get(rollout_manager.dispose.remote())
