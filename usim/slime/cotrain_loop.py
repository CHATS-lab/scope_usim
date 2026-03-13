"""Co-training and self-play training loops for USIM.

Composes Slime's building blocks to implement multi-agent training:
  - cotrain(): Mode 2 — single training group, dual-trajectory rollout
  - selfplay(): Mode 3 — single training group + opponent checkpoint pool

Architecture:
  - ONE training group trains a single model using samples from BOTH sides
    of the conversation (agent + opponent trajectories)
  - Actor SGLang engine: update_weights=true (fast NCCL sync via Slime)
  - Opponent SGLang engine: update_weights=false (periodic disk reload)
  - GRPO clipping handles off-policy gap when opponent is stale

This design works within Slime's limitation of one updatable model per
rollout manager. For truly independent dual-model training (different
architectures), Slime would need per-model weight update support.
"""

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


logger = logging.getLogger(__name__)


def _reload_opponent_weights(args, checkpoint_path: str):
    """Reload opponent SGLang engine weights from disk via HTTP API.

    Uses the model router URL for the "opponent" model from sglang-config,
    then calls /update_weights_from_disk on the router which forwards to
    all backend engines for that model.
    """
    import asyncio

    import aiohttp

    from slime.rollout.sglang_rollout import get_model_url

    async def _do_reload():
        url = get_model_url(args, "opponent", "/update_weights_from_disk")
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
    rollout_data_next = None
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
        if rollout_data_next is not None:
            rollout_data_ref = ray.get(rollout_data_next)
        else:
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

        # Start next rollout early (after weight updates)
        if rollout_id + 1 < args.num_rollout:
            rollout_data_next = rollout_manager.generate.remote(rollout_id + 1)
        else:
            rollout_data_next = None

    ray.get(rollout_manager.dispose.remote())
