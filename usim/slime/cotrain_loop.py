"""Co-training and self-play training loops for USIM.

Composes Slime's building blocks to implement multi-agent training:
  - cotrain(): Mode 2 — both models trained every step
  - selfplay(): Mode 3 — both models trained + opponent checkpoint pool

These functions replace Slime's train_async.py for multi-model setups.
They manage dual placement groups, rollout managers, and training groups.

Requires sglang-config YAML with "actor" and "opponent" named models.
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


def _reload_weights_from_disk(rollout_manager, checkpoint_path: str):
    """Reload SGLang engine weights from disk via HTTP API.

    Uses SGLang's /update_weights_from_disk endpoint (Slime's sglang.patch).
    Called directly via HTTP since Slime's SGLangEngine wrapper doesn't
    expose this method.
    """
    import asyncio

    import aiohttp

    async def _do_reload():
        # Get engine URLs from rollout manager
        engine_urls = ray.get(rollout_manager.get_engine_urls.remote())
        async with aiohttp.ClientSession() as session:
            for url in engine_urls:
                resp = await session.post(
                    f"{url}/update_weights_from_disk",
                    json={"model_path": checkpoint_path},
                )
                if resp.status != 200:
                    body = await resp.text()
                    logger.error(
                        f"Weight reload failed for {url}: {resp.status} {body}"
                    )
                    raise RuntimeError(f"Weight reload failed: {resp.status}")
                logger.info(f"Reloaded weights from {checkpoint_path} on {url}")

    asyncio.run(_do_reload())


def cotrain(args):
    """Mode 2: Co-training loop with dual models.

    Both actor and opponent have their own SGLang engines and training
    groups. Each rollout produces dual samples; both models are trained
    every step.

    Requires sglang-config with "actor" and "opponent" models.
    """
    configure_logger()

    # Create a single large placement group, then split for actor/opponent
    actor_num_gpus = args.actor_num_gpus_per_node * args.actor_num_nodes
    opponent_num_gpus = getattr(args, "opponent_num_gpus", actor_num_gpus)
    total_gpus = actor_num_gpus + opponent_num_gpus + args.rollout_num_gpus

    logger.info(
        f"[COTRAIN] Creating placement group: {total_gpus} GPUs "
        f"(actor={actor_num_gpus}, opponent={opponent_num_gpus}, "
        f"rollout={args.rollout_num_gpus})"
    )

    pg, pg_bundle_indices, pg_gpu_ids = _create_placement_group(total_gpus)

    init_tracking(args)

    # Split placement group indices for actor, opponent, and rollout
    actor_offset = 0
    actor_indices = pg_bundle_indices[actor_offset:actor_offset + actor_num_gpus]
    actor_gpu_ids = pg_gpu_ids[actor_offset:actor_offset + actor_num_gpus]

    opponent_offset = actor_num_gpus
    opponent_indices = pg_bundle_indices[opponent_offset:opponent_offset + opponent_num_gpus]
    opponent_gpu_ids = pg_gpu_ids[opponent_offset:opponent_offset + opponent_num_gpus]

    rollout_offset = actor_num_gpus + opponent_num_gpus
    rollout_indices = pg_bundle_indices[rollout_offset:]
    rollout_gpu_ids = pg_gpu_ids[rollout_offset:]

    # Create rollout manager (manages both SGLang engines via sglang-config)
    rollout_pg = (pg, rollout_indices, rollout_gpu_ids)
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, rollout_pg)

    # Create training groups for both models
    actor_pg = (pg, actor_indices, actor_gpu_ids)
    opponent_pg = (pg, opponent_indices, opponent_gpu_ids)

    actor_model = allocate_train_group(
        args=args,
        num_nodes=args.actor_num_nodes,
        num_gpus_per_node=args.actor_num_gpus_per_node,
        pg=actor_pg,
    )
    opponent_model = allocate_train_group(
        args=args,
        num_nodes=getattr(args, "opponent_num_nodes", 1),
        num_gpus_per_node=getattr(args, "opponent_num_gpus_per_node", opponent_num_gpus),
        pg=opponent_pg,
    )

    # Initialize models
    ray.get(actor_model.async_init(args, role="actor", with_ref=False))
    ray.get(opponent_model.async_init(args, role="opponent", with_ref=False))

    # Sync weights from training workers to SGLang engines
    actor_model.update_weights()
    opponent_model.update_weights()

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

        # Train both models concurrently
        actor_train = actor_model.async_train(rollout_id, rollout_data_ref)
        opponent_train = opponent_model.async_train(rollout_id, rollout_data_ref)
        ray.get([actor_train, opponent_train])

        # Save checkpoints
        if should_run_periodic_action(
            rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout
        ):
            actor_model.save_model(
                rollout_id,
                force_sync=rollout_id == args.num_rollout - 1,
            )
            opponent_model.save_model(
                rollout_id,
                force_sync=rollout_id == args.num_rollout - 1,
            )

        # Update SGLang engine weights
        if (rollout_id + 1) % args.update_weights_interval == 0:
            # Sync generation before weight update
            if rollout_data_next is not None:
                rollout_data_ref = ray.get(rollout_data_next)
                rollout_data_next = None
            actor_model.update_weights()
            opponent_model.update_weights()

        # Eval
        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            ray.get(rollout_manager.eval.remote(rollout_id))

    ray.get(rollout_manager.dispose.remote())


def selfplay(args):
    """Mode 3: Self-play with checkpoint pool.

    Same as cotrain() but adds opponent checkpoint pool for rollout diversity.
    Before each rollout, opponent weights are swapped to a pool checkpoint.
    After training, updated opponent checkpoint is saved to the pool.
    """
    configure_logger()

    actor_num_gpus = args.actor_num_gpus_per_node * args.actor_num_nodes
    opponent_num_gpus = getattr(args, "opponent_num_gpus", actor_num_gpus)
    total_gpus = actor_num_gpus + opponent_num_gpus + args.rollout_num_gpus

    logger.info(
        f"[SELFPLAY] Creating placement group: {total_gpus} GPUs "
        f"(actor={actor_num_gpus}, opponent={opponent_num_gpus}, "
        f"rollout={args.rollout_num_gpus})"
    )

    pg, pg_bundle_indices, pg_gpu_ids = _create_placement_group(total_gpus)

    init_tracking(args)

    # Split placement group
    actor_indices = pg_bundle_indices[:actor_num_gpus]
    actor_gpu_ids = pg_gpu_ids[:actor_num_gpus]
    opponent_indices = pg_bundle_indices[actor_num_gpus:actor_num_gpus + opponent_num_gpus]
    opponent_gpu_ids = pg_gpu_ids[actor_num_gpus:actor_num_gpus + opponent_num_gpus]
    rollout_indices = pg_bundle_indices[actor_num_gpus + opponent_num_gpus:]
    rollout_gpu_ids = pg_gpu_ids[actor_num_gpus + opponent_num_gpus:]

    rollout_pg = (pg, rollout_indices, rollout_gpu_ids)
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, rollout_pg)

    actor_pg = (pg, actor_indices, actor_gpu_ids)
    opponent_pg = (pg, opponent_indices, opponent_gpu_ids)

    actor_model = allocate_train_group(
        args=args,
        num_nodes=args.actor_num_nodes,
        num_gpus_per_node=args.actor_num_gpus_per_node,
        pg=actor_pg,
    )
    opponent_model = allocate_train_group(
        args=args,
        num_nodes=getattr(args, "opponent_num_nodes", 1),
        num_gpus_per_node=getattr(args, "opponent_num_gpus_per_node", opponent_num_gpus),
        pg=opponent_pg,
    )

    ray.get(actor_model.async_init(args, role="actor", with_ref=False))
    ray.get(opponent_model.async_init(args, role="opponent", with_ref=False))

    actor_model.update_weights()
    opponent_model.update_weights()

    # Initialize checkpoint pool
    pool_dir = getattr(args, "pool_dir", "/tmp/usim_checkpoint_pool")
    pool_size = getattr(args, "pool_size", 10)
    pool_save_interval = getattr(args, "pool_save_interval", 16)
    pool_selection = getattr(args, "pool_selection", "random")

    pool = CheckpointPool(pool_dir, max_size=pool_size)

    # Seed pool with initial checkpoint
    opponent_hf = getattr(args, "opponent_hf_checkpoint", None) or args.hf_checkpoint
    if len(pool) == 0:
        pool.add(opponent_hf, step=0)
        logger.info(f"[SELFPLAY] Seeded pool with initial checkpoint: {opponent_hf}")

    # Training loop
    rollout_data_next = None
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        # 1. Swap opponent to a pool checkpoint for rollout diversity
        opponent_ckpt = pool.sample(pool_selection)
        logger.info(
            f"[SELFPLAY] Rollout {rollout_id}: opponent using pool checkpoint {opponent_ckpt}"
        )
        _reload_weights_from_disk(rollout_manager, opponent_ckpt)

        # 2. Run rollout with pool-sampled opponent
        if rollout_data_next is not None:
            rollout_data_ref = ray.get(rollout_data_next)
        else:
            rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))

        # 3. Restore opponent to latest weights for training
        latest_ckpt = pool.latest()
        if latest_ckpt != opponent_ckpt:
            _reload_weights_from_disk(rollout_manager, latest_ckpt)

        # 4. Train both models
        actor_train = actor_model.async_train(rollout_id, rollout_data_ref)
        opponent_train = opponent_model.async_train(rollout_id, rollout_data_ref)
        ray.get([actor_train, opponent_train])

        # 5. Update SGLang engine weights from training workers
        if (rollout_id + 1) % args.update_weights_interval == 0:
            actor_model.update_weights()
            opponent_model.update_weights()

        # 6. Save checkpoints and update pool
        if should_run_periodic_action(
            rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout
        ):
            actor_model.save_model(
                rollout_id,
                force_sync=rollout_id == args.num_rollout - 1,
            )
            opponent_model.save_model(
                rollout_id,
                force_sync=rollout_id == args.num_rollout - 1,
            )

        if should_run_periodic_action(
            rollout_id, pool_save_interval, num_rollout_per_epoch, args.num_rollout
        ):
            ckpt_path = opponent_model.save_model(rollout_id, force_sync=True)
            if ckpt_path:
                pool.add(str(ckpt_path), step=rollout_id)
                logger.info(
                    f"[SELFPLAY] Added opponent checkpoint to pool: step={rollout_id}, "
                    f"pool={pool}"
                )

        # 7. Eval
        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            ray.get(rollout_manager.eval.remote(rollout_id))

        # Start next rollout early (after weight updates)
        if rollout_id + 1 < args.num_rollout:
            rollout_data_next = rollout_manager.generate.remote(rollout_id + 1)
        else:
            rollout_data_next = None

    ray.get(rollout_manager.dispose.remote())
