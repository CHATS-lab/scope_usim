"""Reward computation for CooperBench across all three settings.

Uses CooperBench's eval functions (run_patch_test, test_solo, test_merged)
which spin up short-lived eval sandboxes (separate from agent sandboxes).

All eval functions are synchronous and use os.chdir to the dataset directory
(CooperBench hardcodes Path("dataset") / ... relative paths). A module-level
lock serializes chdir + eval to avoid race conditions when called from threads
via asyncio.to_thread().
"""

import asyncio
import logging
import os
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# Lock serializes os.chdir + eval call — needed because os.chdir is process-wide
# and CooperBench eval functions use relative paths from CWD.
_eval_lock = threading.Lock()


def compute_baseline_reward(
    repo_name: str,
    task_id: int,
    feature_id: int,
    patch: str,
    dataset_dir: Optional[str] = None,
    backend: str = "modal",
) -> float:
    """Test single feature patch. Returns 1.0 if tests pass, 0.0 otherwise."""
    try:
        from cooperbench.eval.sandbox import run_patch_test

        with _eval_lock:
            original_cwd = os.getcwd()
            if dataset_dir:
                os.chdir(dataset_dir)
            try:
                result = run_patch_test(
                    repo_name=repo_name,
                    task_id=task_id,
                    feature_id=feature_id,
                    agent_patch=patch,
                    backend=backend,
                )
            finally:
                if dataset_dir:
                    os.chdir(original_cwd)

        if result.get("error"):
            logger.warning(f"Baseline eval error: {result['error']}")
            return 0.0

        if result.get("passed"):
            logger.info(f"Baseline: feature {feature_id} passed — reward=1.0")
            return 1.0

        logger.info(f"Baseline: feature {feature_id} failed — reward=0.0")
        return 0.0

    except Exception as e:
        logger.error(f"Baseline reward computation failed: {e}")
        return 0.0


def compute_solo_reward(
    repo_name: str,
    task_id: int,
    f1_id: int,
    f2_id: int,
    patch: str,
    dataset_dir: Optional[str] = None,
    backend: str = "modal",
) -> float:
    """Test solo patch against both features. Returns 1.0 if both pass."""
    try:
        from cooperbench.eval.sandbox import test_solo

        with _eval_lock:
            original_cwd = os.getcwd()
            if dataset_dir:
                os.chdir(dataset_dir)
            try:
                result = test_solo(
                    repo_name=repo_name,
                    task_id=task_id,
                    feature1_id=f1_id,
                    feature2_id=f2_id,
                    patch=patch,
                    backend=backend,
                )
            finally:
                if dataset_dir:
                    os.chdir(original_cwd)

        if result.get("error"):
            logger.warning(f"Solo eval error: {result['error']}")
            return 0.0

        if result.get("both_passed"):
            logger.info("Solo: both features passed — reward=1.0")
            return 1.0

        f1 = result.get("feature1", {}).get("passed", False)
        f2 = result.get("feature2", {}).get("passed", False)
        logger.info(f"Solo: f1={'pass' if f1 else 'fail'}, f2={'pass' if f2 else 'fail'} — reward=0.0")
        return 0.0

    except Exception as e:
        logger.error(f"Solo reward computation failed: {e}")
        return 0.0


def compute_coop_reward(
    repo_name: str,
    task_id: int,
    f1_id: int,
    f2_id: int,
    agent_patch: str,
    partner_patch: str,
    dataset_dir: Optional[str] = None,
    backend: str = "modal",
    partial_reward: bool = False,
) -> float:
    """Test merged patches. Returns 1.0 if both pass, 0.5 partial."""
    try:
        from cooperbench.eval.sandbox import test_merged

        with _eval_lock:
            original_cwd = os.getcwd()
            if dataset_dir:
                os.chdir(dataset_dir)
            try:
                result = test_merged(
                    repo_name=repo_name,
                    task_id=task_id,
                    feature1_id=f1_id,
                    feature2_id=f2_id,
                    patch1=agent_patch,
                    patch2=partner_patch,
                    backend=backend,
                )
            finally:
                if dataset_dir:
                    os.chdir(original_cwd)

        if result.get("error"):
            logger.warning(f"Coop eval error: {result['error']}")
            return 0.0

        if result.get("both_passed"):
            logger.info("Coop: both features passed — reward=1.0")
            return 1.0

        f1_passed = result.get("feature1", {}).get("passed", False)
        if partial_reward and f1_passed:
            logger.info("Coop: agent's feature passed but merge issue — reward=0.5")
            return 0.5

        f2_passed = result.get("feature2", {}).get("passed", False)
        logger.info(
            f"Coop: f1={'pass' if f1_passed else 'fail'}, "
            f"f2={'pass' if f2_passed else 'fail'} — reward=0.0"
        )
        return 0.0

    except Exception as e:
        logger.error(f"Coop reward computation failed: {e}")
        return 0.0


# ---------------------------------------------------------------------------
# Async wrappers — run reward in thread pool so they don't block the event loop
# ---------------------------------------------------------------------------


async def compute_baseline_reward_async(
    repo_name: str,
    task_id: int,
    feature_id: int,
    patch: str,
    dataset_dir: Optional[str] = None,
    backend: str = "modal",
) -> float:
    """Async wrapper — runs compute_baseline_reward in a thread."""
    return await asyncio.to_thread(
        compute_baseline_reward,
        repo_name, task_id, feature_id, patch, dataset_dir, backend,
    )


async def compute_solo_reward_async(
    repo_name: str,
    task_id: int,
    f1_id: int,
    f2_id: int,
    patch: str,
    dataset_dir: Optional[str] = None,
    backend: str = "modal",
) -> float:
    """Async wrapper — runs compute_solo_reward in a thread."""
    return await asyncio.to_thread(
        compute_solo_reward,
        repo_name, task_id, f1_id, f2_id, patch, dataset_dir, backend,
    )


async def compute_coop_reward_async(
    repo_name: str,
    task_id: int,
    f1_id: int,
    f2_id: int,
    agent_patch: str,
    partner_patch: str,
    dataset_dir: Optional[str] = None,
    backend: str = "modal",
    partial_reward: bool = False,
) -> float:
    """Async wrapper — runs compute_coop_reward in a thread."""
    return await asyncio.to_thread(
        compute_coop_reward,
        repo_name, task_id, f1_id, f2_id,
        agent_patch, partner_patch, dataset_dir, backend, partial_reward,
    )
