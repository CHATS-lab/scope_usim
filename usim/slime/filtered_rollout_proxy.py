"""Filtered rollout proxy for dual-model NCCL/IPC weight updates.

Wraps a RolloutManager and intercepts get_updatable_engines_and_lock()
to return only a specific named server's engines with remapped GPU offsets.

This enables two training groups to each sync weights to ONLY their own
SGLang engines via Slime's fast NCCL/IPC path, without modifying Slime's
training actor code.

GPU offset remapping:
  The opponent model's engines have global GPU offsets [4,5,6,7], but the
  opponent training group's Megatron ranks are [0,1,2,3]. Slime's
  connect_rollout_engines() uses gpu_offsets to create IPC Gloo groups —
  if offsets don't match training ranks, engines are misclassified as
  distributed instead of colocated. The proxy subtracts gpu_offset_base
  to align offsets with training ranks.

Requires Slime patch: patches/slime_per_server_engines.patch
  (adds get_server_engines_and_lock() to RolloutManager)
"""

import ray


@ray.remote(num_cpus=0)
class FilteredRolloutProxy:
    """Ray actor that filters RolloutManager engines for one model.

    Each training group gets its own FilteredRolloutProxy configured with
    the server name and GPU offset base for its model. The proxy intercepts
    weight-update-related calls and routes them to the correct server.

    Args:
        rollout_manager: The real RolloutManager Ray actor handle.
        server_name: Name of the model server in sglang-config
            (e.g., "actor" or "opponent").
        gpu_offset_base: Starting GPU index for this model's engines.
            Subtracted from global gpu_offsets so IPC groups align with
            training ranks (e.g., 4 for opponent on GPUs 4-7).
    """

    def __init__(self, rollout_manager, server_name: str, gpu_offset_base: int = 0):
        self._rm = rollout_manager
        self._server_name = server_name
        self._gpu_offset_base = gpu_offset_base

    def get_updatable_engines_and_lock(self):
        """Return this model's engines with remapped GPU offsets.

        Called by MegatronTrainRayActor.update_weights() via
        self.rollout_manager.get_updatable_engines_and_lock.remote().
        """
        engines, lock, num_new, gpu_counts, gpu_offsets = ray.get(
            self._rm.get_server_engines_and_lock.remote(self._server_name)
        )
        remapped_offsets = [off - self._gpu_offset_base for off in gpu_offsets]
        return engines, lock, num_new, gpu_counts, remapped_offsets

    def clear_updatable_num_new_engines(self):
        """Clear num_new_engines for this model's server.

        Called by MegatronTrainRayActor.update_weights() after the first
        connect_rollout_engines() call.
        """
        return ray.get(
            self._rm.clear_server_num_new_engines.remote(self._server_name)
        )

    def recover_updatable_engines(self):
        """Recover dead engines for this model's server.

        Called when fault tolerance is enabled.
        """
        # Delegate to the standard method — fault recovery recreates engines
        # and updates num_new_engines on the server.
        engines, lock, num_new, gpu_counts, gpu_offsets = ray.get(
            self._rm.get_server_engines_and_lock.remote(self._server_name)
        )
        remapped_offsets = [off - self._gpu_offset_base for off in gpu_offsets]
        return engines, lock, num_new, gpu_counts, remapped_offsets

    def set_train_parallel_config(self, config):
        """Forward to real RolloutManager, forcing dp_size=1.

        In dual-model mode, generate() must return one combined chunk
        (dp_size=1) so the driver can split agent/opponent samples via
        DualRolloutSplitter.
        """
        config = dict(config)
        config["dp_size"] = 1
        return ray.get(self._rm.set_train_parallel_config.remote(config))
