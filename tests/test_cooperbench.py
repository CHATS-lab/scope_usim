"""Smoke tests for CooperBench integration.

Tests data source loading, agent prompt rendering, and reward function signatures.
No infrastructure (Modal, Redis, Docker) required.

Requires: slime, cooperbench packages installed (run on remote machine).
Run: pytest tests/test_cooperbench.py -v
"""

from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from usim.cooperbench.agent import CooperBenchAgent, STOP_SIGNAL
from usim.cooperbench.data_source import CooperBenchDataSource, get_cooperbench_data_source
from usim.cooperbench.reward import (
    compute_baseline_reward,
    compute_baseline_reward_async,
    compute_coop_reward,
    compute_coop_reward_async,
    compute_solo_reward,
    compute_solo_reward_async,
)
from usim.core.types import Message


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DATA_DIR = str(Path(__file__).parent.parent / "data" / "cooperbench")


def _make_args(setting="solo", n_samples=8):
    return Namespace(
        cooperbench_setting=setting,
        cooperbench_data_dir=DATA_DIR,
        n_samples_per_prompt=n_samples,
    )


# ---------------------------------------------------------------------------
# Data source tests
# ---------------------------------------------------------------------------


class TestCooperBenchDataSource:

    @patch("usim.cooperbench.data_source.get_image_name", return_value="fake:latest")
    def test_baseline_entry_count(self, _):
        ds = CooperBenchDataSource(_make_args(setting="baseline"))
        assert len(ds._load_entries()) == 170

    @patch("usim.cooperbench.data_source.get_image_name", return_value="fake:latest")
    def test_solo_entry_count(self, _):
        ds = CooperBenchDataSource(_make_args(setting="solo"))
        assert len(ds._load_entries()) == 587

    @patch("usim.cooperbench.data_source.get_image_name", return_value="fake:latest")
    def test_coop_entry_count(self, _):
        ds = CooperBenchDataSource(_make_args(setting="coop"))
        # Each pair expanded to 2 directed entries: 587 * 2 = 1174
        assert len(ds._load_entries()) == 1174

    @patch("usim.cooperbench.data_source.get_image_name", return_value="fake:latest")
    def test_baseline_entry_fields(self, _):
        ds = CooperBenchDataSource(_make_args(setting="baseline"))
        entry = ds._load_entries()[0]
        assert "repo" in entry
        assert "task_id" in entry
        assert "feature_ids" in entry
        assert "descriptions" in entry
        assert entry["setting"] == "baseline"
        assert entry["agent_feature_id"] is not None
        assert entry["partner_feature_id"] is None

    @patch("usim.cooperbench.data_source.get_image_name", return_value="fake:latest")
    def test_coop_directed_entries(self, _):
        ds = CooperBenchDataSource(_make_args(setting="coop"))
        entries = ds._load_entries()
        e1, e2 = entries[0], entries[1]
        assert e1["repo"] == e2["repo"]
        assert e1["task_id"] == e2["task_id"]
        assert e1["agent_feature_id"] == e2["partner_feature_id"]
        assert e1["partner_feature_id"] == e2["agent_feature_id"]

    @patch("usim.cooperbench.data_source.get_image_name", return_value="fake:latest")
    def test_get_samples_returns_groups(self, _):
        ds = CooperBenchDataSource(_make_args(setting="baseline", n_samples=4))
        groups = ds.get_samples(num_samples=3)
        assert len(groups) == 3
        for group in groups:
            assert len(group) == 4

    @patch("usim.cooperbench.data_source.get_image_name", return_value="fake:latest")
    def test_sample_metadata_fields(self, _):
        ds = CooperBenchDataSource(_make_args(setting="solo", n_samples=1))
        sample = ds.get_samples(num_samples=1)[0][0]
        meta = sample.metadata
        for key in ["repo", "task_id", "feature_ids", "descriptions", "setting", "image_name"]:
            assert key in meta
        assert meta["setting"] == "solo"
        assert meta["image_name"] == "fake:latest"

    @patch("usim.cooperbench.data_source.get_image_name", return_value="fake:latest")
    def test_sample_prompt_is_nonempty(self, _):
        ds = CooperBenchDataSource(_make_args(setting="baseline", n_samples=1))
        sample = ds.get_samples(num_samples=1)[0][0]
        assert len(sample.prompt) > 0

    @patch("usim.cooperbench.data_source.get_image_name", return_value="fake:latest")
    def test_get_samples_cycles(self, _):
        ds = CooperBenchDataSource(_make_args(setting="baseline", n_samples=1))
        groups = ds.get_samples(num_samples=200)
        assert len(groups) == 200
        # Entry 0 and entry 170 should match (cycled past 170 entries)
        meta_0 = groups[0][0].metadata
        meta_170 = groups[170][0].metadata
        assert meta_0["repo"] == meta_170["repo"]
        assert meta_0["task_id"] == meta_170["task_id"]

    @patch("usim.cooperbench.data_source.get_image_name", return_value="fake:latest")
    def test_factory_function(self, _):
        ds = get_cooperbench_data_source(_make_args(setting="solo"))
        assert isinstance(ds, CooperBenchDataSource)

    def test_invalid_setting_raises(self):
        ds = CooperBenchDataSource(_make_args(setting="invalid"))
        with pytest.raises(ValueError, match="Unknown setting"):
            ds._load_entries()


# ---------------------------------------------------------------------------
# Agent prompt tests
# ---------------------------------------------------------------------------


class TestCooperBenchAgent:

    def test_baseline_no_collaboration(self):
        agent = CooperBenchAgent(agent_id="agent1", setting="baseline")
        assert "team" not in agent.system_prompt.lower()
        assert "send_message" not in agent.system_prompt
        assert agent.messaging_enabled is False

    def test_solo_no_collaboration(self):
        agent = CooperBenchAgent(agent_id="agent1", setting="solo")
        assert "send_message" not in agent.system_prompt
        assert agent.messaging_enabled is False

    def test_coop_has_collaboration(self):
        agent = CooperBenchAgent(
            agent_id="agent1", agents=["agent1", "agent2"],
            messaging_enabled=True, setting="coop",
        )
        assert "agent1" in agent.system_prompt
        assert "agent2" in agent.system_prompt
        assert "send_message" in agent.system_prompt
        # In v2, merge conflict info is in the instance template (not system prompt)
        task_msg = agent.get_task_message("Implement feature X")
        assert "merge" in task_msg.lower()

    def test_coop_no_messaging(self):
        agent = CooperBenchAgent(
            agent_id="agent1", agents=["agent1", "agent2"],
            messaging_enabled=False, setting="coop",
        )
        assert "send_message" not in agent.system_prompt

    def test_task_message_contains_task(self):
        agent = CooperBenchAgent(setting="baseline")
        msg = agent.get_task_message("Implement feature X")
        assert "Implement feature X" in msg
        assert "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in msg

    def test_solo_task_message_combines_features_sorted(self):
        agent = CooperBenchAgent(setting="solo")
        descriptions = {"2": "Feature two desc", "1": "Feature one desc"}
        msg = agent.get_solo_task_message(descriptions)
        assert msg.index("Feature one desc") < msg.index("Feature two desc")
        assert "## Feature 1" in msg
        assert "## Feature 2" in msg

    def test_stop_signal_detected(self):
        msg = Message(role="assistant", content=f"Done. echo {STOP_SIGNAL}")
        assert CooperBenchAgent.is_stop(msg)

    def test_stop_signal_absent(self):
        msg = Message(role="assistant", content="ls -la")
        assert not CooperBenchAgent.is_stop(msg)


# ---------------------------------------------------------------------------
# Reward function tests (mock cooperbench.eval.sandbox to avoid real sandboxes)
# ---------------------------------------------------------------------------


class TestRewardFunctions:
    """Reward functions use lazy imports from cooperbench.eval.sandbox.
    We patch at the source module level."""

    @patch("cooperbench.eval.sandbox.run_patch_test", return_value={"passed": True})
    def test_baseline_reward_pass(self, mock_fn):
        assert compute_baseline_reward("repo", 1, 1, "patch") == 1.0
        mock_fn.assert_called_once()

    @patch("cooperbench.eval.sandbox.run_patch_test", return_value={"passed": False})
    def test_baseline_reward_fail(self, _):
        assert compute_baseline_reward("repo", 1, 1, "patch") == 0.0

    @patch("cooperbench.eval.sandbox.run_patch_test", return_value={"error": "crash"})
    def test_baseline_reward_error(self, _):
        assert compute_baseline_reward("repo", 1, 1, "patch") == 0.0

    @patch("cooperbench.eval.sandbox.test_solo", return_value={"both_passed": True})
    def test_solo_reward_both_pass(self, _):
        assert compute_solo_reward("repo", 1, 1, 2, "patch") == 1.0

    @patch("cooperbench.eval.sandbox.test_solo", return_value={
        "both_passed": False, "feature1": {"passed": True}, "feature2": {"passed": False},
    })
    def test_solo_reward_one_fails(self, _):
        assert compute_solo_reward("repo", 1, 1, 2, "patch") == 0.0

    @patch("cooperbench.eval.sandbox.test_merged", return_value={"both_passed": True})
    def test_coop_reward_both_pass(self, _):
        assert compute_coop_reward("repo", 1, 1, 2, "ap", "pp") == 1.0

    @patch("cooperbench.eval.sandbox.test_merged", return_value={
        "both_passed": False, "feature1": {"passed": True}, "feature2": {"passed": False},
    })
    def test_coop_reward_partial(self, _):
        assert compute_coop_reward("repo", 1, 1, 2, "ap", "pp", partial_reward=True) == 0.5

    @patch("cooperbench.eval.sandbox.test_merged", return_value={
        "both_passed": False, "feature1": {"passed": True}, "feature2": {"passed": False},
    })
    def test_coop_reward_no_partial(self, _):
        assert compute_coop_reward("repo", 1, 1, 2, "ap", "pp", partial_reward=False) == 0.0

    @patch("cooperbench.eval.sandbox.test_merged", return_value={
        "both_passed": False, "feature1": {"passed": False}, "feature2": {"passed": False},
    })
    def test_coop_reward_both_fail(self, _):
        assert compute_coop_reward("repo", 1, 1, 2, "ap", "pp", partial_reward=True) == 0.0

    @patch("cooperbench.eval.sandbox.run_patch_test", side_effect=RuntimeError("boom"))
    def test_baseline_exception_returns_zero(self, _):
        assert compute_baseline_reward("repo", 1, 1, "patch") == 0.0

    @pytest.mark.asyncio
    @patch("cooperbench.eval.sandbox.run_patch_test", return_value={"passed": True})
    async def test_baseline_reward_async(self, _):
        assert await compute_baseline_reward_async("repo", 1, 1, "patch") == 1.0

    @pytest.mark.asyncio
    @patch("cooperbench.eval.sandbox.test_solo", return_value={"both_passed": True})
    async def test_solo_reward_async(self, _):
        assert await compute_solo_reward_async("repo", 1, 1, 2, "patch") == 1.0

    @pytest.mark.asyncio
    @patch("cooperbench.eval.sandbox.test_merged", return_value={"both_passed": True})
    async def test_coop_reward_async(self, _):
        assert await compute_coop_reward_async("repo", 1, 1, 2, "ap", "pp") == 1.0
