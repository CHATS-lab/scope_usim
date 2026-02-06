"""Tests for utility functions."""

import pytest

from usim.core.types import Message, Trajectory, TrainableRole
from usim.core.utils.message_utils import (
    flip_message_roles,
    messages_to_dict_list,
    filter_messages_by_role,
)
from usim.core.utils.trajectory_utils import (
    compute_loss_mask,
    trajectory_to_dict,
    validate_trajectory,
)


class TestMessageUtils:
    """Tests for message utility functions."""

    def test_flip_message_roles(self):
        """Test flipping user/assistant roles."""
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi"),
            Message(role="system", content="System"),
        ]

        flipped = flip_message_roles(messages)

        assert flipped[0].role == "assistant"
        assert flipped[1].role == "user"
        assert flipped[2].role == "system"  # System unchanged

    def test_messages_to_dict_list(self):
        """Test converting messages to dict format."""
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi"),
        ]

        dicts = messages_to_dict_list(messages)

        assert len(dicts) == 2
        assert dicts[0]["role"] == "user"
        assert dicts[1]["content"] == "Hi"

    def test_filter_messages_by_role(self):
        """Test filtering messages by role."""
        messages = [
            Message(role="user", content="User 1"),
            Message(role="assistant", content="Assistant 1"),
            Message(role="user", content="User 2"),
            Message(role="tool", content="Tool result"),
        ]

        user_msgs = filter_messages_by_role(messages, ["user"])
        assert len(user_msgs) == 2

        assistant_msgs = filter_messages_by_role(messages, ["assistant"])
        assert len(assistant_msgs) == 1


class TestTrajectoryUtils:
    """Tests for trajectory utility functions."""

    def test_compute_loss_mask_user_training(self):
        """Test loss mask computation for user training."""
        mask = compute_loss_mask("user", 5, TrainableRole.USER)
        assert mask == [1, 1, 1, 1, 1]

        mask = compute_loss_mask("assistant", 5, TrainableRole.USER)
        assert mask == [0, 0, 0, 0, 0]

    def test_compute_loss_mask_agent_training(self):
        """Test loss mask computation for agent training."""
        mask = compute_loss_mask("assistant", 5, TrainableRole.AGENT)
        assert mask == [1, 1, 1, 1, 1]

        mask = compute_loss_mask("user", 5, TrainableRole.AGENT)
        assert mask == [0, 0, 0, 0, 0]

    def test_compute_loss_mask_both_training(self):
        """Test loss mask computation for training both."""
        mask = compute_loss_mask("user", 3, TrainableRole.BOTH)
        assert mask == [1, 1, 1]

        mask = compute_loss_mask("assistant", 3, TrainableRole.BOTH)
        assert mask == [1, 1, 1]

    def test_compute_loss_mask_tool_never_trained(self):
        """Test that tool messages are never trained."""
        for role in [TrainableRole.USER, TrainableRole.AGENT, TrainableRole.BOTH]:
            mask = compute_loss_mask("tool", 5, role)
            assert mask == [0, 0, 0, 0, 0]

    def test_trajectory_to_dict(self):
        """Test trajectory serialization."""
        traj = Trajectory(
            prompt="Test prompt",
            tokens=[1, 2, 3],
            loss_mask=[0, 1, 1],
            response="Response",
            reward=1.0,
        )

        d = trajectory_to_dict(traj)

        assert d["prompt"] == "Test prompt"
        assert d["tokens"] == [1, 2, 3]
        assert d["reward"] == 1.0

    def test_validate_trajectory_valid(self):
        """Test validation of valid trajectory."""
        traj = Trajectory(
            tokens=[1, 2, 3],
            loss_mask=[0, 1, 1],
        )

        errors = validate_trajectory(traj)
        assert len(errors) == 0

    def test_validate_trajectory_length_mismatch(self):
        """Test validation catches length mismatch (tokens < loss_mask)."""
        traj = Trajectory(
            tokens=[1, 2],
            loss_mask=[0, 1, 1],  # More mask values than tokens
        )

        errors = validate_trajectory(traj)
        assert any("mismatch" in e.lower() for e in errors)

    def test_validate_trajectory_spare_convention(self):
        """Test validation passes when tokens > loss_mask (spare convention)."""
        traj = Trajectory(
            tokens=[1, 2, 3, 4, 5],
            loss_mask=[0, 1, 1],  # Fewer mask values = prompt tokens excluded
        )

        errors = validate_trajectory(traj)
        # Should not report length mismatch
        assert not any("mismatch" in e.lower() for e in errors)

    def test_validate_trajectory_empty(self):
        """Test validation catches empty trajectory."""
        traj = Trajectory()

        errors = validate_trajectory(traj)
        assert any("no tokens" in e.lower() for e in errors)

    def test_validate_trajectory_no_trainable(self):
        """Test validation warns about no trainable tokens."""
        traj = Trajectory(
            tokens=[1, 2, 3],
            loss_mask=[0, 0, 0],
        )

        errors = validate_trajectory(traj)
        assert any("no trainable" in e.lower() for e in errors)
