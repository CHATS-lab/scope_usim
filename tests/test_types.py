"""Tests for core types."""

import pytest
from usim.core.types import (
    Message,
    ToolCall,
    Trajectory,
    TrajectoryStatus,
    TrainableRole,
    UserSimConfig,
    UserState,
    AgentState,
    compute_token_delta,
)


class TestMessage:
    """Tests for Message dataclass."""

    def test_message_creation(self):
        """Test basic message creation."""
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.tool_calls is None

    def test_message_with_tool_calls(self):
        """Test message with tool calls."""
        tool_call = ToolCall(
            id="tc1",
            name="get_user",
            arguments={"user_id": "123"},
        )
        msg = Message(role="assistant", tool_calls=[tool_call])
        assert msg.is_tool_call()
        assert not msg.has_text_content()

    def test_message_to_dict(self):
        """Test message serialization."""
        msg = Message(role="user", content="Hello")
        d = msg.to_dict()
        assert d["role"] == "user"
        assert d["content"] == "Hello"

    def test_message_from_dict(self):
        """Test message deserialization."""
        d = {"role": "assistant", "content": "Hi there!"}
        msg = Message.from_dict(d)
        assert msg.role == "assistant"
        assert msg.content == "Hi there!"


class TestToolCall:
    """Tests for ToolCall dataclass."""

    def test_tool_call_creation(self):
        """Test basic tool call creation."""
        tc = ToolCall(
            id="tc1",
            name="search",
            arguments={"query": "test"},
        )
        assert tc.name == "search"
        assert tc.arguments["query"] == "test"
        assert tc.requestor == "assistant"

    def test_tool_call_to_dict(self):
        """Test tool call serialization."""
        tc = ToolCall(
            id="tc1",
            name="search",
            arguments={"query": "test"},
            requestor="user",
        )
        d = tc.to_dict()
        assert d["name"] == "search"
        assert d["requestor"] == "user"


class TestTrainableRole:
    """Tests for TrainableRole enum."""

    def test_role_values(self):
        """Test enum values."""
        assert TrainableRole.AGENT.value == "agent"
        assert TrainableRole.USER.value == "user"
        assert TrainableRole.BOTH.value == "both"

    def test_role_from_string(self):
        """Test creating role from string."""
        role = TrainableRole("user")
        assert role == TrainableRole.USER


class TestTrajectoryStatus:
    """Tests for TrajectoryStatus enum."""

    def test_status_values(self):
        """Test enum values."""
        assert TrajectoryStatus.PENDING.value == "pending"
        assert TrajectoryStatus.COMPLETED.value == "completed"
        assert TrajectoryStatus.TRUNCATED.value == "truncated"
        assert TrajectoryStatus.FAILED.value == "failed"
        assert TrajectoryStatus.ABORTED.value == "aborted"
        assert TrajectoryStatus.TIMEOUT.value == "timeout"
        assert TrajectoryStatus.RUNNING.value == "running"

    def test_status_is_string_enum(self):
        """Test that TrajectoryStatus is a string enum."""
        assert isinstance(TrajectoryStatus.COMPLETED, str)
        assert TrajectoryStatus.COMPLETED == "completed"


class TestUserSimConfig:
    """Tests for UserSimConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = UserSimConfig()
        assert config.temperature == 0.7
        assert config.max_turns == 30
        assert config.trainable_role == TrainableRole.USER
        assert "###STOP###" in config.stop_tokens

    def test_custom_config(self):
        """Test custom configuration."""
        config = UserSimConfig(
            temperature=0.5,
            max_turns=50,
            trainable_role=TrainableRole.BOTH,
        )
        assert config.temperature == 0.5
        assert config.max_turns == 50
        assert config.trainable_role == TrainableRole.BOTH


class TestTrajectory:
    """Tests for Trajectory dataclass."""

    def test_empty_trajectory(self):
        """Test empty trajectory."""
        traj = Trajectory()
        assert traj.tokens == []
        assert traj.loss_mask == []
        assert traj.response_length == 0
        assert traj.rollout_log_probs == []
        assert traj.index == 0
        assert traj.status == TrajectoryStatus.PENDING

    def test_trajectory_with_data(self):
        """Test trajectory with data."""
        traj = Trajectory(
            tokens=[1, 2, 3, 4, 5],
            loss_mask=[1, 1, 0, 1, 0],
            rollout_log_probs=[-0.1, -0.2, 0.0, -0.3, 0.0],
            response="Hello",
            response_length=3,
            reward=1.0,
            status=TrajectoryStatus.COMPLETED,
        )
        assert len(traj.tokens) == 5
        assert traj.response_length == 3
        assert len(traj.rollout_log_probs) == len(traj.loss_mask)
        assert traj.status == TrajectoryStatus.COMPLETED

    def test_trajectory_spare_convention(self):
        """Test that loss_mask can be shorter than tokens (spare convention)."""
        # Prompt tokens (5) + response tokens (3) = 8 total
        # loss_mask only covers response tokens
        traj = Trajectory(
            tokens=[10, 20, 30, 40, 50, 60, 70, 80],
            loss_mask=[1, 1, 0],
            rollout_log_probs=[-0.1, -0.2, 0.0],
            response_length=2,
        )
        assert len(traj.tokens) == 8
        assert len(traj.loss_mask) == 3
        assert len(traj.rollout_log_probs) == len(traj.loss_mask)
        # base_offset = len(tokens) - len(loss_mask) = 5 (prompt length)
        base_offset = len(traj.tokens) - len(traj.loss_mask)
        assert base_offset == 5

    def test_trajectory_new_fields(self):
        """Test new fields: index, rollout_log_probs, status."""
        traj = Trajectory(
            index=42,
            tokens=[1, 2, 3],
            loss_mask=[1, 0, 1],
            rollout_log_probs=[-0.5, 0.0, -0.3],
            response_length=2,
            status=TrajectoryStatus.TRUNCATED,
        )
        assert traj.index == 42
        assert traj.rollout_log_probs == [-0.5, 0.0, -0.3]
        assert traj.status == TrajectoryStatus.TRUNCATED


class TestComputeTokenDelta:
    """Tests for compute_token_delta function."""

    class SimpleTokenizer:
        """Simple tokenizer for testing."""

        def apply_chat_template(
            self, messages, tokenize=False, add_generation_prompt=True, **kwargs
        ):
            parts = []
            for msg in messages:
                content = msg.get("content", "")
                if content:
                    parts.append(f"<{msg['role']}>{content}</{msg['role']}>")
            if add_generation_prompt:
                parts.append("<assistant>")
            return "".join(parts)

        def encode(self, text, add_special_tokens=False):
            return [ord(c) for c in text[:50]]

    def test_assistant_message_delta(self):
        """Test token delta for assistant messages produces loss_mask=1."""
        tokenizer = self.SimpleTokenizer()
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        tokens, mask = compute_token_delta(tokenizer, messages)
        assert len(tokens) > 0
        assert len(tokens) == len(mask)
        assert all(m == 1 for m in mask)

    def test_user_message_delta(self):
        """Test token delta for user messages produces loss_mask=0."""
        tokenizer = self.SimpleTokenizer()
        messages = [
            {"role": "system", "content": "System"},
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "Hi"},
        ]
        tokens, mask = compute_token_delta(tokenizer, messages)
        assert len(tokens) > 0
        assert len(tokens) == len(mask)
        assert all(m == 0 for m in mask)


class TestUserState:
    """Tests for UserState dataclass."""

    def test_add_message(self):
        """Test adding message to state."""
        state = UserState()
        msg = Message(role="user", content="Hello")
        new_state = state.add_message(msg)

        assert len(new_state.messages) == 1
        assert len(state.messages) == 0  # Original unchanged

    def test_flip_roles(self):
        """Test role flipping for user simulator perspective."""
        state = UserState(
            messages=[
                Message(role="assistant", content="Hi"),
                Message(role="user", content="Hello"),
            ]
        )
        flipped = state.flip_roles()

        assert flipped[0].role == "user"
        assert flipped[1].role == "assistant"


class TestAgentState:
    """Tests for AgentState dataclass."""

    def test_add_message(self):
        """Test adding message to agent state."""
        state = AgentState()
        msg = Message(role="assistant", content="Hello")
        new_state = state.add_message(msg)

        assert len(new_state.messages) == 1
        assert len(state.messages) == 0
