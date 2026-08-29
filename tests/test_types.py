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
    get_token_delta,
    probe_inter_message_glue,
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

    def test_user_delta_includes_generation_prompt(self):
        """Regression test for Fix 1: user-message delta MUST include the
        assistant generation prompt. Otherwise the next generate_fn call is
        fed a prompt with no assistant cue and the model goes OOD.
        """
        tokenizer = self.SimpleTokenizer()
        messages = [
            {"role": "user", "content": "obs1"},
            {"role": "assistant", "content": "resp1"},
            {"role": "user", "content": "obs2"},
        ]
        tokens, mask = get_token_delta(tokenizer, messages)
        decoded = "".join(chr(t) for t in tokens)
        assert "<assistant>" in decoded, (
            f"user-message delta must include the generation prompt; got: {decoded!r}"
        )
        assert all(m == 0 for m in mask), "user-message delta must not be trainable"

    def test_assistant_delta_excludes_extra_generation_prompt(self):
        """Assistant-message delta must NOT include a duplicate generation
        prompt (the previous state already has it)."""
        tokenizer = self.SimpleTokenizer()
        messages = [
            {"role": "user", "content": "obs1"},
            {"role": "assistant", "content": "resp1"},
        ]
        tokens, mask = get_token_delta(tokenizer, messages)
        decoded = "".join(chr(t) for t in tokens)
        # The prev state ended with <assistant>, so curr's delta should
        # contain resp1's content and its </assistant> closing tag, not a
        # second <assistant> opening tag.
        assert "resp1" in decoded
        assert decoded.count("<assistant>") == 0, (
            f"assistant delta should not re-introduce generation prompt; got: {decoded!r}"
        )
        assert all(m == 1 for m in mask), "assistant delta must be trainable"


class TestMultiTurnRoundTrip:
    """Round-trip tests that exercise incremental delta building.

    These tests use a real HF tokenizer to catch template-specific bugs that
    the SimpleTokenizer would miss. Skipped if transformers isn't installed.
    """

    @pytest.fixture
    def tokenizer(self):
        transformers = pytest.importorskip("transformers")
        try:
            return transformers.AutoTokenizer.from_pretrained(
                "Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True
            )
        except Exception as e:
            pytest.skip(f"Could not load test tokenizer: {e}")

    def test_multi_turn_round_trip(self, tokenizer):
        """Incrementally-built buffer must exactly equal batch re-tokenization.

        This is the strongest possible check for the token delta helper: if
        you build all_tokens by appending the initial prompt + per-turn deltas,
        the result should be identical to re-tokenizing the full conversation
        at the end with add_generation_prompt=False.
        """
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's the capital of France?"},
        ]

        # Turn 0: initial prompt with generation prompt
        initial = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        all_tokens = list(initial)

        # Simulate assistant response
        assistant_content = "The capital of France is Paris."
        messages.append({"role": "assistant", "content": assistant_content})

        # In a real rollout, response tokens come from the inference server.
        # We simulate that by computing the assistant delta.
        asst_delta, asst_mask = get_token_delta(tokenizer, messages)
        all_tokens.extend(asst_delta)

        # Turn 1: append a new user obs
        messages.append({"role": "user", "content": "And Germany?"})
        user_delta, user_mask = get_token_delta(tokenizer, messages)
        all_tokens.extend(user_delta)

        # The user-message delta MUST include the next assistant generation
        # prompt so that subsequent generation is in-distribution.
        decoded_delta = tokenizer.decode(user_delta)
        assert "assistant" in decoded_delta.lower(), (
            f"user-message delta must include generation prompt; got: {decoded_delta!r}"
        )

        # Final reference: re-tokenize the full conversation
        reference = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )

        assert all_tokens == list(reference), (
            "Incrementally built buffer must equal batch re-tokenization. "
            f"len(incremental)={len(all_tokens)}, len(reference)={len(reference)}"
        )

    def test_probe_inter_message_glue(self, tokenizer):
        """Fix 2: probe must return the template's post-EOS glue.

        For Qwen tokenizers this is typically [198] (a single newline).
        """
        glue = probe_inter_message_glue(tokenizer)
        # Glue should be either empty or a short sequence (usually just \n)
        assert isinstance(glue, list)
        assert len(glue) <= 4, f"Glue unexpectedly long: {glue}"


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
