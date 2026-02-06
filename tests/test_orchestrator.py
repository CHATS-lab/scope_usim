"""Tests for UserSimOrchestrator."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from usim.core.types import (
    Message,
    Trajectory,
    TrajectoryStatus,
    TrainableRole,
    UserSimConfig,
    UserState,
    AgentState,
)
from usim.core.orchestrator import UserSimOrchestrator


class MockTokenizer:
    """Mock tokenizer for testing."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        # Simple mock: join message contents
        parts = []
        for msg in messages:
            content = msg.get("content", "")
            if content:
                parts.append(f"<{msg['role']}>{content}</{msg['role']}>")
        if add_generation_prompt:
            parts.append("<assistant>")
        return "".join(parts)

    def encode(self, text, add_special_tokens=False):
        # Simple mock: return list of character codes
        return [ord(c) for c in text[:50]]  # Limit to 50 chars


class MockModelAdapter:
    """Mock model adapter for testing."""

    def __init__(self):
        self._tokenizer = MockTokenizer()
        self.generate_response = "Hello, how can I help you?"

    @property
    def tokenizer(self):
        return self._tokenizer

    def apply_template(self, messages, tokenize=True, add_generation_prompt=True, tools=None):
        if tokenize:
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt
            )
            return self._tokenizer.encode(text)
        return self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )

    async def generate_async(self, messages, **kwargs):
        return [{
            "text": self.generate_response,
            "token_ids": [ord(c) for c in self.generate_response],
            "logprobs": [-0.1] * len(self.generate_response),
            "prompt_token_ids": [],
        }]


class MockAgent:
    """Mock agent for testing.

    Implements build_messages() and parse_response() for the new orchestrator flow.
    """

    def __init__(self):
        self.response_count = 0

    def get_init_state(self, message_history=None):
        return AgentState(
            system_messages=[Message(role="system", content="You are a helpful agent.")],
            messages=message_history or [],
        )

    def build_messages(self, state):
        """Build messages for LLM generation."""
        llm_messages = []
        for sys_msg in state.system_messages:
            llm_messages.append(sys_msg.to_dict())
        for msg in state.messages:
            llm_messages.append(msg.to_dict())
        return llm_messages

    def parse_response(self, text, state):
        """Parse generation output into a Message and update state."""
        self.response_count += 1
        if self.response_count >= 3:
            content = "Goodbye! ###STOP###"
        else:
            content = f"Agent response {self.response_count}"
        msg = Message(role="assistant", content=content)
        return msg, state.add_message(msg)

    async def generate_next_message_async(self, user_message, state):
        if user_message is not None:
            state = state.add_message(user_message)
        msgs = self.build_messages(state)
        msg, state = self.parse_response("", state)
        return msg, state

    @classmethod
    def is_stop(cls, message):
        return message.content and "###STOP###" in message.content

    def stop(self, last_message, state):
        pass


class MockUserSimulator:
    """Mock user simulator for testing.

    Implements build_messages() and parse_response() for the new orchestrator flow.
    """

    def __init__(self):
        self.response_count = 0

    def get_init_state(self, message_history=None):
        return UserState(
            system_messages=[Message(role="system", content="You are simulating a user.")],
            messages=message_history or [],
        )

    def build_messages(self, state):
        """Build messages for LLM generation (with flipped roles)."""
        llm_messages = []
        for sys_msg in state.system_messages:
            llm_messages.append(sys_msg.to_dict())
        for msg in state.flip_roles():
            llm_messages.append(msg.to_dict())
        return llm_messages

    def parse_response(self, text, state):
        """Parse generation output into a user Message and update state."""
        self.response_count += 1
        msg = Message(role="user", content=f"User response {self.response_count}")
        return msg, state.add_message(msg)

    async def generate_next_message_async(self, agent_message, state):
        state = state.add_message(agent_message)
        msg, state = self.parse_response("", state)
        return msg, state

    @classmethod
    def is_stop(cls, message):
        return message.content and "###STOP###" in message.content

    def stop(self, last_message, state):
        pass


class TestUserSimOrchestrator:
    """Tests for UserSimOrchestrator."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return UserSimConfig(
            trainable_role=TrainableRole.USER,
            max_turns=10,
            temperature=0.7,
        )

    @pytest.fixture
    def orchestrator(self, config):
        """Create test orchestrator."""
        agent_model = MockModelAdapter()
        user_model = MockModelAdapter()
        return UserSimOrchestrator(
            agent_model=agent_model,
            user_model=user_model,
            config=config,
        )

    @pytest.mark.asyncio
    async def test_run_session_basic(self, orchestrator):
        """Test basic session execution."""
        agent = MockAgent()
        user_sim = MockUserSimulator()
        task = {"id": "test_task", "instructions": "Help the user"}

        trajectory = await orchestrator.run_session(task, agent, user_sim)

        assert isinstance(trajectory, Trajectory)
        assert len(trajectory.messages) > 0
        assert trajectory.turn_count > 0
        # Verify spare convention: loss_mask excludes prompt tokens
        assert len(trajectory.tokens) >= len(trajectory.loss_mask)
        # Verify invariant: loss_mask and logprobs same length
        assert len(trajectory.loss_mask) == len(trajectory.rollout_log_probs)

    @pytest.mark.asyncio
    async def test_run_session_stops_on_agent_stop(self, orchestrator):
        """Test session stops when agent signals stop."""
        agent = MockAgent()
        user_sim = MockUserSimulator()
        task = {"id": "test_task", "instructions": "Help the user"}

        trajectory = await orchestrator.run_session(task, agent, user_sim)

        # Agent should have stopped after 3 responses (via parse_response counter)
        assert agent.response_count == 3

    @pytest.mark.asyncio
    async def test_run_session_trajectory_status(self, orchestrator):
        """Test trajectory has correct status."""
        agent = MockAgent()
        user_sim = MockUserSimulator()
        task = {"id": "test_task", "instructions": "Help the user"}

        trajectory = await orchestrator.run_session(task, agent, user_sim)

        # Agent stops via ###STOP### so status should be COMPLETED
        assert trajectory.status == TrajectoryStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_run_session_logprobs_tracked(self, orchestrator):
        """Test that logprobs are tracked alongside loss_mask."""
        agent = MockAgent()
        user_sim = MockUserSimulator()
        task = {"id": "test_task", "instructions": "Help the user"}

        trajectory = await orchestrator.run_session(task, agent, user_sim)

        # rollout_log_probs should be populated
        assert len(trajectory.rollout_log_probs) > 0
        assert len(trajectory.rollout_log_probs) == len(trajectory.loss_mask)

    def test_compute_loss_mask_user_training(self, orchestrator):
        """Test loss mask computation for user training."""
        orchestrator.config.trainable_role = TrainableRole.USER

        assert orchestrator._compute_loss_mask_for_role("user") == 1
        assert orchestrator._compute_loss_mask_for_role("assistant") == 0
        assert orchestrator._compute_loss_mask_for_role("tool") == 0

    def test_compute_loss_mask_agent_training(self, orchestrator):
        """Test loss mask computation for agent training."""
        orchestrator.config.trainable_role = TrainableRole.AGENT

        assert orchestrator._compute_loss_mask_for_role("assistant") == 1
        assert orchestrator._compute_loss_mask_for_role("user") == 0
        assert orchestrator._compute_loss_mask_for_role("tool") == 0

    def test_compute_loss_mask_both_training(self, orchestrator):
        """Test loss mask computation for training both roles."""
        orchestrator.config.trainable_role = TrainableRole.BOTH

        assert orchestrator._compute_loss_mask_for_role("assistant") == 1
        assert orchestrator._compute_loss_mask_for_role("user") == 1
        assert orchestrator._compute_loss_mask_for_role("tool") == 0


class TestTokenTracking:
    """Tests for token tracking functionality."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator for token tracking tests."""
        config = UserSimConfig(trainable_role=TrainableRole.USER)
        return UserSimOrchestrator(
            agent_model=MockModelAdapter(),
            user_model=MockModelAdapter(),
            config=config,
        )

    def test_get_token_delta_assistant(self, orchestrator):
        """Test token delta calculation for assistant messages."""
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]

        tokens, mask = orchestrator._get_token_delta(
            messages, orchestrator.agent_model, "assistant"
        )

        assert len(tokens) > 0
        assert len(tokens) == len(mask)
        # For USER training, assistant should have mask=0
        assert all(m == 0 for m in mask)

    def test_get_token_delta_user(self, orchestrator):
        """Test token delta calculation for user messages."""
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "Hi there"},
        ]

        tokens, mask = orchestrator._get_token_delta(
            messages, orchestrator.user_model, "user"
        )

        assert len(tokens) > 0
        assert len(tokens) == len(mask)
        # For USER training, user should have mask=1
        assert all(m == 1 for m in mask)

    def test_extract_logprobs_exact(self, orchestrator):
        """Test logprob extraction with exact length match."""
        results = [{"text": "Hi", "logprobs": [-0.1, -0.2, -0.3]}]
        logprobs = orchestrator._extract_logprobs(results, 3)
        assert logprobs == [-0.1, -0.2, -0.3]

    def test_extract_logprobs_padding(self, orchestrator):
        """Test logprob extraction with padding needed."""
        results = [{"text": "Hi", "logprobs": [-0.1]}]
        logprobs = orchestrator._extract_logprobs(results, 3)
        assert logprobs == [-0.1, 0.0, 0.0]

    def test_extract_logprobs_error(self, orchestrator):
        """Test logprob extraction on error results."""
        results = [{"text": "", "error": "failed"}]
        logprobs = orchestrator._extract_logprobs(results, 3)
        assert logprobs == [0.0, 0.0, 0.0]

    def test_extract_logprobs_missing(self, orchestrator):
        """Test logprob extraction with missing logprobs key."""
        results = [{"text": "Hi"}]
        logprobs = orchestrator._extract_logprobs(results, 3)
        assert logprobs == [0.0, 0.0, 0.0]
