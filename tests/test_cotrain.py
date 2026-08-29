"""Tests for co-training orchestrator and checkpoint pool."""

import json
import os
import tempfile

import pytest

from usim.core.checkpoint_pool import CheckpointPool
from usim.core.types import TrainingMode, TrajectoryStatus, UserSimConfig, TrainableRole


class MockTokenizer:
    """Mock tokenizer for co-training tests."""

    eos_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
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

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": self.encode(text, add_special_tokens)}


class MockP4gEnvironment:
    """Mock P4G environment with opponent_generate_fn support."""

    def __init__(self, opponent_generate_fn=None, num_exchanges=2):
        self._opponent_generate_fn = opponent_generate_fn
        self._num_exchanges = num_exchanges
        self._exchange_count = 0
        self._persuadee_messages = []
        self._persuadee_system = "You are the persuadee."
        self.last_opponent_output = None

    async def reset(self):
        self._exchange_count = 0
        self._persuadee_messages = [
            {"role": "system", "content": self._persuadee_system}
        ]
        self.last_opponent_output = None
        initial_messages = [{"role": "system", "content": "You are the persuader."}]
        return initial_messages, None, {"id": "test", "domain": "p4g"}

    async def step(self, action):
        # Add persuader's message to persuadee's view
        self._persuadee_messages.append({"role": "user", "content": action})

        # Generate persuadee response
        if self._opponent_generate_fn is not None:
            output = await self._opponent_generate_fn(
                self._persuadee_messages,
                {"temperature": 0.7, "max_new_tokens": 100},
            )
            self.last_opponent_output = output
            response = output["text"]
        else:
            response = f"Persuadee response {self._exchange_count + 1}"
            self.last_opponent_output = None

        self._persuadee_messages.append({"role": "assistant", "content": response})
        self._exchange_count += 1

        terminated = self._exchange_count >= self._num_exchanges
        return response, 0.5 if terminated else 0.0, terminated, False, {
            "exchange": self._exchange_count,
        }

    def parse_response(self, response_text):
        return {"normal_text": response_text, "calls": []}

    @property
    def prompt_postprocess_fn(self):
        return None

    @property
    def persuadee_system_prompt(self):
        return self._persuadee_system

    @property
    def persuadee_messages(self):
        return self._persuadee_messages


class TestTrainingMode:
    """Tests for TrainingMode enum."""

    def test_enum_values(self):
        assert TrainingMode.SINGLE == "single"
        assert TrainingMode.COTRAIN == "cotrain"
        assert TrainingMode.SELFPLAY == "selfplay"

    def test_from_string(self):
        assert TrainingMode("single") == TrainingMode.SINGLE
        assert TrainingMode("cotrain") == TrainingMode.COTRAIN
        assert TrainingMode("selfplay") == TrainingMode.SELFPLAY


class TestCheckpointPool:
    """Tests for CheckpointPool."""

    @pytest.fixture
    def pool_dir(self):
        with tempfile.TemporaryDirectory() as d:
            yield d

    def test_add_and_len(self, pool_dir):
        pool = CheckpointPool(pool_dir, max_size=5)
        assert len(pool) == 0

        pool.add("/ckpt/step_0", step=0)
        assert len(pool) == 1

        pool.add("/ckpt/step_16", step=16)
        assert len(pool) == 2

    def test_sample_random(self, pool_dir):
        pool = CheckpointPool(pool_dir, max_size=5)
        pool.add("/ckpt/step_0", step=0)
        pool.add("/ckpt/step_16", step=16)

        ckpt = pool.sample("random")
        assert ckpt in ["/ckpt/step_0", "/ckpt/step_16"]

    def test_sample_latest(self, pool_dir):
        pool = CheckpointPool(pool_dir, max_size=5)
        pool.add("/ckpt/step_0", step=0)
        pool.add("/ckpt/step_16", step=16)

        assert pool.sample("latest") == "/ckpt/step_16"
        assert pool.latest() == "/ckpt/step_16"

    def test_fifo_eviction(self, pool_dir):
        pool = CheckpointPool(pool_dir, max_size=3)
        pool.add("/ckpt/step_0", step=0)
        pool.add("/ckpt/step_1", step=1)
        pool.add("/ckpt/step_2", step=2)
        assert len(pool) == 3

        # Adding a 4th should evict step_0
        pool.add("/ckpt/step_3", step=3)
        assert len(pool) == 3
        assert pool.sample("latest") == "/ckpt/step_3"

        # step_0 should be gone
        all_paths = [pool.sample("random") for _ in range(100)]
        assert "/ckpt/step_0" not in all_paths

    def test_persistence(self, pool_dir):
        pool = CheckpointPool(pool_dir, max_size=5)
        pool.add("/ckpt/step_0", step=0)
        pool.add("/ckpt/step_16", step=16)

        # Create new pool from same dir
        pool2 = CheckpointPool(pool_dir, max_size=5)
        assert len(pool2) == 2
        assert pool2.latest() == "/ckpt/step_16"

    def test_empty_pool_raises(self, pool_dir):
        pool = CheckpointPool(pool_dir, max_size=5)
        with pytest.raises(ValueError, match="empty"):
            pool.sample()

    def test_duplicate_step_updates(self, pool_dir):
        pool = CheckpointPool(pool_dir, max_size=5)
        pool.add("/ckpt/v1/step_0", step=0)
        pool.add("/ckpt/v2/step_0", step=0)  # same step, different path
        assert len(pool) == 1
        assert pool.latest() == "/ckpt/v2/step_0"

    def test_manifest_format(self, pool_dir):
        pool = CheckpointPool(pool_dir, max_size=5)
        pool.add("/ckpt/step_0", step=0)

        manifest_path = os.path.join(pool_dir, "pool_manifest.json")
        with open(manifest_path) as f:
            data = json.load(f)

        assert "entries" in data
        assert len(data["entries"]) == 1
        assert data["entries"][0]["path"] == "/ckpt/step_0"
        assert data["entries"][0]["step"] == 0


class TestCoTrainingOrchestrator:
    """Tests for CoTrainingOrchestrator."""

    @pytest.fixture
    def tokenizer(self):
        return MockTokenizer()

    @pytest.fixture
    def config(self):
        return UserSimConfig(
            trainable_role=TrainableRole.AGENT,
            max_turns=3,
            max_tokens=100,
            max_context_length=10000,
        )

    @pytest.mark.asyncio
    async def test_dual_trajectory_basic(self, tokenizer, config):
        """Test that rollout produces two trajectories."""
        from usim.core.co_training_orchestrator import CoTrainingOrchestrator

        agent_call_count = 0
        opponent_call_count = 0

        async def agent_generate_fn(input_ids, sampling_params):
            nonlocal agent_call_count
            agent_call_count += 1
            text = f"Persuader message {agent_call_count}"
            tokens = [ord(c) for c in text[:20]] + [tokenizer.eos_token_id]
            return {
                "text": text,
                "token_ids": tokens,
                "logprobs": [-0.1] * len(tokens),
                "meta_info": {},
            }

        async def opponent_generate_fn(messages, sampling_params):
            nonlocal opponent_call_count
            opponent_call_count += 1
            text = f"Persuadee response {opponent_call_count}"
            tokens = [ord(c) for c in text[:20]] + [tokenizer.eos_token_id]
            return {
                "text": text,
                "token_ids": tokens,
                "logprobs": [-0.2] * len(tokens),
                "meta_info": {},
            }

        env = MockP4gEnvironment(
            opponent_generate_fn=opponent_generate_fn,
            num_exchanges=2,
        )

        orchestrator = CoTrainingOrchestrator(
            agent_tokenizer=tokenizer,
            opponent_tokenizer=tokenizer,
            config=config,
        )

        agent_traj, opponent_traj = await orchestrator.rollout(
            env, agent_generate_fn, {}
        )

        # Both trajectories should exist
        assert agent_traj is not None
        assert opponent_traj is not None

        # Both should have tokens
        assert len(agent_traj.tokens) > 0
        assert len(opponent_traj.tokens) > 0

        # Both should have masks matching logprobs
        assert len(agent_traj.loss_mask) == len(agent_traj.rollout_log_probs)
        assert len(opponent_traj.loss_mask) == len(opponent_traj.rollout_log_probs)

        # Agent trajectory: agent tokens should have mask=1
        assert sum(agent_traj.loss_mask) > 0
        # Opponent trajectory: opponent tokens should have mask=1
        assert sum(opponent_traj.loss_mask) > 0

    @pytest.mark.asyncio
    async def test_agent_loss_mask_complement(self, tokenizer, config):
        """Test that agent and opponent loss masks are complementary."""
        from usim.core.co_training_orchestrator import CoTrainingOrchestrator

        async def agent_generate_fn(input_ids, sampling_params):
            text = "Hello persuadee"
            tokens = [1, 2, 3, 4, 5, tokenizer.eos_token_id]
            return {
                "text": text,
                "token_ids": tokens,
                "logprobs": [-0.1] * len(tokens),
                "meta_info": {},
            }

        async def opponent_generate_fn(messages, sampling_params):
            text = "I'm not donating"
            tokens = [6, 7, 8, 9, tokenizer.eos_token_id]
            return {
                "text": text,
                "token_ids": tokens,
                "logprobs": [-0.2] * len(tokens),
                "meta_info": {},
            }

        env = MockP4gEnvironment(
            opponent_generate_fn=opponent_generate_fn,
            num_exchanges=1,
        )

        orchestrator = CoTrainingOrchestrator(
            agent_tokenizer=tokenizer,
            opponent_tokenizer=tokenizer,
            config=config,
        )

        agent_traj, opponent_traj = await orchestrator.rollout(
            env, agent_generate_fn, {}
        )

        # Agent: mask=1 on agent tokens, mask=0 on opponent tokens
        # So agent mask should NOT be all-0 or all-1
        assert 0 in agent_traj.loss_mask
        assert 1 in agent_traj.loss_mask

        # Opponent: mask=1 on opponent tokens, mask=0 on agent tokens
        assert 0 in opponent_traj.loss_mask
        assert 1 in opponent_traj.loss_mask

    @pytest.mark.asyncio
    async def test_opponent_logprobs_from_generate_fn(self, tokenizer, config):
        """Test that opponent logprobs come from opponent_generate_fn output."""
        from usim.core.co_training_orchestrator import CoTrainingOrchestrator

        async def agent_generate_fn(input_ids, sampling_params):
            return {
                "text": "Agent says hi",
                "token_ids": [1, 2, 3, tokenizer.eos_token_id],
                "logprobs": [-0.1, -0.1, -0.1, -0.1],
                "meta_info": {},
            }

        async def opponent_generate_fn(messages, sampling_params):
            return {
                "text": "Opponent says no",
                "token_ids": [4, 5, 6, tokenizer.eos_token_id],
                "logprobs": [-0.5, -0.5, -0.5, -0.5],
                "meta_info": {},
            }

        env = MockP4gEnvironment(
            opponent_generate_fn=opponent_generate_fn,
            num_exchanges=1,
        )

        orchestrator = CoTrainingOrchestrator(
            agent_tokenizer=tokenizer,
            opponent_tokenizer=tokenizer,
            config=config,
        )

        agent_traj, opponent_traj = await orchestrator.rollout(
            env, agent_generate_fn, {}
        )

        # Agent logprobs should contain -0.1 values (from agent generate_fn)
        agent_nonzero = [lp for lp in agent_traj.rollout_log_probs if lp != 0.0]
        assert all(abs(lp - (-0.1)) < 1e-6 for lp in agent_nonzero)

        # Opponent logprobs should contain -0.5 values (from opponent generate_fn)
        opp_nonzero = [lp for lp in opponent_traj.rollout_log_probs if lp != 0.0]
        assert all(abs(lp - (-0.5)) < 1e-6 for lp in opp_nonzero)

    @pytest.mark.asyncio
    async def test_failed_generation_returns_failed_status(self, tokenizer, config):
        """Test that a failed agent generation results in FAILED trajectories."""
        from usim.core.co_training_orchestrator import CoTrainingOrchestrator

        async def failing_generate_fn(input_ids, sampling_params):
            raise RuntimeError("Generation failed")

        async def opponent_generate_fn(messages, sampling_params):
            return {
                "text": "response",
                "token_ids": [1],
                "logprobs": [-0.1],
                "meta_info": {},
            }

        env = MockP4gEnvironment(
            opponent_generate_fn=opponent_generate_fn,
            num_exchanges=2,
        )

        orchestrator = CoTrainingOrchestrator(
            agent_tokenizer=tokenizer,
            opponent_tokenizer=tokenizer,
            config=config,
        )

        agent_traj, opponent_traj = await orchestrator.rollout(
            env, failing_generate_fn, {}
        )

        assert agent_traj.status == TrajectoryStatus.FAILED
        assert opponent_traj.status == TrajectoryStatus.FAILED


class TestP4gEnvironmentOpponentFn:
    """Tests for P4gEnvironment with pluggable opponent_generate_fn."""

    @pytest.mark.asyncio
    async def test_opponent_generate_fn_replaces_api(self):
        """Test that opponent_generate_fn is called instead of litellm."""
        from usim.core.environment.p4g.environment import P4gEnvironment

        call_count = 0

        async def mock_opponent_fn(messages, sampling_params):
            nonlocal call_count
            call_count += 1
            return {
                "text": f"Generated response {call_count}",
                "token_ids": [1, 2, 3],
                "logprobs": [-0.1, -0.2, -0.3],
                "meta_info": {},
            }

        env = P4gEnvironment(
            persuader_persona="<persona>Persuader</persona>",
            persuadee_persona="<persona>Persuadee</persona>",
            num_turns=2,
            opponent_generate_fn=mock_opponent_fn,
        )

        await env.reset()
        obs, reward, terminated, truncated, info = await env.step("Hello persuadee!")

        assert call_count == 1
        assert "Generated response 1" in obs
        assert env.last_opponent_output is not None
        assert env.last_opponent_output["token_ids"] == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_last_opponent_output_cleared_on_reset(self):
        """Test that last_opponent_output is cleared on reset."""
        from usim.core.environment.p4g.environment import P4gEnvironment

        async def mock_opponent_fn(messages, sampling_params):
            return {
                "text": "response",
                "token_ids": [1],
                "logprobs": [-0.1],
                "meta_info": {},
            }

        env = P4gEnvironment(
            persuader_persona="<persona>P</persona>",
            persuadee_persona="<persona>P</persona>",
            num_turns=2,
            opponent_generate_fn=mock_opponent_fn,
        )

        await env.reset()
        await env.step("message")
        assert env.last_opponent_output is not None

        await env.reset()
        assert env.last_opponent_output is None

    @pytest.mark.asyncio
    async def test_api_path_when_no_opponent_fn(self):
        """Test that API path is used when opponent_generate_fn is None."""
        from usim.core.environment.p4g.environment import P4gEnvironment

        env = P4gEnvironment(
            persuader_persona="<persona>P</persona>",
            persuadee_persona="<persona>P</persona>",
            num_turns=2,
            opponent_generate_fn=None,
            persuadee_model="test-model",
        )

        assert env._opponent_generate_fn is None
        # last_opponent_output should be None after API call
        # (can't actually call API in tests, just verify the attribute exists)
        assert env.last_opponent_output is None
