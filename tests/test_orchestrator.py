"""Tests for the current Gym-style user-simulation orchestrator."""

import json

import pytest

from usim.core.orchestrator import UserSimOrchestrator
from usim.core.types import TrajectoryStatus, UserSimConfig


class MockTokenizer:
    """Small chat tokenizer with the interface used by the orchestrator."""

    eos_token_id = 0

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **kwargs,
    ):
        text = "".join(
            f"<{message['role']}>{message.get('content', '')}</{message['role']}>"
            for message in messages
        )
        if add_generation_prompt:
            text += "<assistant>"
        if tokenize:
            return self.encode(text) + [self.eos_token_id]
        return text

    def encode(self, text, add_special_tokens=False):
        return [ord(character) for character in text]

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": self.encode(text, add_special_tokens)}


class MockEnvironment:
    def __init__(self, terminate_after=2):
        self.terminate_after = terminate_after
        self.steps = 0
        self.actions = []

    async def reset(self):
        self.steps = 0
        self.actions = []
        return (
            [
                {"role": "system", "content": "You are a helpful agent."},
                {"role": "user", "content": "Please help."},
            ],
            None,
            {"id": "task-1", "domain": "test", "instructions": "Help the user"},
        )

    async def step(self, action):
        self.actions.append(action)
        self.steps += 1
        terminated = self.steps >= self.terminate_after
        reward = 1.0 if terminated else 0.0
        return f"User reply {self.steps}", reward, terminated, False, {"step": self.steps}

    def parse_response(self, response_text):
        return {"normal_text": response_text, "calls": []}

    @property
    def prompt_postprocess_fn(self):
        return None


class ToolEnvironment(MockEnvironment):
    def parse_response(self, response_text):
        return {
            "normal_text": "",
            "calls": [
                {
                    "id": "call-1",
                    "name": "lookup_order",
                    "arguments": {"order_id": "A-123"},
                }
            ],
        }

    async def step(self, action):
        self.actions.append(action)
        self.steps += 1
        return "Order found", 1.0, True, False, {"step": self.steps}


@pytest.fixture
def tokenizer():
    return MockTokenizer()


@pytest.fixture
def orchestrator(tokenizer):
    return UserSimOrchestrator(
        tokenizer=tokenizer,
        config=UserSimConfig(max_turns=3, max_context_length=10_000),
    )


def _successful_generate(tokenizer, responses=None):
    response_iter = iter(responses or ["Agent reply 1", "Agent reply 2"])

    async def generate(input_ids, sampling_params):
        text = next(response_iter)
        token_ids = tokenizer.encode(text) + [tokenizer.eos_token_id]
        return {
            "text": text,
            "token_ids": token_ids,
            "logprobs": [-0.1] * len(token_ids),
            "meta_info": {"finish_reason": {"type": "stop"}},
        }

    return generate


@pytest.mark.asyncio
async def test_rollout_completes_with_consistent_training_buffers(orchestrator, tokenizer):
    environment = MockEnvironment(terminate_after=2)

    trajectory = await orchestrator.rollout(
        environment,
        _successful_generate(tokenizer),
        {"temperature": 0.7},
    )

    assert trajectory.status == TrajectoryStatus.COMPLETED
    assert trajectory.turn_count == 2
    assert trajectory.reward == 1.0
    assert trajectory.metadata == {"task_id": "task-1", "domain": "test", "step": 2}
    assert len(trajectory.tokens) >= len(trajectory.loss_mask)
    assert len(trajectory.loss_mask) == len(trajectory.rollout_log_probs)
    assert sum(trajectory.loss_mask) > 0


@pytest.mark.asyncio
async def test_generation_exception_marks_trajectory_failed(orchestrator):
    async def generate(input_ids, sampling_params):
        raise RuntimeError("inference unavailable")

    trajectory = await orchestrator.rollout(MockEnvironment(), generate, {})

    assert trajectory.status == TrajectoryStatus.FAILED
    assert trajectory.turn_count == 0


@pytest.mark.asyncio
async def test_generation_error_sentinel_marks_trajectory_failed(orchestrator):
    async def generate(input_ids, sampling_params):
        return {"error": "server aborted"}

    trajectory = await orchestrator.rollout(MockEnvironment(), generate, {})

    assert trajectory.status == TrajectoryStatus.FAILED


@pytest.mark.asyncio
async def test_missing_eos_truncates_before_environment_step(orchestrator, tokenizer):
    async def generate(input_ids, sampling_params):
        token_ids = tokenizer.encode("unfinished")
        return {
            "text": "unfinished",
            "token_ids": token_ids,
            "logprobs": [-0.1] * len(token_ids),
            "meta_info": {},
        }

    environment = MockEnvironment()
    trajectory = await orchestrator.rollout(environment, generate, {})

    assert trajectory.status == TrajectoryStatus.TRUNCATED
    assert trajectory.turn_count == 1
    assert environment.steps == 0


@pytest.mark.asyncio
async def test_tool_call_is_serialized_for_environment(orchestrator, tokenizer):
    environment = ToolEnvironment(terminate_after=1)
    trajectory = await orchestrator.rollout(
        environment,
        _successful_generate(tokenizer, responses=["I'll look that up."]),
        {},
    )

    action = json.loads(environment.actions[0])
    assert action == {
        "name": "lookup_order",
        "arguments": {"order_id": "A-123"},
        "id": "call-1",
    }
    assert trajectory.status == TrajectoryStatus.COMPLETED


def test_token_delta_marks_template_assistant_content(orchestrator):
    assistant_tokens, assistant_mask = orchestrator._get_token_delta(
        [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ],
        tools_schema=None,
        postprocess=lambda text: text,
    )
    user_tokens, user_mask = orchestrator._get_token_delta(
        [
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "Thanks"},
        ],
        tools_schema=None,
        postprocess=lambda text: text,
    )

    assert assistant_tokens and all(assistant_mask)
    assert user_tokens and not any(user_mask)
