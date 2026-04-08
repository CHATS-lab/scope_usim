"""Tests for the tau2 Verbalized Sampling user simulator.

Parser / sampler tests in the ``Test*`` classes below use only the pure
helpers in ``usim.core.environment.tau2.vs_parsing`` and do not require
``tau2`` to be installed. The final ``TestUserSimulatorIntegration`` class
imports the full ``VerbalizedSamplingUserSimulator`` and is skipped when
tau2 is missing.
"""

import random
import pytest

from usim.core.environment.tau2.vs_parsing import (
    extract_json_object,
    sample_candidate,
    strip_code_fence,
    validate_candidates,
)


class TestJsonExtraction:
    """Tests for the defensive JSON parser."""

    def test_clean_json(self):
        obj = extract_json_object(
            '{"responses":[{"text":"Yes","probability":0.7}]}'
        )
        assert obj == {"responses": [{"text": "Yes", "probability": 0.7}]}

    def test_markdown_fence_json(self):
        obj = extract_json_object(
            '```json\n{"responses":[{"text":"Yes","probability":0.5}]}\n```'
        )
        assert obj is not None
        assert obj["responses"][0]["text"] == "Yes"

    def test_markdown_fence_no_lang(self):
        obj = extract_json_object(
            '```\n{"responses":[{"text":"Hi","probability":1.0}]}\n```'
        )
        assert obj is not None

    def test_surrounding_text(self):
        obj = extract_json_object(
            'Sure, here you go: {"responses":[{"text":"Hi","probability":1.0}]} done.'
        )
        assert obj is not None
        assert obj["responses"][0]["text"] == "Hi"

    def test_malformed_returns_none(self):
        assert extract_json_object("not json at all") is None

    def test_empty_string_returns_none(self):
        assert extract_json_object("") is None


class TestStripCodeFence:
    def test_json_fence(self):
        assert strip_code_fence('```json\n{"a":1}\n```') == '{"a":1}'

    def test_plain_fence(self):
        assert strip_code_fence('```\n{"a":1}\n```') == '{"a":1}'

    def test_no_fence(self):
        assert strip_code_fence('{"a":1}') == '{"a":1}'


class TestValidateCandidates:
    """Tests for the shape validator."""

    def test_valid_candidates(self):
        result = validate_candidates(
            {"responses": [
                {"text": "A", "probability": 0.6},
                {"text": "B", "probability": 0.4},
            ]}
        )
        assert result is not None
        assert len(result) == 2
        assert result[0]["text"] == "A"

    def test_missing_probability_defaults_uniform(self):
        result = validate_candidates(
            {"responses": [{"text": "A"}, {"text": "B"}]}
        )
        assert result is not None
        assert len(result) == 2
        assert result[0]["probability"] == 0.5

    def test_empty_text_filtered(self):
        result = validate_candidates(
            {"responses": [
                {"text": "", "probability": 0.5},
                {"text": "Hi", "probability": 0.5},
            ]}
        )
        assert result is not None
        assert len(result) == 1
        assert result[0]["text"] == "Hi"

    def test_empty_list_returns_none(self):
        assert validate_candidates({"responses": []}) is None

    def test_not_a_dict_returns_none(self):
        assert validate_candidates([]) is None

    def test_missing_responses_key(self):
        assert validate_candidates({"other": []}) is None


class TestSampling:
    """Tests for the pure sampler."""

    def test_prob_sampling_respects_weights(self):
        candidates = [
            {"text": "A", "probability": 0.9},
            {"text": "B", "probability": 0.1},
        ]
        random.seed(42)
        results = [sample_candidate(candidates, method="prob") for _ in range(2000)]
        count_a = results.count("A")
        # p=0.9 should give ~1800 out of 2000, allow wide tolerance
        assert 1700 < count_a < 1900, f"expected ~1800 A, got {count_a}"

    def test_random_sampling_is_uniform(self):
        candidates = [
            {"text": "A", "probability": 0.9},
            {"text": "B", "probability": 0.1},
        ]
        random.seed(42)
        results = [sample_candidate(candidates, method="random") for _ in range(2000)]
        count_a = results.count("A")
        # method='random' should ignore probabilities, ~1000/2000 each
        assert 900 < count_a < 1100, f"expected ~1000 A, got {count_a}"

    def test_zero_weights_falls_back_to_uniform(self):
        candidates = [
            {"text": "A", "probability": 0},
            {"text": "B", "probability": 0},
        ]
        # Should not raise and should still return a valid text
        sampled = sample_candidate(candidates, method="prob")
        assert sampled in ("A", "B")

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="empty candidate list"):
            sample_candidate([], method="prob")

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="prob.*random"):
            sample_candidate([{"text": "A"}], method="bogus")


# ---------------------------------------------------------------------------
# Integration tests below — only run if tau2 is installed
# ---------------------------------------------------------------------------

try:
    from usim.core.environment.tau2.vs_user_simulator import (
        VerbalizedSamplingUserSimulator,
    )
    _HAS_TAU2 = True
except ImportError:
    VerbalizedSamplingUserSimulator = None  # type: ignore
    _HAS_TAU2 = False


@pytest.mark.skipif(not _HAS_TAU2, reason="tau2-bench not installed")
class TestSystemPrompt:
    """Verify the VS instruction is appended to the base system prompt."""

    def test_prob_instruction_appended(self):
        sim = VerbalizedSamplingUserSimulator(
            tools=None,
            instructions="You want to return a defective item.",
            llm="fake-model",
            llm_args={},
            vs_num_samples=5,
            vs_method="prob",
        )
        prompt = sim.system_prompt
        assert "probability" in prompt.lower()
        assert "responses" in prompt
        assert "5" in prompt  # num_samples made it into the template

    def test_random_instruction_appended(self):
        sim = VerbalizedSamplingUserSimulator(
            tools=None,
            instructions="You want to return a defective item.",
            llm="fake-model",
            llm_args={},
            vs_num_samples=3,
            vs_method="random",
        )
        prompt = sim.system_prompt
        assert "responses" in prompt
        # The random-method instruction template does not mention the word
        # "empirical probability" (only the prob template does).
        assert "empirical probability" not in prompt.lower()

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="prob.*random"):
            VerbalizedSamplingUserSimulator(
                tools=None,
                instructions=None,
                llm="fake-model",
                llm_args={},
                vs_method="bogus",
            )
