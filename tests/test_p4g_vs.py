"""Tests for P4G Verbalized Sampling integration.

Covers:
- `build_p4g_vs_instruction` template (donation markers preserved, method
  validation)
- `build_persuadee_system_prompt` VS toggling (VS block appended only when
  requested, composes with `prompt_prefix`)
- End-to-end pipeline: VS JSON parsing -> candidate sampling -> P4G reward
  extraction still fires on sampled text with [DONATE $N] / [GIVE $N]
  markers.
"""

import random

import pytest

from usim.core.types import Message
from usim.core.vs_parsing import (
    extract_json_object,
    sample_candidate,
    validate_candidates,
)
from usim.p4g.prompts import (
    build_p4g_vs_instruction,
    build_persuadee_system_prompt,
)
from usim.p4g.reward import compute_p4g_reward


class TestVsInstruction:
    """Tests for the VS instruction template."""

    def test_prob_template_contains_donation_markers(self):
        inst = build_p4g_vs_instruction(word_limit=50, num_samples=5, method="prob")
        assert "[DONATE $N]" in inst
        assert "[GIVE $N]" in inst
        assert "responses" in inst
        assert "probability" in inst
        assert "JSON" in inst.upper() or "json" in inst

    def test_prob_template_uses_num_samples(self):
        inst = build_p4g_vs_instruction(word_limit=50, num_samples=3, method="prob")
        assert "generate 3 plausible replies" in inst

    def test_prob_template_uses_word_limit(self):
        inst = build_p4g_vs_instruction(word_limit=75, num_samples=5, method="prob")
        assert "under 75 words" in inst

    def test_random_template_contains_donation_markers(self):
        inst = build_p4g_vs_instruction(word_limit=50, num_samples=5, method="random")
        assert "[DONATE $N]" in inst
        assert "[GIVE $N]" in inst
        assert "responses" in inst

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="prob.*random"):
            build_p4g_vs_instruction(word_limit=50, num_samples=5, method="bogus")


class TestPersuadeeSystemPrompt:
    """Tests for VS toggling in the P4G persuadee prompt builder."""

    def test_base_prompt_no_vs_block(self):
        prompt = build_persuadee_system_prompt(
            "<persona>Alice</persona>", word_limit=50
        )
        assert "Response Diversity Instructions" not in prompt
        # Base prompt itself already mentions the donation format
        assert "[DONATE $N]" in prompt

    def test_vs_prompt_has_vs_block(self):
        prompt = build_persuadee_system_prompt(
            "<persona>Alice</persona>",
            word_limit=50,
            verbalized_sampling=True,
            vs_num_samples=5,
            vs_method="prob",
        )
        assert "Response Diversity Instructions" in prompt
        assert "responses" in prompt
        # The VS block reminds the model to preserve donation markers, so
        # the markers now appear in both the base and VS sections.
        assert prompt.count("[DONATE $N]") >= 2

    def test_vs_prompt_starts_with_base_prompt(self):
        prompt = build_persuadee_system_prompt(
            "<persona>Alice</persona>",
            word_limit=50,
            verbalized_sampling=True,
        )
        assert prompt.startswith("You are an Amazon Mechanical Turk worker")

    def test_vs_composes_with_prefix(self):
        prompt = build_persuadee_system_prompt(
            "<persona>Alice</persona>",
            word_limit=50,
            prompt_prefix="Be extra skeptical.",
            verbalized_sampling=True,
        )
        assert prompt.startswith("Be extra skeptical.")
        assert "Response Diversity Instructions" in prompt

    def test_vs_num_samples_propagates(self):
        prompt = build_persuadee_system_prompt(
            "<persona>Alice</persona>",
            word_limit=50,
            verbalized_sampling=True,
            vs_num_samples=3,
        )
        assert "generate 3 plausible replies" in prompt


class TestVsRewardPipeline:
    """End-to-end: VS JSON -> sampler -> P4G reward extraction.

    Critical property: the [DONATE $N] / [GIVE $N] markers must survive
    through the VS sampling path, otherwise the reward function silently
    returns 0 for every sampled response and training goes nowhere.
    """

    VS_RESPONSE_DONATE = """
    {"responses": [
      {"text": "That's a worthy cause. I'll help out. [DONATE $0.50]", "probability": 0.3},
      {"text": "I'm not sure, I earned this money and want to keep it for myself.", "probability": 0.5},
      {"text": "Can you tell me more about the charity first?", "probability": 0.2}
    ]}
    """

    def test_donate_marker_preserved_in_sampled_text(self):
        parsed = extract_json_object(self.VS_RESPONSE_DONATE)
        candidates = validate_candidates(parsed)
        assert candidates is not None and len(candidates) == 3
        # The first candidate contains the [DONATE $0.50] marker
        donate_text = candidates[0]["text"]
        assert "[DONATE $0.50]" in donate_text

    def test_sampling_distribution_matches_verbalized_probs(self):
        parsed = extract_json_object(self.VS_RESPONSE_DONATE)
        candidates = validate_candidates(parsed)
        assert candidates is not None

        random.seed(42)
        trials = 2000
        donate_count = 0
        for _ in range(trials):
            text = sample_candidate(candidates, method="prob")
            if "[DONATE" in text:
                donate_count += 1

        observed_rate = donate_count / trials
        # p=0.3 with 2000 samples -> ~600. Allow a generous band.
        assert 0.25 < observed_rate < 0.35, (
            f"expected ~30% donate rate, got {observed_rate:.1%}"
        )

    def test_reward_extraction_fires_on_sampled_donate_text(self):
        """Simulate a rollout where VS sampled the donate candidate."""
        parsed = extract_json_object(self.VS_RESPONSE_DONATE)
        candidates = validate_candidates(parsed)

        # Force-pick the donate candidate
        donate_text = candidates[0]["text"]
        msg = Message(role="user", content=donate_text)
        reward = compute_p4g_reward([msg])
        # $0.50 / $2.00 = 0.25
        assert abs(reward - 0.25) < 1e-6, f"expected 0.25, got {reward}"

    def test_reward_is_zero_for_non_donate_candidate(self):
        parsed = extract_json_object(self.VS_RESPONSE_DONATE)
        candidates = validate_candidates(parsed)

        # Pick one of the non-donate candidates
        non_donate_text = candidates[1]["text"]
        msg = Message(role="user", content=non_donate_text)
        assert compute_p4g_reward([msg]) == 0.0

    def test_give_marker_also_extracted(self):
        msg = Message(role="user", content="OK, [GIVE $1.25]")
        reward = compute_p4g_reward([msg])
        # $1.25 / $2.00 = 0.625
        assert abs(reward - 0.625) < 1e-6
