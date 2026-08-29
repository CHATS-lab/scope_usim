"""Tests for P4G prompt construction."""

from usim.p4g.prompts import build_persuadee_system_prompt, build_persuader_system_prompt


SAMPLE_PERSONA = """<persona>
- Demographics: Age: 25, Sex: Male, Race: White, Education: Bachelor's, Marital Status: Single, Employment: Full-time, Income: $40k, Religion: None, Ideology: Moderate
- Big-Five Personality Traits [1-5]: Extroversion: 3, Agreeableness: 4, Conscientiousness: 3, Neuroticism: 2, Openness: 4
- Moral Foundations [1-6]: Care: 5, Fairness: 4, Loyalty: 3, Authority: 2, Purity: 2, Freedom: 5
- Schwartz Values [1-6]: Conformity: 2, Tradition: 2, Benevolence: 5, Universalism: 4, Self Direction: 5, Stimulation: 3, Hedonism: 3, Achievement: 4, Power: 2, Security: 3
- Decision Making Style [1-5]: Rational: 4, Intuitive: 3
</persona>"""


class TestPersuaderPrompt:
    def test_contains_persona(self):
        prompt = build_persuader_system_prompt(SAMPLE_PERSONA)
        assert "<persona>" in prompt
        assert "Age: 25" in prompt

    def test_contains_scenario(self):
        prompt = build_persuader_system_prompt(SAMPLE_PERSONA)
        assert "<scenario>" in prompt
        assert "Save the Children" in prompt

    def test_contains_rules(self):
        prompt = build_persuader_system_prompt(SAMPLE_PERSONA)
        assert "<rules>" in prompt

    def test_word_limit_injected(self):
        prompt = build_persuader_system_prompt(SAMPLE_PERSONA, word_limit=100)
        assert "100 words" in prompt

    def test_num_turns_injected(self):
        prompt = build_persuader_system_prompt(SAMPLE_PERSONA, num_turns=20)
        assert "10 chat turns" in prompt  # 20 // 2 = 10

    def test_amt_framing(self):
        prompt = build_persuader_system_prompt(SAMPLE_PERSONA)
        assert "Amazon Mechanical Turk" in prompt

    def test_no_reasoning_instruction(self):
        prompt = build_persuader_system_prompt(SAMPLE_PERSONA)
        assert "Only output your reply" in prompt


class TestPersuadeePrompt:
    def test_contains_persona(self):
        prompt = build_persuadee_system_prompt(SAMPLE_PERSONA)
        assert "<persona>" in prompt
        assert "Age: 25" in prompt

    def test_contains_task(self):
        prompt = build_persuadee_system_prompt(SAMPLE_PERSONA)
        assert "<task>" in prompt
        assert "$2.00" in prompt

    def test_donation_format_instructions(self):
        prompt = build_persuadee_system_prompt(SAMPLE_PERSONA)
        assert "[DONATE $N]" in prompt
        assert "[GIVE $N]" in prompt

    def test_word_limit_injected(self):
        prompt = build_persuadee_system_prompt(SAMPLE_PERSONA, word_limit=80)
        assert "80 words" in prompt

    def test_no_reasoning_instruction(self):
        prompt = build_persuadee_system_prompt(SAMPLE_PERSONA)
        assert "Only output your reply" in prompt
