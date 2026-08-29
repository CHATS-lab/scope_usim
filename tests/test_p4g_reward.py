"""Tests for P4G reward extraction."""

from usim.core.types import Message
from usim.p4g.reward import compute_p4g_reward, extract_donation_amount


def _msgs(*user_texts):
    """Helper to create alternating assistant/user messages."""
    messages = []
    for i, text in enumerate(user_texts):
        messages.append(Message(role="assistant", content=f"Turn {i}"))
        messages.append(Message(role="user", content=text))
    return messages


class TestExtractDonationAmount:
    def test_structured_donate(self):
        msgs = _msgs("Sure, I'll [DONATE $1.00] to the charity.")
        assert extract_donation_amount(msgs) == 1.0

    def test_structured_give(self):
        msgs = _msgs("I'll [GIVE $0.50] to Save the Children.")
        assert extract_donation_amount(msgs) == 0.50

    def test_structured_no_dollar_sign(self):
        msgs = _msgs("Ok [DONATE 2] sounds good.")
        assert extract_donation_amount(msgs) == 2.0

    def test_structured_decimal(self):
        msgs = _msgs("[GIVE $1.25]")
        assert extract_donation_amount(msgs) == 1.25

    def test_fallback_donate(self):
        msgs = _msgs("I would like to donate $1.50 to the charity.")
        assert extract_donation_amount(msgs) == 1.50

    def test_fallback_give(self):
        msgs = _msgs("I can give 0.75 to the cause.")
        assert extract_donation_amount(msgs) == 0.75

    def test_fallback_contribute(self):
        msgs = _msgs("I'll contribute $1 to the charity.")
        assert extract_donation_amount(msgs) == 1.0

    def test_last_amount_wins(self):
        msgs = _msgs(
            "I'll [DONATE $0.50]",
            "Actually, let me [DONATE $1.00] instead.",
        )
        assert extract_donation_amount(msgs) == 1.0

    def test_no_donation(self):
        msgs = _msgs("No thanks, I'll keep my money.", "I'm not interested.")
        assert extract_donation_amount(msgs) == 0.0

    def test_agent_messages_ignored(self):
        messages = [
            Message(role="assistant", content="Would you [DONATE $2.00]?"),
            Message(role="user", content="Let me think about it."),
        ]
        assert extract_donation_amount(messages) == 0.0

    def test_empty_messages(self):
        assert extract_donation_amount([]) == 0.0

    def test_case_insensitive(self):
        msgs = _msgs("[donate $1.00]")
        assert extract_donation_amount(msgs) == 1.0


class TestComputeP4GReward:
    def test_full_donation(self):
        msgs = _msgs("[DONATE $2.00]")
        assert compute_p4g_reward(msgs) == 1.0

    def test_half_donation(self):
        msgs = _msgs("[DONATE $1.00]")
        assert compute_p4g_reward(msgs) == 0.5

    def test_no_donation(self):
        msgs = _msgs("No thanks.")
        assert compute_p4g_reward(msgs) == 0.0

    def test_over_donation_clamped(self):
        msgs = _msgs("[DONATE $5.00]")
        assert compute_p4g_reward(msgs) == 1.0

    def test_small_donation(self):
        msgs = _msgs("[GIVE $0.25]")
        assert compute_p4g_reward(msgs) == 0.125
