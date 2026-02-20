"""Reward computation for Persuasion for Good.

Extracts donation amount from persuadee messages and normalizes to [0, 1].

Regex patterns ported from persuasion_simulation/src/evaluation/metrics/donation_metrics.py.
"""

import re
from typing import List

from usim.core.types import Message


def extract_donation_amount(messages: List[Message]) -> float:
    """Extract the final donation amount from conversation messages.

    Scans persuadee (role="user") messages for donation amounts using:
    1. Structured format: [DONATE $N] or [GIVE $N]
    2. Fallback regex: natural language donation mentions

    Takes the last matched amount (final decision).

    Args:
        messages: Conversation messages

    Returns:
        Donation amount in dollars (0.0 if none found)
    """
    last_amount = 0.0

    # Pattern 1: Structured format [DONATE $N] / [GIVE $N]
    structured_pattern = re.compile(r"\[(DONATE|GIVE)\s*\$?(\d+\.?\d*)\]", re.IGNORECASE)

    # Pattern 2: Fallback — natural language donation mentions
    fallback_pattern = re.compile(
        r"\b(?:donate|donating|give|giving|contribute|contributing)"
        r"\s*(?:the whole|all|entire)?\s*\$?(\d*\.\d{1,2}|\d+)\b",
        re.IGNORECASE,
    )

    for msg in messages:
        if msg.role != "user" or not msg.content:
            continue

        # Try structured format first
        structured_matches = structured_pattern.findall(msg.content)
        if structured_matches:
            # Each match is (DONATE|GIVE, amount)
            last_amount = float(structured_matches[-1][1])
            continue

        # Fallback to natural language
        fallback_matches = fallback_pattern.findall(msg.content)
        if fallback_matches:
            last_amount = float(fallback_matches[-1])

    return last_amount


def compute_p4g_reward(messages: List[Message]) -> float:
    """Compute reward from P4G conversation.

    Reward = donation_amount / 2.0, clamped to [0, 1].

    Args:
        messages: Conversation messages

    Returns:
        Reward in [0.0, 1.0]
    """
    donation = extract_donation_amount(messages)
    return min(donation / 2.0, 1.0)
