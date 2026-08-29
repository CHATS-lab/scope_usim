"""Condition assignment + model routing."""
from __future__ import annotations

import hashlib
import hmac

from sqlmodel import Session, func, select

from ..config import settings
from ..models import Condition, StudySession


def assign_condition(db: Session, prolific_pid: str) -> Condition:
    """Assign least-populated condition for load balancing.

    Ties broken deterministically by a hash of the PID so re-invocations are
    stable if a participant somehow triggers assignment twice.
    """
    counts = dict(
        db.exec(
            select(StudySession.condition, func.count(StudySession.id)).group_by(
                StudySession.condition
            )
        ).all()
    )
    # Ensure every condition is represented so min() works.
    for c in Condition:
        counts.setdefault(c, 0)

    min_count = min(counts.values())
    candidates = sorted([c for c, n in counts.items() if n == min_count], key=lambda c: c.value)

    h = int(hashlib.sha256(prolific_pid.encode()).hexdigest(), 16)
    return candidates[h % len(candidates)]


def model_for_condition(condition: Condition) -> tuple[str, str | None, str | None]:
    """Return (model_name, base_url_or_None, api_key_or_None) for the client.

    api_key_or_None is a per-condition override; when None, callers should
    fall back to settings.openai_api_key.
    """
    match condition:
        case Condition.BASE:
            return (
                settings.model_base,
                settings.openai_base_url_base or None,
                settings.openai_api_key_base or None,
            )
        case Condition.RL_SINGLE:
            return (
                settings.model_rl_single,
                settings.openai_base_url_rl_single or None,
                settings.openai_api_key_rl_single or None,
            )
        case Condition.COTRAINING:
            return (
                settings.model_cotraining,
                settings.openai_base_url_cotraining or None,
                settings.openai_api_key_cotraining or None,
            )


def generate_completion_code(session_id: str) -> str:
    """HMAC-signed completion code so we can verify on Prolific webhook / manual check."""
    mac = hmac.new(
        settings.completion_code_secret.encode(), session_id.encode(), hashlib.sha256
    ).hexdigest()[:12]
    return f"USIM-{mac.upper()}"
