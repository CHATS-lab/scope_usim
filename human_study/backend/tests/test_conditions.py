"""Unit tests for condition assignment and completion codes.

Run from backend/ with: pytest tests/ -q
Requires: pip install -e .[dev]
"""
from unittest.mock import MagicMock

from app.models import Condition
from app.services.conditions import assign_condition, generate_completion_code


def _db_with_counts(counts: dict[Condition, int]) -> MagicMock:
    rows = [(c, n) for c, n in counts.items()]
    db = MagicMock()
    db.exec.return_value.all.return_value = rows
    return db


def test_assign_picks_least_populated():
    db = _db_with_counts({Condition.BASE: 5, Condition.RL_SINGLE: 2, Condition.COTRAINING: 10})
    assert assign_condition(db, "pid-123") == Condition.RL_SINGLE


def test_assign_deterministic_on_tie():
    db = _db_with_counts({Condition.BASE: 3, Condition.RL_SINGLE: 3, Condition.COTRAINING: 3})
    # Same pid → same condition every call (stable under ties).
    a = assign_condition(db, "pid-xyz")
    b = assign_condition(db, "pid-xyz")
    assert a == b


def test_completion_code_is_deterministic_and_prefixed():
    code1 = generate_completion_code("session-A")
    code2 = generate_completion_code("session-A")
    assert code1 == code2
    assert code1.startswith("USIM-")
    assert generate_completion_code("session-B") != code1
