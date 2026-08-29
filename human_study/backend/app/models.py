from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import Column
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Condition(str, Enum):
    BASE = "base"
    RL_SINGLE = "rl_single"
    COTRAINING = "cotraining"


class TaskType(str, Enum):
    TAU2 = "tau2"
    P4G = "p4g"


class SessionStatus(str, Enum):
    ACTIVE = "active"
    STOPPED = "stopped"
    SURVEY_DONE = "survey_done"
    ABANDONED = "abandoned"


class StudyVariant(str, Enum):
    V1 = "v1"
    V2 = "v2"


class EndedReason(str, Enum):
    USER_STOP = "user_stop"
    AGENT_TRANSFERRED = "agent_transferred"
    TURN_LIMIT = "turn_limit"
    UNKNOWN = "unknown"


class Participant(SQLModel, table=True):
    prolific_pid: str = Field(primary_key=True)
    created_at: datetime = Field(default_factory=utcnow)


class StudySession(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    prolific_pid: str = Field(foreign_key="participant.prolific_pid", index=True)
    study_id: str
    prolific_session_id: str

    condition: Condition
    task_type: TaskType
    task_split: str  # "retail" / "airline" / "p4g"
    task_idx: int
    task_payload: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSONB))

    status: SessionStatus = SessionStatus.ACTIVE
    completion_code: str | None = None

    # Which participant-UI variant this session was served under. v2 (scannable
    # structured panel) is now the default; v1 (prose) is opt-in via ?variant=v1.
    variant: StudyVariant = Field(default=StudyVariant.V2)

    # Why the conversation ended. `user_stop` = participant hit /stop,
    # `agent_transferred` = the agent invoked transfer_to_human_agents tool,
    # `turn_limit` = hit MAX_TURNS, `unknown` = anything else.
    ended_reason: EndedReason = Field(default=EndedReason.UNKNOWN)

    created_at: datetime = Field(default_factory=utcnow)
    ended_at: datetime | None = None


class TurnRole(str, Enum):
    SYSTEM = "system"
    AGENT = "assistant"
    USER = "user"
    TOOL = "tool"


class Turn(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    session_id: UUID = Field(foreign_key="studysession.id", index=True)
    turn_idx: int
    role: TurnRole
    content: str | None = None
    # Raw OpenAI-format tool calls for assistant turns, or tool result payload
    # for tool turns. Kept as JSONB so we can replay the exact API exchange.
    tool_calls: list[dict[str, Any]] | None = Field(
        default=None, sa_column=Column(JSONB, nullable=True)
    )
    tool_call_id: str | None = None
    tool_name: str | None = None
    created_at: datetime = Field(default_factory=utcnow)


class SurveyResponse(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    session_id: UUID = Field(foreign_key="studysession.id", unique=True, index=True)
    responses: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSONB))
    free_text: str | None = None
    created_at: datetime = Field(default_factory=utcnow)


class Annotation(SQLModel, table=True):
    """Third-party annotation of a completed conversation.

    Multiple annotators can review the same session. Each (session, annotator)
    pair can have at most one annotation (enforced by a unique constraint).
    """

    id: int | None = Field(default=None, primary_key=True)
    session_id: UUID = Field(foreign_key="studysession.id", index=True)
    annotator_id: str = Field(index=True)
    responses: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSONB))
    free_text: str | None = None
    created_at: datetime = Field(default_factory=utcnow)

    __table_args__ = (
        # SQLModel picks up UniqueConstraint from sa_column args too, but a
        # tuple is the portable form.
    )
