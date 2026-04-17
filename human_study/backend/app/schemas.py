from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel

from .models import Condition, SessionStatus, TaskType


class SessionStartRequest(BaseModel):
    prolific_pid: str
    study_id: str
    prolific_session_id: str
    task_type: TaskType  # chosen by the researcher when building the Prolific link


class SessionStartResponse(BaseModel):
    session_id: UUID
    condition: Condition
    task_type: TaskType
    task_split: str
    task_idx: int
    task_instruction: str
    task_metadata: dict[str, Any]
    resumed: bool = False


class ChatMessage(BaseModel):
    role: str
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None


class ChatRequest(BaseModel):
    session_id: UUID
    user_message: str


class ChatResponse(BaseModel):
    messages: list[ChatMessage]  # new turns appended in this request (agent + tool turns)
    session_status: SessionStatus


class StopRequest(BaseModel):
    session_id: UUID


class StopResponse(BaseModel):
    session_status: SessionStatus
    survey_schema: dict[str, Any]


class SurveySubmitRequest(BaseModel):
    session_id: UUID
    responses: dict[str, Any]
    free_text: str | None = None


class SurveySubmitResponse(BaseModel):
    completion_code: str
