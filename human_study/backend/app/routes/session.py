from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from ..db import get_session
from ..models import Participant, SessionStatus, StudySession, TaskType, Turn, TurnRole
from ..schemas import ChatMessage, SessionStartRequest, SessionStartResponse
from ..services.conditions import assign_condition
from ..services.tasks import get_task, pick_task_for_session

router = APIRouter(prefix="/session", tags=["session"])


@router.post("/start", response_model=SessionStartResponse)
def start_session(
    req: SessionStartRequest,
    db: Session = Depends(get_session),
) -> SessionStartResponse:
    # Resume an existing session if this PID has one in-progress.
    existing = db.exec(
        select(StudySession).where(StudySession.prolific_pid == req.prolific_pid)
    ).first()
    if existing:
        if existing.status == SessionStatus.SURVEY_DONE:
            raise HTTPException(
                status_code=409,
                detail="This Prolific ID has already completed a session.",
            )
        task = get_task(existing.task_split, existing.task_idx)
        return SessionStartResponse(
            session_id=existing.id,
            condition=existing.condition,
            task_type=existing.task_type,
            task_split=existing.task_split,
            task_idx=existing.task_idx,
            task_instruction=task["instruction"],
            task_metadata=task.get("metadata", {}),
            resumed=True,
        )

    # Create participant if new.
    if not db.get(Participant, req.prolific_pid):
        db.add(Participant(prolific_pid=req.prolific_pid))

    total_sessions = db.exec(select(StudySession)).all()
    split, idx = pick_task_for_session(req.task_type.value, len(total_sessions))
    task = get_task(split, idx)

    condition = assign_condition(db, req.prolific_pid)
    session = StudySession(
        prolific_pid=req.prolific_pid,
        study_id=req.study_id,
        prolific_session_id=req.prolific_session_id,
        condition=condition,
        task_type=req.task_type,
        task_split=split,
        task_idx=idx,
        task_payload=task,
    )
    db.add(session)
    db.commit()
    db.refresh(session)

    return SessionStartResponse(
        session_id=session.id,
        condition=condition,
        task_type=req.task_type,
        task_split=split,
        task_idx=idx,
        task_instruction=task["instruction"],
        task_metadata=task.get("metadata", {}),
        resumed=False,
    )


@router.get("/{session_id}/turns", response_model=list[ChatMessage])
def get_turns(session_id: UUID, db: Session = Depends(get_session)) -> list[ChatMessage]:
    """Return the full conversation history for a session so the UI can restore
    state after a reload. Exposes the same message shape the /chat endpoints do.
    """
    session = db.get(StudySession, session_id)
    if session is None:
        raise HTTPException(404, "session not found")

    turns = db.exec(
        select(Turn).where(Turn.session_id == session.id).order_by(Turn.turn_idx)
    ).all()

    messages: list[ChatMessage] = []
    for t in turns:
        if t.role == TurnRole.USER:
            messages.append(ChatMessage(role="user", content=t.content))
        elif t.role == TurnRole.AGENT:
            messages.append(
                ChatMessage(
                    role="assistant",
                    content=t.content,
                    tool_calls=t.tool_calls,
                )
            )
        elif t.role == TurnRole.TOOL:
            messages.append(
                ChatMessage(
                    role="tool",
                    content=t.content,
                    tool_call_id=t.tool_call_id,
                    tool_name=t.tool_name,
                )
            )
    return messages
