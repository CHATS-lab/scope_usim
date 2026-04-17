from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, func, select

from ..db import get_session
from ..models import (
    Condition,
    Participant,
    SessionStatus,
    StudySession,
    TaskType,
    Turn,
    TurnRole,
)
from ..schemas import (
    ChatMessage,
    DebriefInfo,
    SessionStartRequest,
    SessionStartResponse,
    TaskOutcome,
)
from ..services.conditions import assign_condition
from ..services.tasks import get_task, pick_task_for_session

router = APIRouter(prefix="/session", tags=["session"])


_CONDITION_LABELS: dict[Condition, tuple[str, str]] = {
    Condition.BASE: (
        "Base model (no RL training)",
        "You interacted with the baseline language model before any reinforcement "
        "learning. It has general instruction-following ability but hasn't been "
        "specialised for this kind of conversation.",
    ),
    Condition.RL_SINGLE: (
        "RL-Single: trained against a single simulated user",
        "You interacted with a model that was fine-tuned via reinforcement "
        "learning against one fixed LLM user simulator.",
    ),
    Condition.COTRAINING: (
        "Co-training: agent and user simulator co-evolved",
        "You interacted with a model trained using our new method (SCOPE), where "
        "both the agent and its user simulator were trained together.",
    ),
}


def _debrief_for_completed(db: Session, session: StudySession) -> DebriefInfo:
    """Build a minimal debrief for a session that's already SURVEY_DONE.
    We don't re-run tau2 evaluation here to keep the endpoint fast; we surface
    a neutral outcome card and leave detailed scoring for the original submit."""
    turn_count = db.exec(
        select(func.count(Turn.id)).where(
            Turn.session_id == session.id, Turn.role == TurnRole.USER
        )
    ).one() or 0
    label, desc = _CONDITION_LABELS[session.condition]
    return DebriefInfo(
        condition=session.condition,
        condition_label=label,
        condition_description=desc,
        task_type=session.task_type,
        task_split=session.task_split,
        task_idx=session.task_idx,
        turn_count=int(turn_count),
        completed=True,
        task_outcome=TaskOutcome(
            status="not_evaluated",
            label="Your session is already submitted",
            detail=(
                "You completed this study earlier. Your responses were saved and "
                "your completion code is shown above."
            ),
        ),
    )


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
        task = get_task(existing.task_split, existing.task_idx)
        base = dict(
            session_id=existing.id,
            condition=existing.condition,
            task_type=existing.task_type,
            task_split=existing.task_split,
            task_idx=existing.task_idx,
            task_instruction=task["instruction"],
            task_metadata=task.get("metadata", {}),
            resumed=True,
        )
        if existing.status == SessionStatus.SURVEY_DONE:
            # Re-surface the completion code + debrief instead of rejecting —
            # a participant re-opening the link shouldn't hit a dead end.
            return SessionStartResponse(
                **base,
                already_completed=True,
                completion_code=existing.completion_code,
                debrief=_debrief_for_completed(db, existing),
            )
        return SessionStartResponse(**base)

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
