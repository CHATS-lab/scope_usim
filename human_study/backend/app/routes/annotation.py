"""Annotation-mode endpoints.

A second cohort of participants ("annotators") reviews already-completed
conversations and rates them on quality, naturalness, and agent weirdness.
Each (session, annotator) pair produces at most one annotation; each session
can be annotated by multiple annotators for inter-annotator agreement.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import UUID

import yaml
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func as sa_func, select as sa_select
from sqlmodel import Session, func, select

from ..db import get_session
from ..models import (
    Annotation,
    SessionStatus,
    StudySession,
    Turn,
    TurnRole,
)
from ..schemas import (
    AnnotationNextResponse,
    AnnotationSubmitRequest,
    AnnotationSubmitResponse,
    ChatMessage,
)
from ..services.conditions import generate_completion_code

router = APIRouter(prefix="/annotation", tags=["annotation"])

SURVEYS_DIR = Path(__file__).resolve().parents[2].parent / "surveys"

# Cap on annotations per session so we don't waste annotator effort on
# already-well-covered conversations. Matches the "2 annotators per task"
# target from Appendix D.
MAX_ANNOTATIONS_PER_SESSION = 2


def _load_annotation_schema() -> dict[str, Any]:
    with (SURVEYS_DIR / "annotation.yaml").open() as f:
        return yaml.safe_load(f)


def _turns_as_messages(db: Session, session: StudySession) -> list[ChatMessage]:
    turns = db.exec(
        select(Turn).where(Turn.session_id == session.id).order_by(Turn.turn_idx)
    ).all()
    out: list[ChatMessage] = []
    for t in turns:
        if t.role == TurnRole.USER:
            out.append(ChatMessage(role="user", content=t.content))
        elif t.role == TurnRole.AGENT:
            out.append(
                ChatMessage(
                    role="assistant", content=t.content, tool_calls=t.tool_calls
                )
            )
        elif t.role == TurnRole.TOOL:
            out.append(
                ChatMessage(
                    role="tool",
                    content=t.content,
                    tool_call_id=t.tool_call_id,
                    tool_name=t.tool_name,
                )
            )
    return out


def _user_turn_count(db: Session, session: StudySession) -> int:
    return int(
        db.exec(
            select(func.count(Turn.id)).where(
                Turn.session_id == session.id, Turn.role == TurnRole.USER
            )
        ).one()
        or 0
    )


@router.get("/next", response_model=AnnotationNextResponse)
def get_next_session(
    annotator_id: str,
    session_id: UUID | None = None,
    db: Session = Depends(get_session),
) -> AnnotationNextResponse:
    """Return the next SURVEY_DONE session this annotator hasn't rated yet.

    If `session_id` is supplied, return that specific session instead of
    pulling from the queue — useful for researcher spot-checks or letting
    someone review their own completed session.
    """
    if not annotator_id:
        raise HTTPException(400, "annotator_id is required")

    if session_id is not None:
        target = db.get(StudySession, session_id)
        if target is None:
            raise HTTPException(404, "session not found")
        if target.status != SessionStatus.SURVEY_DONE:
            raise HTTPException(
                409, "session has not been completed by a participant yet"
            )
        return AnnotationNextResponse(
            session_id=target.id,
            task_type=target.task_type,
            task_split=target.task_split,
            task_idx=target.task_idx,
            task_instruction=(target.task_payload or {}).get("instruction", ""),
            turn_count=_user_turn_count(db, target),
            messages=_turns_as_messages(db, target),
            survey_schema=_load_annotation_schema(),
            annotations_done=0,
            annotations_available=1,
            done=False,
        )

    already_annotated = db.exec(
        select(Annotation.session_id).where(Annotation.annotator_id == annotator_id)
    ).all()
    already_set = {sid for sid in already_annotated}

    # Count annotations per session.
    ann_counts_rows = db.exec(
        sa_select(Annotation.session_id, sa_func.count(Annotation.id)).group_by(
            Annotation.session_id
        )
    ).all()
    ann_counts: dict[UUID, int] = {sid: n for sid, n in ann_counts_rows}

    # All finished sessions, ordered by existing annotation count (ascending)
    # then created_at so each annotator fills gaps first.
    finished = db.exec(
        select(StudySession)
        .where(StudySession.status == SessionStatus.SURVEY_DONE)
        .order_by(StudySession.created_at)
    ).all()

    finished_sorted = sorted(
        finished, key=lambda s: (ann_counts.get(s.id, 0), s.created_at)
    )

    target: StudySession | None = None
    for s in finished_sorted:
        if s.id in already_set:
            continue
        if ann_counts.get(s.id, 0) >= MAX_ANNOTATIONS_PER_SESSION:
            continue
        target = s
        break

    total_available_for_you = sum(
        1
        for s in finished_sorted
        if s.id not in already_set
        and ann_counts.get(s.id, 0) < MAX_ANNOTATIONS_PER_SESSION
    )
    done_by_you = len(already_set)

    if target is None:
        return AnnotationNextResponse(
            session_id=UUID(int=0),
            task_type="tau2",  # type: ignore[arg-type]
            task_split="",
            task_idx=0,
            task_instruction="",
            turn_count=0,
            messages=[],
            survey_schema={},
            annotations_done=done_by_you,
            annotations_available=0,
            done=True,
        )

    return AnnotationNextResponse(
        session_id=target.id,
        task_type=target.task_type,
        task_split=target.task_split,
        task_idx=target.task_idx,
        task_instruction=(target.task_payload or {}).get("instruction", ""),
        turn_count=_user_turn_count(db, target),
        messages=_turns_as_messages(db, target),
        survey_schema=_load_annotation_schema(),
        annotations_done=done_by_you,
        annotations_available=total_available_for_you,
    )


@router.post("", response_model=AnnotationSubmitResponse)
def submit_annotation(
    req: AnnotationSubmitRequest, db: Session = Depends(get_session)
) -> AnnotationSubmitResponse:
    if not req.annotator_id:
        raise HTTPException(400, "annotator_id is required")
    session = db.get(StudySession, req.session_id)
    if session is None:
        raise HTTPException(404, "session not found")
    if session.status != SessionStatus.SURVEY_DONE:
        raise HTTPException(409, "session is not yet completed by a participant")

    # Idempotent: if this annotator already submitted for this session, don't
    # double-count.
    existing = db.exec(
        select(Annotation).where(
            Annotation.session_id == session.id,
            Annotation.annotator_id == req.annotator_id,
        )
    ).first()
    if existing is None:
        db.add(
            Annotation(
                session_id=session.id,
                annotator_id=req.annotator_id,
                responses=req.responses,
                free_text=req.free_text,
            )
        )
        db.commit()

    # Does a next session exist?
    nxt = get_next_session(annotator_id=req.annotator_id, db=db)
    next_available = not nxt.done

    return AnnotationSubmitResponse(
        completion_code=(
            generate_completion_code(f"ann:{req.annotator_id}")
            if not next_available
            else None
        ),
        next_available=next_available,
    )
