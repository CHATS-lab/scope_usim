from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session

from ..db import get_session
from ..models import EndedReason, SessionStatus, StudySession, TaskType
from ..schemas import StopRequest, StopResponse

router = APIRouter(prefix="/stop", tags=["stop"])

SURVEYS_DIR = Path(__file__).resolve().parents[2].parent / "surveys"


def _load_survey(task_type: TaskType) -> dict[str, Any]:
    path = SURVEYS_DIR / f"{task_type.value}.yaml"
    if not path.exists():
        raise HTTPException(500, f"survey schema missing: {path}")
    with path.open() as f:
        return yaml.safe_load(f)


@router.post("", response_model=StopResponse)
def post_stop(req: StopRequest, db: Session = Depends(get_session)) -> StopResponse:
    session = db.get(StudySession, req.session_id)
    if session is None:
        raise HTTPException(404, "session not found")
    if session.status == SessionStatus.SURVEY_DONE:
        raise HTTPException(409, "session already completed")
    if session.status == SessionStatus.ACTIVE:
        from datetime import datetime, timezone

        session.status = SessionStatus.STOPPED
        session.ended_reason = EndedReason.USER_STOP
        session.ended_at = datetime.now(timezone.utc)
        db.add(session)
        db.commit()
    return StopResponse(
        session_status=session.status,
        survey_schema=_load_survey(session.task_type),
    )
