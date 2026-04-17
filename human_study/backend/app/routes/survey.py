from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session

from ..db import get_session
from ..models import SessionStatus, StudySession, SurveyResponse
from ..schemas import SurveySubmitRequest, SurveySubmitResponse
from ..services.conditions import generate_completion_code

router = APIRouter(prefix="/survey", tags=["survey"])


@router.post("", response_model=SurveySubmitResponse)
def submit_survey(
    req: SurveySubmitRequest, db: Session = Depends(get_session)
) -> SurveySubmitResponse:
    session = db.get(StudySession, req.session_id)
    if session is None:
        raise HTTPException(404, "session not found")
    if session.status == SessionStatus.SURVEY_DONE:
        raise HTTPException(409, "survey already submitted")

    # Upsert response.
    existing = db.get(SurveyResponse, req.session_id)  # unique by session_id
    if existing is None:
        db.add(
            SurveyResponse(
                session_id=session.id,
                responses=req.responses,
                free_text=req.free_text,
            )
        )

    code = generate_completion_code(str(session.id))
    session.status = SessionStatus.SURVEY_DONE
    session.completion_code = code
    db.add(session)
    db.commit()

    return SurveySubmitResponse(completion_code=code)
