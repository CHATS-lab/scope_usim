from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, func, select

from ..db import get_session
from ..models import Condition, SessionStatus, StudySession, SurveyResponse, Turn, TurnRole
from ..schemas import DebriefInfo, SurveySubmitRequest, SurveySubmitResponse
from ..services.conditions import generate_completion_code


# Human-readable descriptions revealed post-survey. Phrased in plain language
# so non-technical participants understand what they were interacting with.
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
        "learning against one fixed LLM user simulator. This is the standard "
        "multi-turn RL approach.",
    ),
    Condition.COTRAINING: (
        "Co-training: agent and user simulator co-evolved",
        "You interacted with a model trained using our new method (SCOPE), "
        "where both the agent and its user simulator were trained together so "
        "the agent sees a diverse range of user behaviours during training.",
    ),
}

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

    # Upsert response keyed by session_id (unique).
    existing = db.exec(
        select(SurveyResponse).where(SurveyResponse.session_id == session.id)
    ).first()
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

    turn_count = db.exec(
        select(func.count(Turn.id)).where(
            Turn.session_id == session.id, Turn.role == TurnRole.USER
        )
    ).one() or 0

    label, description = _CONDITION_LABELS[session.condition]
    debrief = DebriefInfo(
        condition=session.condition,
        condition_label=label,
        condition_description=description,
        task_type=session.task_type,
        task_split=session.task_split,
        task_idx=session.task_idx,
        turn_count=int(turn_count),
        completed=True,
    )

    return SurveySubmitResponse(completion_code=code, debrief=debrief)
