import json

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, func, select

from ..db import get_session
from ..models import (
    Condition,
    SessionStatus,
    StudySession,
    SurveyResponse,
    TaskType,
    Turn,
    TurnRole,
)
from ..schemas import DebriefInfo, SurveySubmitRequest, SurveySubmitResponse, TaskOutcome
from ..services.conditions import generate_completion_code
from ..services.outcome import compute_task_outcome


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

    existing = db.exec(
        select(SurveyResponse).where(SurveyResponse.session_id == session.id)
    ).first()

    # Idempotent: if the participant's client retries the POST (slow network,
    # timeout-then-success), we return the same code + debrief instead of 409.
    # The first successful write owns the canonical responses; replays are no-ops.
    if session.status == SessionStatus.SURVEY_DONE and existing is not None:
        code = session.completion_code or generate_completion_code(str(session.id))
    else:
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
    task_outcome = compute_task_outcome(db, session, survey_responses=req.responses)

    debrief = DebriefInfo(
        condition=session.condition,
        condition_label=label,
        condition_description=description,
        task_type=session.task_type,
        task_split=session.task_split,
        task_idx=session.task_idx,
        turn_count=int(turn_count),
        completed=True,
        task_outcome=task_outcome,
    )

    # Per-task_type Prolific completion code. The participant should paste
    # this (or get auto-redirected) — NOT our internal HMAC `code`.
    from ..config import settings as _s
    pcc = {
        "tau2": _s.prolific_completion_code_tau2,
        "p4g": _s.prolific_completion_code_p4g,
    }.get(session.task_type.value, "") or None
    redirect_url = (
        f"https://app.prolific.com/submissions/complete?cc={pcc}" if pcc else None
    )

    return SurveySubmitResponse(
        completion_code=code,
        prolific_completion_code=pcc,
        prolific_redirect_url=redirect_url,
        debrief=debrief,
    )


# --- legacy inline helpers retained only as a reference; real logic now lives
# --- in app/services/outcome.py so /session/start can share it.

def _legacy_evaluate_outcome_unused(
    db: Session, session: StudySession, req: SurveySubmitRequest
) -> TaskOutcome:
    """Produce a task-type-specific outcome reveal for the debrief."""
    if session.task_type == TaskType.P4G:
        amount = req.responses.get("intended_donation")
        try:
            amt = float(amount) if amount is not None else None
        except (TypeError, ValueError):
            amt = None
        if amt is None:
            return TaskOutcome(
                status="not_evaluated",
                label="Persuasion outcome",
                detail="You did not record an intended donation amount in the survey.",
            )
        if amt <= 0:
            return TaskOutcome(
                status="failure",
                label="No donation",
                detail="You indicated you would not donate. The agent did not succeed in persuading you.",
            )
        status = "success" if amt >= 1.0 else "partial"
        label = "Successful persuasion" if amt >= 1.0 else "Partial persuasion"
        return TaskOutcome(
            status=status,
            label=label,
            detail=f"You indicated you would donate ${amt:.2f}.",
        )

    # τ²-bench: run the automatic evaluator.
    turns = db.exec(
        select(Turn).where(Turn.session_id == session.id).order_by(Turn.turn_idx)
    ).all()
    api_messages: list[dict] = []
    for t in turns:
        if t.role == TurnRole.USER:
            api_messages.append({"role": "user", "content": t.content or ""})
        elif t.role == TurnRole.AGENT:
            msg: dict = {"role": "assistant", "content": t.content}
            if t.tool_calls:
                msg["tool_calls"] = t.tool_calls
            api_messages.append(msg)
        elif t.role == TurnRole.TOOL:
            api_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": t.tool_call_id,
                    "name": t.tool_name,
                    "content": t.content or "",
                }
            )

    split = session.task_split.replace("tau2_", "")  # "retail" | "airline"
    task_id = session.task_payload.get("metadata", {}).get("tau2_task_id") or str(
        session.task_idx
    )
    result = evaluate_completed_session(split=split, task_id=task_id, turns=api_messages)

    reward = result.get("reward", 0.0)
    if result.get("success"):
        return TaskOutcome(
            status="success",
            label="Task completed successfully",
            detail=(
                "The agent met all of the task's success criteria based on "
                "automatic evaluation against the task specification."
            ),
        )
    if reward > 0:
        return TaskOutcome(
            status="partial",
            label="Task partially completed",
            detail=(
                f"Automatic evaluation scored the session {reward:.2f}/1.00. "
                "Some of the task's success criteria were met, but not all."
            ),
        )
    if result.get("reason"):
        return TaskOutcome(
            status="not_evaluated",
            label="Outcome not available",
            detail=(
                "We weren't able to automatically evaluate this session. "
                "Your responses still count toward the study."
            ),
        )
    return TaskOutcome(
        status="failure",
        label="Task not completed",
        detail=(
            "Automatic evaluation did not detect that the agent completed "
            "the task's success criteria. This is still useful data for the study."
        ),
    )
