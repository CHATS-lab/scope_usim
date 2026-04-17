"""Task-outcome computation shared by /survey (first submit) and /session/start
(returning participant). Both paths should produce the same OutcomeCard so a
participant who re-opens their link sees the real success/partial/failure
indicator, not a placeholder.
"""
from __future__ import annotations

from typing import Any

from sqlmodel import Session, select

from ..models import StudySession, SurveyResponse, TaskType, Turn, TurnRole
from ..schemas import TaskOutcome
from .tau2_tools import evaluate_completed_session


def compute_task_outcome(
    db: Session,
    session: StudySession,
    survey_responses: dict[str, Any] | None = None,
) -> TaskOutcome:
    """Produce a task-type-specific outcome reveal for the debrief.

    `survey_responses` can be passed in for the first-submit path (where we
    already have the payload in memory). If omitted, we look up the stored
    SurveyResponse row.
    """
    if session.task_type == TaskType.P4G:
        responses = survey_responses
        if responses is None:
            row = db.exec(
                select(SurveyResponse).where(SurveyResponse.session_id == session.id)
            ).first()
            responses = row.responses if row else None
        return _p4g_outcome(responses or {})

    return _tau2_outcome(db, session)


def _p4g_outcome(responses: dict[str, Any]) -> TaskOutcome:
    amount = responses.get("intended_donation")
    try:
        amt = float(amount) if amount is not None else None
    except (TypeError, ValueError):
        amt = None
    if amt is None:
        return TaskOutcome(
            status="not_evaluated",
            label="Persuasion outcome",
            detail="No intended donation amount was recorded in the survey.",
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


def _tau2_outcome(db: Session, session: StudySession) -> TaskOutcome:
    turns = db.exec(
        select(Turn).where(Turn.session_id == session.id).order_by(Turn.turn_idx)
    ).all()
    api_messages: list[dict[str, Any]] = []
    for t in turns:
        if t.role == TurnRole.USER:
            api_messages.append({"role": "user", "content": t.content or ""})
        elif t.role == TurnRole.AGENT:
            msg: dict[str, Any] = {"role": "assistant", "content": t.content}
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

    split = session.task_split.replace("tau2_", "")
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
                "Some success criteria were met, but not all."
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
            "Automatic evaluation did not detect that the agent completed the "
            "task's success criteria. This is still useful data for the study."
        ),
    )
