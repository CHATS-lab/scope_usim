"""Chat endpoint: user → policy → (optional tool loop) → response."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from ..config import settings
from ..db import get_session
from ..models import SessionStatus, StudySession, TaskType, Turn, TurnRole
from ..schemas import ChatMessage, ChatRequest, ChatResponse
from ..services.conditions import model_for_condition
from ..services.llm import chat_once
from ..services.tau2_tools import TAU2_AVAILABLE, Tau2Runtime, parse_tool_arguments

router = APIRouter(prefix="/chat", tags=["chat"])

# Cache of live Tau2Runtime per session. In a multi-worker deployment we'd
# pin sessions to workers or move this to Redis; fine for a single-worker pilot.
_TAU2_RUNTIMES: dict[str, Tau2Runtime] = {}


def _system_prompt(session: StudySession) -> str:
    if session.task_type == TaskType.TAU2:
        return (
            "You are a helpful customer-service agent. The user is a real customer. "
            "Use the provided tools to look up information and complete their request. "
            "Be concise and verify details before taking irreversible actions."
        )
    return (
        "You are a persuader trying to encourage the user to donate to a children's "
        "charity. Engage them in a natural, respectful conversation."
    )


def _build_api_messages(session: StudySession, turns: list[Turn]) -> list[dict[str, Any]]:
    msgs: list[dict[str, Any]] = [{"role": "system", "content": _system_prompt(session)}]
    for t in turns:
        if t.role == TurnRole.USER:
            msgs.append({"role": "user", "content": t.content or ""})
        elif t.role == TurnRole.AGENT:
            msg: dict[str, Any] = {"role": "assistant", "content": t.content}
            if t.tool_calls:
                msg["tool_calls"] = t.tool_calls
            msgs.append(msg)
        elif t.role == TurnRole.TOOL:
            msgs.append(
                {
                    "role": "tool",
                    "tool_call_id": t.tool_call_id,
                    "name": t.tool_name,
                    "content": t.content or "",
                }
            )
    return msgs


def _get_tau2_runtime(session: StudySession) -> Tau2Runtime | None:
    if session.task_type != TaskType.TAU2:
        return None
    key = str(session.id)
    rt = _TAU2_RUNTIMES.get(key)
    if rt is None:
        split = session.task_split.replace("tau2_", "")  # "retail" / "airline"
        rt = Tau2Runtime(split=split, task_idx=session.task_idx)
        if TAU2_AVAILABLE:
            rt.reset()
        _TAU2_RUNTIMES[key] = rt
    return rt


@router.post("", response_model=ChatResponse)
async def post_chat(
    req: ChatRequest,
    db: Session = Depends(get_session),
) -> ChatResponse:
    session = db.get(StudySession, req.session_id)
    if session is None:
        raise HTTPException(404, "session not found")
    if session.status != SessionStatus.ACTIVE:
        raise HTTPException(409, f"session is {session.status}")

    turns = db.exec(
        select(Turn).where(Turn.session_id == session.id).order_by(Turn.turn_idx)
    ).all()

    if len(turns) >= settings.max_turns:
        raise HTTPException(409, "turn limit reached")

    next_idx = turns[-1].turn_idx + 1 if turns else 0

    # Append the user turn.
    user_turn = Turn(
        session_id=session.id,
        turn_idx=next_idx,
        role=TurnRole.USER,
        content=req.user_message,
    )
    db.add(user_turn)
    turns.append(user_turn)
    next_idx += 1

    model, base_url = model_for_condition(session.condition)
    tau2_rt = _get_tau2_runtime(session)
    tools = tau2_rt.tools_openai if tau2_rt else None

    emitted: list[ChatMessage] = []

    # Tool-calling loop. Safety cap so a misbehaving model can't spin forever.
    for _ in range(8):
        api_messages = _build_api_messages(session, turns)
        reply = await chat_once(
            model=model, messages=api_messages, tools=tools, base_url=base_url
        )

        agent_turn = Turn(
            session_id=session.id,
            turn_idx=next_idx,
            role=TurnRole.AGENT,
            content=reply.get("content"),
            tool_calls=reply.get("tool_calls"),
        )
        db.add(agent_turn)
        turns.append(agent_turn)
        next_idx += 1
        emitted.append(
            ChatMessage(
                role="assistant",
                content=agent_turn.content,
                tool_calls=agent_turn.tool_calls,
            )
        )

        tool_calls = reply.get("tool_calls")
        if not tool_calls:
            break

        if tau2_rt is None:
            # Agent tried to call a tool in a no-tool task (shouldn't happen).
            break

        # Execute each tool call, append tool turns.
        for tc in tool_calls:
            fn = tc["function"]
            args = parse_tool_arguments(fn.get("arguments", ""))
            result = tau2_rt.execute(fn["name"], args)
            tool_turn = Turn(
                session_id=session.id,
                turn_idx=next_idx,
                role=TurnRole.TOOL,
                content=_json_compact(result),
                tool_call_id=tc["id"],
                tool_name=fn["name"],
            )
            db.add(tool_turn)
            turns.append(tool_turn)
            next_idx += 1
            emitted.append(
                ChatMessage(
                    role="tool",
                    content=tool_turn.content,
                    tool_call_id=tool_turn.tool_call_id,
                    tool_name=tool_turn.tool_name,
                )
            )

    db.commit()
    return ChatResponse(messages=emitted, session_status=session.status)


def _json_compact(obj: Any) -> str:
    import json as _json

    return _json.dumps(obj, ensure_ascii=False, default=str)
