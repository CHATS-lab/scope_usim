"""Chat endpoints: non-streaming POST /chat and streaming POST /chat/stream."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, AsyncIterator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlmodel import Session, select

from ..config import settings
from ..db import engine, get_session
from ..models import EndedReason, SessionStatus, StudySession, TaskType, Turn, TurnRole
from ..schemas import ChatMessage, ChatRequest, ChatResponse
from ..services.conditions import model_for_condition
from ..services.llm import build_client
from ..services.tau2_tools import Tau2Runtime, parse_tool_arguments

router = APIRouter(tags=["chat"])

# One Tau2Runtime per session. In multi-worker deployments we'd pin sessions
# to workers or move to Redis; single-worker pilot doesn't need that.
_TAU2_RUNTIMES: dict[str, Tau2Runtime] = {}

_TRANSFER_TOOL = "transfer_to_human_agents"


def _transferred(tool_calls: list[dict[str, Any]] | None) -> bool:
    """True if the agent called the transfer-to-human-agent tool this turn."""
    if not tool_calls:
        return False
    return any(tc.get("function", {}).get("name") == _TRANSFER_TOOL for tc in tool_calls)


_TAU2_CONFIRMATION_RULES = (
    "Action-confirmation rules (very important):\n"
    "- Never tell the user that a change has been made unless you have just "
    "called the corresponding tool in this turn AND the tool returned a "
    "successful result. Do not paraphrase a planned action as if it were "
    "completed.\n"
    "- If the user agrees to a change AND adds a different question or "
    "request in the same message (e.g. 'yes, and could you also ...'), "
    "treat the agreement as the confirmation: invoke the tool to perform "
    "the agreed action FIRST, then address the additional request.\n"
    "- After a tool call you intended to make does not appear in your "
    "tool_calls, do not claim the action happened. Either call the tool now "
    "or tell the user the action was not completed.\n"
)


def _system_prompt(session: StudySession, tau2_rt: Tau2Runtime | None) -> str:
    if session.task_type == TaskType.TAU2 and tau2_rt is not None:
        return (
            "You are a helpful customer-service agent. The user is a real customer. "
            "Use the provided tools to look up information and complete their request. "
            "Be concise and verify details before taking irreversible actions.\n\n"
            f"{_TAU2_CONFIRMATION_RULES}\n"
            "Domain policy:\n" + tau2_rt.env.get_policy()
        )
    return (
        "You are a persuader trying to encourage the user to donate to Save the "
        "Children, an international charity. Engage them in a natural, respectful "
        "conversation."
    )


def _build_api_messages(
    session: StudySession, turns: list[Turn], tau2_rt: Tau2Runtime | None
) -> list[dict[str, Any]]:
    msgs: list[dict[str, Any]] = [
        {"role": "system", "content": _system_prompt(session, tau2_rt)}
    ]
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
        # task_payload stored by session route includes the resolved tau2 task id.
        task_id = session.task_payload.get("metadata", {}).get(
            "tau2_task_id"
        ) or str(session.task_idx)
        rt = Tau2Runtime(split=split, task_id=task_id)
        rt.reset()
        _TAU2_RUNTIMES[key] = rt
    return rt


def _max_token_kwarg(model: str, max_tokens: int, has_tools: bool = False) -> dict[str, Any]:
    # gpt-5.x and o-series: no temperature, use max_completion_tokens.
    # reasoning_effort is *not* permitted with function tools on
    # /v1/chat/completions for gpt-5.x — OpenAI returns 400 and redirects to
    # /v1/responses. We drop it when tools are present so tau2 (which needs
    # tools) keeps working; p4g (no tools) keeps medium reasoning.
    if model.startswith(("gpt-5", "o1", "o3", "o4")):
        kw: dict[str, Any] = {"max_completion_tokens": max_tokens}
        if not has_tools:
            kw["reasoning_effort"] = "medium"
        return kw
    return {"max_tokens": max_tokens, "temperature": 0.7}


@router.post("/chat", response_model=ChatResponse)
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
    # Limit is on *user turns* only. Agent and tool turns are bookkeeping that
    # doesn't map to what the participant sees, so counting them here would
    # reject well under 30 user messages on tool-heavy (tau2) sessions.
    user_turns = sum(1 for t in turns if t.role == TurnRole.USER)
    if user_turns >= settings.max_turns:
        raise HTTPException(
            409,
            "You've reached the maximum number of turns for this task. Please end the conversation with /stop to continue to the survey.",
        )
    next_idx = turns[-1].turn_idx + 1 if turns else 0

    user_turn = Turn(
        session_id=session.id,
        turn_idx=next_idx,
        role=TurnRole.USER,
        content=req.user_message,
    )
    db.add(user_turn)
    turns.append(user_turn)
    next_idx += 1

    model, base_url, api_key = model_for_condition(session.condition)
    tau2_rt = _get_tau2_runtime(session)
    tools = tau2_rt.tools_openai if tau2_rt else None
    client = build_client(base_url, api_key)
    emitted: list[ChatMessage] = []

    for _ in range(8):
        api_messages = _build_api_messages(session, turns, tau2_rt)
        kwargs = {
            "model": model,
            "messages": api_messages,
            **_max_token_kwarg(model, 1024, has_tools=bool(tools)),
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        # Qwen3 base models default to chain-of-thought; disable via SGLang's
        # per-request chat_template_kwargs so participants see only the final
        # reply. Only sent when base_url is overridden (i.e. we're hitting our
        # own SGLang endpoint), since OpenAI rejects unknown request params.
        if base_url:
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        resp = await client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        content = choice.message.content
        tool_calls = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in (choice.message.tool_calls or [])
        ] or None

        agent_turn = Turn(
            session_id=session.id,
            turn_idx=next_idx,
            role=TurnRole.AGENT,
            content=content,
            tool_calls=tool_calls,
        )
        db.add(agent_turn)
        turns.append(agent_turn)
        next_idx += 1
        emitted.append(
            ChatMessage(role="assistant", content=content, tool_calls=tool_calls)
        )

        if not tool_calls or tau2_rt is None:
            break

        for tc in tool_calls:
            fn = tc["function"]
            args = parse_tool_arguments(fn.get("arguments", ""))
            result = tau2_rt.execute(fn["name"], args)
            tool_turn = Turn(
                session_id=session.id,
                turn_idx=next_idx,
                role=TurnRole.TOOL,
                content=json.dumps(result, ensure_ascii=False, default=str),
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

        # tau2 marks transfer as an irreversible hand-off, so we end the
        # conversation rather than letting the agent keep going. Outcome
        # evaluation still runs at survey time — some scenarios expect escalation.
        if _transferred(tool_calls):
            session.status = SessionStatus.STOPPED
            session.ended_reason = EndedReason.AGENT_TRANSFERRED
            session.ended_at = datetime.now(timezone.utc)
            db.add(session)
            break

    db.commit()
    return ChatResponse(messages=emitted, session_status=session.status)


# --------------------------------------------------------------------------- #
# Streaming variant                                                           #
# --------------------------------------------------------------------------- #


def _sse(event: str, data: dict[str, Any]) -> bytes:
    """Format a Server-Sent Event frame."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode()


async def _stream_chat(req: ChatRequest) -> AsyncIterator[bytes]:
    # Use a short-lived session per event batch; write turns as we go so the DB
    # reflects partial progress even if the client disconnects mid-stream.
    with Session(engine) as db:
        session = db.get(StudySession, req.session_id)
        if session is None:
            yield _sse("error", {"message": "session not found"})
            return
        if session.status != SessionStatus.ACTIVE:
            yield _sse("error", {"message": f"session is {session.status}"})
            return

        turns = db.exec(
            select(Turn).where(Turn.session_id == session.id).order_by(Turn.turn_idx)
        ).all()
        user_turns = sum(1 for t in turns if t.role == TurnRole.USER)
        if user_turns >= settings.max_turns:
            yield _sse(
                "error",
                {
                    "message": (
                        "You've reached the maximum number of turns for this task. "
                        "Please end the conversation with /stop to continue to the "
                        "survey."
                    )
                },
            )
            return
        next_idx = turns[-1].turn_idx + 1 if turns else 0

        user_turn = Turn(
            session_id=session.id,
            turn_idx=next_idx,
            role=TurnRole.USER,
            content=req.user_message,
        )
        db.add(user_turn)
        db.commit()
        turns.append(user_turn)
        next_idx += 1

        model, base_url, api_key = model_for_condition(session.condition)
        tau2_rt = _get_tau2_runtime(session)
        tools = tau2_rt.tools_openai if tau2_rt else None
        client = build_client(base_url, api_key)

        for _ in range(8):
            api_messages = _build_api_messages(session, turns, tau2_rt)
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": api_messages,
                "stream": True,
                **_max_token_kwarg(model, 1024, has_tools=bool(tools)),
            }
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            # Qwen3 base models default to chain-of-thought; disable via SGLang's
            # per-request chat_template_kwargs. ONLY send when base_url is set
            # (our own SGLang endpoint) — OpenAI rejects unknown request params
            # with HTTP 400 and would silently break every gpt-* session.
            if base_url:
                kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

            yield _sse("assistant_start", {})
            content_buf = ""
            tool_calls_buf: dict[int, dict[str, Any]] = {}

            try:
                stream = await client.chat.completions.create(**kwargs)
                async for chunk in stream:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    if delta.content:
                        content_buf += delta.content
                        yield _sse("assistant_delta", {"content": delta.content})
                    for tc_delta in delta.tool_calls or []:
                        slot = tool_calls_buf.setdefault(
                            tc_delta.index,
                            {"id": "", "name": "", "arguments": ""},
                        )
                        if tc_delta.id:
                            slot["id"] = tc_delta.id
                        if tc_delta.function and tc_delta.function.name:
                            slot["name"] = tc_delta.function.name
                        if tc_delta.function and tc_delta.function.arguments:
                            slot["arguments"] += tc_delta.function.arguments
            except Exception as e:  # noqa: BLE001
                yield _sse("error", {"message": f"{type(e).__name__}: {e}"})
                return

            tool_calls: list[dict[str, Any]] | None = None
            if tool_calls_buf:
                tool_calls = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": tc["arguments"]},
                    }
                    for tc in tool_calls_buf.values()
                ]

            # Persist assistant turn.
            agent_turn = Turn(
                session_id=session.id,
                turn_idx=next_idx,
                role=TurnRole.AGENT,
                content=content_buf or None,
                tool_calls=tool_calls,
            )
            db.add(agent_turn)
            db.commit()
            turns.append(agent_turn)
            next_idx += 1

            yield _sse(
                "assistant_end",
                {"content": content_buf or None, "tool_calls": tool_calls},
            )

            if not tool_calls or tau2_rt is None:
                break

            # Execute each tool call, stream the result as it completes.
            for tc in tool_calls:
                fn = tc["function"]
                args = parse_tool_arguments(fn.get("arguments", ""))
                yield _sse(
                    "tool_start",
                    {"tool_call_id": tc["id"], "name": fn["name"], "arguments": args},
                )
                result = tau2_rt.execute(fn["name"], args)
                result_json = json.dumps(result, ensure_ascii=False, default=str)
                tool_turn = Turn(
                    session_id=session.id,
                    turn_idx=next_idx,
                    role=TurnRole.TOOL,
                    content=result_json,
                    tool_call_id=tc["id"],
                    tool_name=fn["name"],
                )
                db.add(tool_turn)
                db.commit()
                turns.append(tool_turn)
                next_idx += 1
                yield _sse(
                    "tool_end",
                    {"tool_call_id": tc["id"], "name": fn["name"], "result": result},
                )

            # Transfer-to-human is a hard hand-off: stop the loop, mark the
            # session, and let the UI jump to the survey. Outcome is still
            # evaluated post-hoc (some tau2 scenarios expect escalation).
            if _transferred(tool_calls):
                session.status = SessionStatus.STOPPED
                session.ended_reason = EndedReason.AGENT_TRANSFERRED
                session.ended_at = datetime.now(timezone.utc)
                db.add(session)
                db.commit()
                yield _sse(
                    "session_ended",
                    {
                        "session_status": session.status,
                        "ended_reason": session.ended_reason.value,
                    },
                )
                return

        yield _sse("done", {"session_status": session.status})


@router.post("/chat/stream")
async def post_chat_stream(req: ChatRequest) -> StreamingResponse:
    return StreamingResponse(
        _stream_chat(req),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering
            "Connection": "keep-alive",
        },
    )
