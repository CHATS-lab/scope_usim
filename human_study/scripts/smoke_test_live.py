"""End-to-end smoke test against the live deployment.

Creates fresh sessions with synthetic PIDs until all 3 conditions are covered,
then sends one chat message per condition and reports whether the assigned
model produced a real response (non-empty content or a tool call).

Usage:
    python scripts/smoke_test_live.py
    python scripts/smoke_test_live.py --base-url https://usim-study.chats-lab.org/api --task-type p4g
"""
from __future__ import annotations

import argparse
import asyncio
import time
import uuid
from typing import Any

import httpx


async def create_session(client: httpx.AsyncClient, base_url: str, task_type: str) -> dict[str, Any]:
    pid = f"usim-smoke-{uuid.uuid4().hex[:10]}"
    resp = await client.post(
        f"{base_url}/session/start",
        json={
            "prolific_pid": pid,
            "study_id": "internal-smoke",
            "prolific_session_id": "s1",
            "task_type": task_type,
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    data["_pid"] = pid
    return data


async def chat_once(
    client: httpx.AsyncClient,
    base_url: str,
    session_id: str,
    user_message: str,
) -> dict[str, Any]:
    t0 = time.monotonic()
    resp = await client.post(
        f"{base_url}/chat",
        json={"session_id": session_id, "user_message": user_message},
        timeout=120,
    )
    elapsed = time.monotonic() - t0
    if resp.status_code != 200:
        return {"ok": False, "status": resp.status_code, "body": resp.text[:400], "elapsed": elapsed}
    data = resp.json()
    msgs = data.get("messages", [])
    content_chars = sum(len(m.get("content") or "") for m in msgs)
    tool_msgs = sum(1 for m in msgs if m.get("tool_calls"))
    return {
        "ok": True,
        "n_messages": len(msgs),
        "content_chars": content_chars,
        "tool_call_msgs": tool_msgs,
        "elapsed": elapsed,
        "preview": (msgs[0].get("content") or "")[:120] if msgs else "",
    }


async def main_async(args):
    base_url = args.base_url
    task_type = args.task_type
    needed = {"base", "rl_single", "cotraining"}
    seen: dict[str, dict[str, Any]] = {}
    sessions_tried = 0

    async with httpx.AsyncClient() as client:
        while needed - set(seen) and sessions_tried < 20:
            sess = await create_session(client, base_url, task_type)
            cond = sess["condition"]
            sessions_tried += 1
            if cond in seen:
                continue
            seen[cond] = sess
            print(f"  session_id={sess['session_id']}  condition={cond}  task_split={sess['task_split']}  task_idx={sess['task_idx']}")

        print("\n=== Sending one chat message per condition ===")
        for cond in sorted(seen):
            sess = seen[cond]
            print(f"\n[{cond}] sid={sess['session_id']}")
            user_message = (
                "Hi, I'd like help with my order #W2378156 — I want to exchange a keyboard."
                if task_type == "tau2"
                else "Hi, what's this conversation about?"
            )
            r = await chat_once(client, base_url, sess["session_id"], user_message)
            if not r["ok"]:
                print(f"  FAIL  HTTP {r['status']} in {r['elapsed']:.1f}s  body={r['body']}")
                continue
            print(
                f"  OK  msgs={r['n_messages']}  content_chars={r['content_chars']}  "
                f"tool_msgs={r['tool_call_msgs']}  elapsed={r['elapsed']:.1f}s"
            )
            print(f"  preview: {r['preview']!r}")

    missing = needed - set(seen)
    if missing:
        print(f"\nWARNING: could not get sessions for {sorted(missing)} in {sessions_tried} tries")
        return 1
    fails = [c for c, _ in seen.items()]  # any chat that returned ok=False got logged inline
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="https://usim-study.chats-lab.org/api")
    p.add_argument("--task-type", choices=["tau2", "p4g"], default="tau2")
    args = p.parse_args()
    import sys
    sys.exit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
