"""Capture three light-mode screenshots of the human-study UI for the paper.

Drives the live deployment at https://usim-study.chats-lab.org, creates fresh
sessions for each screenshot via the API, drives each session to the correct
UI state (chatting / survey / debrief), and saves PNGs to a configurable
output directory.

We inject a CSS override at page load to invert the dark theme tokens to a
light palette — the production frontend ships dark mode, and we don't want
to rebuild the container just for paper figures.

Usage:
    python human_study/scripts/capture_screenshots.py
    python human_study/scripts/capture_screenshots.py --output-dir /path/to/figures
"""
from __future__ import annotations

import argparse
import asyncio
import time
import uuid
from pathlib import Path

import httpx
from playwright.async_api import async_playwright


BASE_API = "https://usim-study.chats-lab.org/api"
BASE_WEB = "https://usim-study.chats-lab.org"
DEFAULT_OUT_DIR = Path(__file__).resolve().parents[1] / "screenshots"

# Inject this as a <style> on every page. Targets the Tailwind utility classes
# generated for the custom theme tokens in tailwind.config.ts.
LIGHT_MODE_CSS = """
:root { color-scheme: light !important; }

html, body { background-color: #ffffff !important; color: #0f172a !important; }

.bg-bg { background-color: #ffffff !important; }
.bg-bg\\/60 { background-color: rgba(255, 255, 255, 0.6) !important; }
.bg-panel { background-color: #f8fafc !important; }
.bg-panel\\/40 { background-color: rgba(248, 250, 252, 0.4) !important; }
.bg-panel\\/60 { background-color: rgba(248, 250, 252, 0.6) !important; }
.bg-panelAlt { background-color: #f1f5f9 !important; }

.border-border { border-color: #e2e8f0 !important; }

.text-text { color: #0f172a !important; }
.text-muted { color: #475569 !important; }
.text-muted\\/70 { color: rgba(71, 85, 105, 0.7) !important; }
.text-muted\\/60 { color: rgba(71, 85, 105, 0.6) !important; }

.text-accent { color: #2563eb !important; }
.bg-accent { background-color: #2563eb !important; }
.bg-accent\\/10 { background-color: rgba(37, 99, 235, 0.1) !important; }
.bg-accent\\/20 { background-color: rgba(37, 99, 235, 0.2) !important; }
.border-accent { border-color: #2563eb !important; }
.border-accent\\/30 { border-color: rgba(37, 99, 235, 0.3) !important; }
.border-accent\\/50 { border-color: rgba(37, 99, 235, 0.5) !important; }
.border-accent\\/70 { border-color: rgba(37, 99, 235, 0.7) !important; }

/* Card-like wrappers: lighten their tinted backgrounds */
[class*="bg-emerald-400/10"] { background-color: rgba(16, 185, 129, 0.08) !important; }
[class*="bg-sky-400/10"]     { background-color: rgba(14, 165, 233, 0.08) !important; }
[class*="bg-amber-400/10"]   { background-color: rgba(245, 158, 11, 0.08) !important; }
[class*="bg-red-400/10"]     { background-color: rgba(239, 68, 68, 0.08) !important; }
[class*="bg-emerald-400/5"]  { background-color: rgba(16, 185, 129, 0.05) !important; }

/* Emerald/sky/amber text in callouts: deepen for legibility on light bg */
[class*="text-emerald-50"]  { color: #064e3b !important; }
[class*="text-emerald-200"] { color: #047857 !important; }
[class*="text-emerald-300"] { color: #059669 !important; }
[class*="text-sky-50"]      { color: #0c4a6e !important; }
[class*="text-sky-200"]     { color: #0369a1 !important; }
[class*="text-sky-300"]     { color: #0284c7 !important; }
[class*="text-amber-50"]    { color: #78350f !important; }
[class*="text-amber-200"]   { color: #92400e !important; }
[class*="text-amber-300"]   { color: #b45309 !important; }
[class*="text-red-50"]      { color: #7f1d1d !important; }

/* Inputs and code blocks */
input, textarea { background-color: #f8fafc !important; color: #0f172a !important; }
code { background-color: #f1f5f9 !important; color: #0f172a !important; }

/* Scrollbars */
::-webkit-scrollbar-track { background: #f1f5f9 !important; }
::-webkit-scrollbar-thumb { background: #cbd5e1 !important; }
"""


async def create_session(client: httpx.AsyncClient, pid: str, task_type: str = "tau2") -> dict:
    r = await client.post(
        f"{BASE_API}/session/start",
        json={
            "prolific_pid": pid,
            "study_id": "appendix-screenshots",
            "prolific_session_id": pid + "-s1",
            "task_type": task_type,
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


async def post_user_message(client: httpx.AsyncClient, session_id: str, message: str) -> None:
    # Non-streaming so we don't have to read SSE events here.
    r = await client.post(
        f"{BASE_API}/chat",
        json={"session_id": session_id, "user_message": message},
        timeout=120,
    )
    r.raise_for_status()


async def stop_session(client: httpx.AsyncClient, session_id: str) -> None:
    r = await client.post(f"{BASE_API}/stop", json={"session_id": session_id}, timeout=15)
    r.raise_for_status()


async def submit_survey(client: httpx.AsyncClient, session_id: str, task_type: str) -> None:
    if task_type == "tau2":
        responses = {
            "task_success": 6, "helpfulness": 6, "honesty": 7,
            "efficiency": 5, "instruction_following": 6, "safety": 7,
            "frustration": 3, "overall": 6,
        }
    else:
        responses = {
            "intended_donation": 1.5, "argument_quality": 6, "empathy": 5,
            "manipulation": 3, "engagement": 6, "future_likelihood": 5,
        }
    r = await client.post(
        f"{BASE_API}/survey",
        json={"session_id": session_id, "responses": responses,
              "free_text": "Clear explanations, handled the policy edge cases well."},
        timeout=30,
    )
    r.raise_for_status()


async def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time())
    tau2_landing_pid = f"shot-tau2-landing-{timestamp}"
    tau2_survey_pid = f"shot-tau2-survey-{timestamp}"
    p4g_landing_pid = f"shot-p4g-landing-{timestamp}"
    p4g_survey_pid = f"shot-p4g-survey-{timestamp}"
    debrief_pid = f"shot-debrief-{timestamp}"

    # --- 1. set up backend state for each shot ---------------------------- #
    async with httpx.AsyncClient() as client:
        print("Setting up tau2 landing/chat session…")
        tau2_landing = await create_session(client, tau2_landing_pid, "tau2")
        await post_user_message(
            client, tau2_landing["session_id"],
            "Hi, I'd like to exchange the mechanical keyboard on order #W2378156 "
            "for one with clicky switches.",
        )
        print(f"  tau2 landing: {tau2_landing['session_id']}  cond={tau2_landing['condition']}")

        print("Setting up tau2 survey-state session…")
        tau2_survey = await create_session(client, tau2_survey_pid, "tau2")
        await post_user_message(
            client, tau2_survey["session_id"],
            "Hi, I want to upgrade my return flight to business class.",
        )
        await post_user_message(
            client, tau2_survey["session_id"],
            "Yes please proceed — and could you also add a checked bag?",
        )
        # NOTE: don't call /stop here — the frontend's resume logic on page load
        # routes a STOPPED-not-SURVEY_DONE session straight back into chat mode,
        # so we have to click the End button in playwright to trigger the
        # survey-rendering phase transition.
        print(f"  tau2 survey:  {tau2_survey['session_id']}  cond={tau2_survey['condition']}")

        print("Setting up p4g landing/chat session…")
        p4g_landing = await create_session(client, p4g_landing_pid, "p4g")
        await post_user_message(
            client, p4g_landing["session_id"],
            "Hi, I'm not really interested in donating right now.",
        )
        print(f"  p4g landing:  {p4g_landing['session_id']}  cond={p4g_landing['condition']}")

        print("Setting up p4g survey-state session…")
        p4g_survey = await create_session(client, p4g_survey_pid, "p4g")
        await post_user_message(
            client, p4g_survey["session_id"],
            "Hi — I'm not sure I want to donate anything to be honest.",
        )
        await post_user_message(
            client, p4g_survey["session_id"],
            "Tell me more about how the donation reaches the children.",
        )
        print(f"  p4g survey:   {p4g_survey['session_id']}  cond={p4g_survey['condition']}")

        print("Setting up debrief session (chat history only — drive UI for the rest)…")
        debrief = await create_session(client, debrief_pid, "tau2")
        await post_user_message(
            client, debrief["session_id"],
            "Hi, can you help me cancel my reservation YAX4DR?",
        )
        # NOTE: don't /stop and don't submit /survey via API — we drive those
        # in playwright so the frontend's in-component state has the correct
        # prolific_completion_code/redirect (which the resume flow currently
        # doesn't restore).
        print(f"  debrief session: {debrief['session_id']}  cond={debrief['condition']}")

    # --- 2. drive UI + screenshot via playwright -------------------------- #
    async with async_playwright() as pw:
        browser = await pw.chromium.launch()
        context = await browser.new_context(
            viewport={"width": 1440, "height": 900},
            device_scale_factor=2,  # retina-quality screenshots
            color_scheme="light",
        )
        await context.add_init_script(
            f"""
            const css = `{LIGHT_MODE_CSS}`;
            const inject = () => {{
                if (document.getElementById('paper-light-mode')) return;
                const s = document.createElement('style');
                s.id = 'paper-light-mode';
                s.textContent = css;
                document.head.appendChild(s);
            }};
            if (document.readyState !== 'loading') inject();
            else document.addEventListener('DOMContentLoaded', inject);
            """
        )

        async def open_page(pid: str, task_type: str = "tau2"):
            page = await context.new_page()
            url = (
                f"{BASE_WEB}/study?PROLIFIC_PID={pid}&STUDY_ID=appendix-screenshots"
                f"&SESSION_ID={pid}-s1&task_type={task_type}"
            )
            await page.goto(url, wait_until="networkidle")
            return page

        async def save(page, name: str) -> None:
            out = out_dir / f"fig_human_study_{name}.png"
            await page.screenshot(path=str(out), full_page=False)
            print(f"    saved {out}")

        print("\nCapturing screenshots…")

        async def shot_landing(name: str, pid: str, task_type: str):
            print(f"  → {name}")
            page = await open_page(pid, task_type)
            await page.wait_for_selector("textarea", timeout=15000)
            await page.wait_for_timeout(2500)
            await save(page, name)
            await page.close()

        async def shot_survey(name: str, pid: str, task_type: str, survey_anchor: str):
            print(f"  → {name} (click End → confirm)")
            page = await open_page(pid, task_type)
            await page.wait_for_selector("textarea", timeout=15000)
            await page.wait_for_timeout(1500)
            await page.get_by_role("button", name="End conversation and go to survey").click()
            await page.get_by_role("button", name="End & continue to survey").click()
            await page.wait_for_selector(f"text={survey_anchor}", timeout=20000)
            await page.wait_for_timeout(1500)
            await save(page, name)
            await page.close()

        await shot_landing("tau2_landing", tau2_landing_pid, "tau2")
        await shot_survey("tau2_survey", tau2_survey_pid, "tau2", "Did the agent")
        await shot_landing("p4g_landing", p4g_landing_pid, "p4g")
        await shot_survey("p4g_survey", p4g_survey_pid, "p4g", "How much would you donate")

        # Debrief shot — drive the full UI: chat already loaded, click End,
        # confirm, fill survey, submit. The in-component state then carries
        # prolific_completion_code + redirect_url straight through.
        # Install the redirect-blocking route so the post-submit auto-navigate
        # doesn't fire before we screenshot.
        print("  → debrief (drive UI through survey submission)")
        await context.route(
            "**app.prolific.com/submissions/complete**",
            lambda route: route.abort(),
        )
        page = await open_page(debrief_pid)
        await page.wait_for_selector("textarea", timeout=15000)
        await page.wait_for_timeout(1500)
        await page.get_by_role("button", name="End conversation and go to survey").click()
        await page.get_by_role("button", name="End & continue to survey").click()
        await page.wait_for_selector("text=Did the agent", timeout=20000)
        # The survey has 8 Likert items; each renders as a 7-button group.
        # Buttons literally text-content "6" appear only inside Likert rows,
        # so click each in DOM order — that fills all 8 items with "6".
        button6 = page.get_by_role("button", name="6", exact=True)
        count = await button6.count()
        print(f"    found {count} button-6 candidates; clicking up to 8")
        for i in range(min(count, 8)):
            await button6.nth(i).click()
        # Click Submit.
        await page.get_by_role("button", name="Submit survey").click()
        await page.wait_for_selector("text=Thank you for participating", timeout=20000)
        await page.wait_for_timeout(2000)
        await save(page, "debrief")
        await page.close()

        await browser.close()

    print(f"\nDone. Files written to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"directory for PNG files (default: {DEFAULT_OUT_DIR})",
    )
    args = parser.parse_args()
    asyncio.run(main(args.output_dir.resolve()))
