# USIM Human Study Platform

Custom chat interface for running the Prolific-based human evaluation described
in the human-study appendix of the SCOPE paper. Inspired by the annotation
interface in Mind the Sim2Real Gap (Zhou et al. 2026): two panels, chat on the
left, task instructions on the right, `/stop` to end and submit a survey.

## Layout

```
human_study/
├── backend/     FastAPI app (session, chat, tools, survey)
├── frontend/    Next.js + Tailwind two-panel UI
├── tasks/       YAML task pools (tau2_retail, tau2_airline, p4g)
├── surveys/     Survey schemas (tau2, p4g)
├── deploy/      Docker Compose + cloudflared configs
└── scripts/     Seed τ²-bench tasks from the installed package
```

## Quick start

See `deploy/README.md` for local smoke-test steps. The short version:

```bash
# backend
cd backend && python -m venv .venv && source .venv/bin/activate
pip install -e . && cp .env.example .env   # fill in OPENAI_API_KEY
uvicorn app.main:app --reload --port 8000

# frontend
cd frontend && npm install && cp .env.example .env.local
npm run dev

# visit
open "http://localhost:3000/study?PROLIFIC_PID=test&STUDY_ID=s&SESSION_ID=x&task_type=p4g"
```

## Prolific URL template

```
https://study.example.org/study?PROLIFIC_PID={{%PROLIFIC_PID%}}&STUDY_ID={{%STUDY_ID%}}&SESSION_ID={{%SESSION_ID%}}&task_type=tau2
```

Replace `study.example.org` with your own deployment's domain.

The backend assigns a condition server-side on session creation, stratified to
keep cell counts balanced. It defines three conditions: `base`, `rl_single`,
and `cotraining` (the paper's Co-Training policy). Each condition is served by
its own `MODEL_*` / `OPENAI_BASE_URL_*` pair in `backend/.env`. The app has no
Verbalized Sampling condition; see the note below.

## Configuration notes

- τ²-bench tool dispatch uses a direct wrapper around tau2-bench's `Environment`
  object (see `backend/app/services/tau2_tools.py`). The import path assumes
  tau2-bench is installed in the same venv. If you hit import errors, fall back
  to the P4G flow, which has no tool dependency.
- `backend/.env.example` points every condition at the placeholder model
  `gpt-5.4`. To serve trained checkpoints, set `MODEL_*` and
  `OPENAI_BASE_URL_*` to your OpenAI-compatible (for example SGLang) endpoints.
- The task pools in `tasks/` are populated: 15 retail and 15 airline τ²-bench
  scenarios and 30 P4G personas. Regenerate them with
  `scripts/seed_tau2_tasks.py` (needs `tau2-bench` on the `PYTHONPATH`) and
  `scripts/seed_p4g_tasks.py`.
- The paper's human study reports a fourth condition, Verbalized Sampling.
  This app only defines `base`, `rl_single`, and `cotraining`, so running that
  arm needs an extra `Condition` value and model entry in the backend.
- `scripts/smoke_test_live.py` and `scripts/capture_screenshots.py` create real
  sessions (and, for the screenshot script, survey responses) in the database
  of whatever deployment you point them at. Both require an explicit
  `--base-url`; do not point them at a deployment that is collecting study data.
