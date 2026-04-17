# USIM Human Study Platform

Custom chat interface for running the Prolific-based human evaluation described
in the SCOPE paper (Appendix D). Inspired by the annotation interface in
Mind the Sim2Real Gap (Zhou et al. 2026) — two panels, chat on the left, task
instructions on the right, `/stop` to end and submit a survey.

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
https://study.your-domain.com/study?PROLIFIC_PID={{%PROLIFIC_PID%}}&STUDY_ID={{%STUDY_ID%}}&SESSION_ID={{%SESSION_ID%}}&task_type=tau2
```

Condition (Base / RL-Single / Co-training) is assigned server-side on session
creation, stratified to keep cell counts balanced.

## Known open items

- τ²-bench tool dispatch uses a direct wrapper around tau2-bench's `Environment`
  object (see `backend/app/services/tau2_tools.py`). The import path assumes
  tau2-bench is installed in the same venv. If you hit import errors, fall back
  to the P4G flow which has no tool dependency.
- OpenAI models are placeholders (`gpt-5.4`). Swap to our SGLang endpoints by
  setting `OPENAI_BASE_URL_*` and `MODEL_*` in `backend/.env`.
- Task pool files in `tasks/tau2_*.yaml` are stubs. Run
  `scripts/seed_tau2_tasks.py` once `tau2-bench` is on the PYTHONPATH.

## Related docs

- Plan: `~/notes/projects/usim_human_study.md`
- IRB: `~/Downloads/Usim -- IRB/human_study_plan.md`
- Paper appendix: `scope_paper/neurips2026_conference.tex` (Appendix D)
