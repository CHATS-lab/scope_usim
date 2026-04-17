# Deploying the USIM Human Study

This directory holds the deployment artifacts for the Vultr + Cloudflare Tunnel
setup described in `~/notes/projects/usim_human_study.md`.

## Local smoke test (macOS / Linux)

```bash
# 1) Backend
cd human_study/backend
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env            # fill in OPENAI_API_KEY
createdb usim_study             # or use docker: docker compose up postgres
uvicorn app.main:app --reload --port 8000

# 2) Frontend (separate terminal)
cd human_study/frontend
npm install
cp .env.example .env.local
npm run dev

# 3) Visit http://localhost:3000/study?PROLIFIC_PID=testuser&STUDY_ID=s1&SESSION_ID=session1&task_type=p4g
```

## Vultr bootstrap

Assume Ubuntu 22.04, a fresh 4 vCPU / 16 GB High-Frequency instance.
Export `VULTR_API_KEY` locally and provision via `vultr-cli` or the web console.

```bash
# On the VPS
sudo apt update && sudo apt install -y docker.io docker-compose-plugin
git clone <your-fork> /opt/usim
cd /opt/usim && git checkout human_study
cd human_study
cp backend/.env.example backend/.env
$EDITOR backend/.env             # set OPENAI_API_KEY, COMPLETION_CODE_SECRET
cd deploy
docker compose up -d --build
```

## Cloudflare Tunnel

Follow the inline instructions in `cloudflared-config.yml`, then confirm
`https://study.your-domain.com` reaches the frontend.

## Data access

From your laptop:

```bash
ssh vultr "docker compose -f /opt/usim/human_study/deploy/docker-compose.yml exec postgres \\
  pg_dump -U usim usim_study" > study_dump.sql
```

## Operational checklist

- [ ] Rotate the Vultr API token before provisioning
- [ ] Rotate `COMPLETION_CODE_SECRET` before launch; don't reuse the example
- [ ] Set up nightly `pg_dump` cron + off-box backup
- [ ] Add Prolific study with URL template:
      `https://study.your-domain.com/study?PROLIFIC_PID={{%PROLIFIC_PID%}}&STUDY_ID={{%STUDY_ID%}}&SESSION_ID={{%SESSION_ID%}}&task_type=tau2`
- [ ] Pilot with internal testers before opening Prolific
