# Deploying the USIM Human Study

This directory holds the deployment artifacts for a single-VPS setup (Docker
Compose for Postgres, backend and frontend) exposed through a Cloudflare Tunnel.

The helper scripts run commands over SSH and take the target from environment
variables. None of them has a default target, so nothing touches a remote host
or database until you set these:

| Variable | Used by | Meaning |
| --- | --- | --- |
| `SSH_HOST` | `bootstrap_vps.sh`, `setup_tunnel.sh`, `bring_up_stack.sh` | SSH host (alias or `user@ip`) of your VPS |
| `STUDY_DOMAIN` | `setup_tunnel.sh`, `bring_up_stack.sh` | public hostname of the study, e.g. `study.example.org` |
| `REPO_URL`, `REPO_BRANCH` | `bootstrap_vps.sh` | repository to clone on the VPS (defaults: this repository, `main`) |

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

## VPS bootstrap

Assume Ubuntu 22.04 on a fresh 4 vCPU / 16 GB instance from any provider.
`bootstrap_vps.sh` automates the steps below from your laptop
(`SSH_HOST=<your-vps> ./deploy/bootstrap_vps.sh`); the manual equivalent is:

```bash
# On the VPS
sudo apt update && sudo apt install -y docker.io docker-compose-plugin
git clone <your-fork> /opt/usim
cd /opt/usim
cd human_study
cp backend/.env.example backend/.env
$EDITOR backend/.env             # set OPENAI_API_KEY, COMPLETION_CODE_SECRET
cd deploy
docker compose up -d --build
```

## Cloudflare Tunnel

Follow the inline instructions in `cloudflared-config.yml`, or run
`SSH_HOST=<your-vps> STUDY_DOMAIN=study.example.org ./deploy/setup_tunnel.sh`,
then confirm `https://study.example.org` (your domain) reaches the frontend.

## Migrations

`init_db()` only uses `SQLModel.create_all()` which creates missing tables but
never alters existing ones. When a release adds columns, apply the matching
SQL file in `deploy/migrations/` against the production Postgres container before
rolling out the new code. `bring_up_stack.sh` applies every file in
`deploy/migrations/` on the VPS given by `SSH_HOST`; to apply one by hand:

```bash
# Copy the migration onto the VPS (or git pull) and then:
docker compose exec -T postgres \
  psql -U usim -d usim_study \
  < deploy/migrations/001_variant_and_ended_reason.sql
```

Migrations are idempotent (`ADD COLUMN IF NOT EXISTS`), so re-running is safe.

## v1 → v2 instruction panel rollout

The `?variant=v2` path requires columns added by `001_variant_and_ended_reason.sql`.
Rollout order on the VPS:

```bash
cd /opt/usim && git pull
cd human_study

# 1. Apply schema migration (idempotent, data-preserving)
docker compose -f deploy/docker-compose.yml exec -T postgres \
  psql -U usim -d usim_study \
  < deploy/migrations/001_variant_and_ended_reason.sql

# 2. Regenerate task YAMLs so `structured` is present
python human_study/scripts/seed_tau2_tasks.py --domain retail --limit 15
python human_study/scripts/seed_tau2_tasks.py --domain airline --limit 15

# 3. Rebuild + restart
docker compose -f deploy/docker-compose.yml up -d --build

# 4. Smoke-test both variants:
#    /study?...&variant=v1  -> prose instruction panel
#    /study?...&variant=v2  -> structured instruction panel
#    Trigger transfer_to_human_agents -> conversation auto-ends, survey opens.
```

## Data access

From your laptop, with `SSH_HOST` set to your VPS:

```bash
ssh "$SSH_HOST" "docker compose -f /opt/usim/human_study/deploy/docker-compose.yml exec postgres \\
  pg_dump -U usim usim_study" > study_dump.sql
```

## Operational checklist

- [ ] Rotate your VPS provider's API token before provisioning
- [ ] Rotate `COMPLETION_CODE_SECRET` before launch; don't reuse the example
- [ ] Set up nightly `pg_dump` cron + off-box backup
- [ ] Add Prolific study with URL template:
      `https://study.example.org/study?PROLIFIC_PID={{%PROLIFIC_PID%}}&STUDY_ID={{%STUDY_ID%}}&SESSION_ID={{%SESSION_ID%}}&task_type=tau2`
- [ ] Pilot with internal testers before opening Prolific
