#!/usr/bin/env bash
# Build + start the human-study stack on the VPS, apply migrations, seed
# tau2 tasks. Run AFTER bootstrap_vps.sh + setup_tunnel.sh.
#
# Usage: ./deploy/bring_up_stack.sh
set -euo pipefail

SSH_HOST="${SSH_HOST:-usim-study}"

echo "==> Pulling latest code on VPS"
ssh "$SSH_HOST" 'cd /opt/usim && git pull --ff-only'

echo "==> docker compose up -d --build"
ssh "$SSH_HOST" 'cd /opt/usim/human_study && docker compose -f deploy/docker-compose.yml up -d --build'

echo "==> Waiting for postgres to be healthy"
ssh "$SSH_HOST" '
  for i in $(seq 1 30); do
    if docker compose -f /opt/usim/human_study/deploy/docker-compose.yml ps postgres | grep -q healthy; then
      echo "postgres healthy"
      break
    fi
    sleep 2
  done
'

echo "==> Applying migrations"
ssh "$SSH_HOST" '
  cd /opt/usim/human_study
  for f in deploy/migrations/*.sql; do
    echo "  applying $f"
    docker compose -f deploy/docker-compose.yml exec -T postgres \
      psql -U usim -d usim_study < "$f"
  done
'

echo "==> Seeding tau2 tasks (retail + airline, 15 each)"
ssh "$SSH_HOST" '
  cd /opt/usim/human_study
  docker compose -f deploy/docker-compose.yml exec -T backend \
    python scripts/seed_tau2_tasks.py --domain retail --limit 15 || true
  docker compose -f deploy/docker-compose.yml exec -T backend \
    python scripts/seed_tau2_tasks.py --domain airline --limit 15 || true
'

echo "==> Stack up. Check:"
echo "  https://usim-study.chats-lab.org/healthz                   (frontend)"
echo "  https://usim-study.chats-lab.org/api/healthz               (backend)"
echo "  https://usim-study.chats-lab.org/study?PROLIFIC_PID=test1&STUDY_ID=internal&SESSION_ID=s1&task_type=tau2"
