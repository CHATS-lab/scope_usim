#!/usr/bin/env bash
# Bootstrap a freshly-provisioned Ubuntu 22.04 Vultr VPS for the USIM
# human-study deployment.
#
# Run this from your laptop AFTER:
#   - The VPS is reachable via the `usim-study` SSH host (update ~/.ssh/config
#     with the new IP first)
#   - You have a checkout of this repo at /opt/usim on the VPS (the script
#     git-clones if not)
#
# Usage:
#   ./deploy/bootstrap_vps.sh
#
# Idempotent: re-running is safe (skips already-installed pieces).
set -euo pipefail

SSH_HOST="${SSH_HOST:-usim-study}"
REPO_URL="${REPO_URL:-https://github.com/CHATS-lab/scope_usim.git}"
REPO_BRANCH="${REPO_BRANCH:-main}"
REPO_DIR="/opt/usim"

echo "==> Verifying SSH reachability to $SSH_HOST"
ssh -o BatchMode=yes -o ConnectTimeout=10 "$SSH_HOST" "echo ok" >/dev/null

echo "==> Installing docker + docker-compose plugin on the VPS"
ssh "$SSH_HOST" bash <<'REMOTE'
set -euo pipefail
if ! command -v docker >/dev/null 2>&1; then
  apt update
  apt install -y ca-certificates curl gnupg lsb-release
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  chmod a+r /etc/apt/keyrings/docker.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" \
    > /etc/apt/sources.list.d/docker.list
  apt update
  apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
fi
docker --version
docker compose version
REMOTE

echo "==> Cloning repo to $REPO_DIR (if needed)"
ssh "$SSH_HOST" bash <<REMOTE
set -euo pipefail
if [[ ! -d "$REPO_DIR/.git" ]]; then
  git clone --branch "$REPO_BRANCH" "$REPO_URL" "$REPO_DIR"
else
  cd "$REPO_DIR" && git fetch origin && git checkout "$REPO_BRANCH" && git pull --ff-only
fi
REMOTE

echo "==> Generating per-VPS secrets if not already present"
ssh "$SSH_HOST" bash <<'REMOTE'
set -euo pipefail
cd /opt/usim/human_study
if [[ ! -f deploy/.env ]]; then
  PG_PASS=$(openssl rand -base64 32 | tr -d '=+/' | head -c 32)
  cat > deploy/.env <<EOF
POSTGRES_USER=usim
POSTGRES_PASSWORD=${PG_PASS}
POSTGRES_DB=usim_study
EOF
  echo "  wrote deploy/.env"
fi
REMOTE

echo "==> Copying backend/.env from laptop -> VPS"
# The Modal URLs + OpenAI key live in our local backend/.env. We rewrite the
# DATABASE_URL line to use the docker-compose service hostname + the password
# we just generated on the VPS.
PG_PASS=$(ssh "$SSH_HOST" 'grep ^POSTGRES_PASSWORD= /opt/usim/human_study/deploy/.env | cut -d= -f2-')
COMPLETION_SECRET=$(openssl rand -base64 48 | tr -d '=+/' | head -c 64)

LOCAL_ENV="$(dirname "$0")/../backend/.env"
if [[ ! -f "$LOCAL_ENV" ]]; then
  echo "ERROR: $LOCAL_ENV missing. Configure it locally first." >&2
  exit 1
fi

# Build a VPS-specific .env (postgres uses the docker service hostname, fresh
# completion secret, no FRONTEND_ORIGIN since cloudflared fronts the same
# origin in prod).
TMP=$(mktemp)
awk -v pw="$PG_PASS" -v sec="$COMPLETION_SECRET" '
  BEGIN { ds=0; cs=0; fo=0 }
  /^DATABASE_URL=/         { print "DATABASE_URL=postgresql+psycopg://usim:" pw "@postgres:5432/usim_study"; ds=1; next }
  /^COMPLETION_CODE_SECRET=/ { print "COMPLETION_CODE_SECRET=" sec; cs=1; next }
  /^FRONTEND_ORIGIN=/      { print "FRONTEND_ORIGIN="; fo=1; next }
  { print }
  END {
    if (!ds) print "DATABASE_URL=postgresql+psycopg://usim:" pw "@postgres:5432/usim_study"
    if (!cs) print "COMPLETION_CODE_SECRET=" sec
    if (!fo) print "FRONTEND_ORIGIN="
  }' "$LOCAL_ENV" > "$TMP"

scp "$TMP" "$SSH_HOST:/opt/usim/human_study/backend/.env"
rm -f "$TMP"

echo
echo "==> Bootstrap complete."
echo
echo "Next steps:"
echo "  1) ssh $SSH_HOST 'cloudflared tunnel login'   # interactive, needs browser"
echo "  2) Re-run ./deploy/setup_tunnel.sh"
echo "  3) ./deploy/bring_up_stack.sh"
