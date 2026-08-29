#!/usr/bin/env bash
# Configure Cloudflare Tunnel for usim-study.chats-lab.org.
#
# Run AFTER you've completed `cloudflared tunnel login` interactively on
# the VPS (that writes ~/.cloudflared/cert.pem on the VPS, which is what
# `tunnel create` and `tunnel route dns` need).
#
# Usage: ./deploy/setup_tunnel.sh
set -euo pipefail

SSH_HOST="${SSH_HOST:-usim-study}"
TUNNEL_NAME="${TUNNEL_NAME:-usim-study}"
HOSTNAME="${HOSTNAME_FQDN:-usim-study.chats-lab.org}"

echo "==> Installing cloudflared on VPS (if needed)"
ssh "$SSH_HOST" bash <<'REMOTE'
set -euo pipefail
if ! command -v cloudflared >/dev/null 2>&1; then
  curl -L --fail -o /tmp/cloudflared.deb \
    https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
  dpkg -i /tmp/cloudflared.deb
fi
cloudflared --version
REMOTE

echo "==> Creating tunnel '$TUNNEL_NAME' (idempotent)"
ssh "$SSH_HOST" bash <<REMOTE
set -euo pipefail
if [[ ! -f ~/.cloudflared/cert.pem ]]; then
  echo "ERROR: ~/.cloudflared/cert.pem missing on VPS." >&2
  echo "Run: ssh $SSH_HOST 'cloudflared tunnel login'   and complete the browser flow." >&2
  exit 1
fi

# Skip create if a tunnel by this name already exists.
if cloudflared tunnel list 2>/dev/null | awk 'NR>1 {print \$2}' | grep -qx "$TUNNEL_NAME"; then
  echo "Tunnel '$TUNNEL_NAME' already exists, skipping create."
else
  cloudflared tunnel create "$TUNNEL_NAME"
fi

TUNNEL_UUID=\$(cloudflared tunnel list 2>/dev/null | awk -v n="$TUNNEL_NAME" 'NR>1 && \$2==n {print \$1}')
echo "Tunnel UUID: \$TUNNEL_UUID"

# Move credentials JSON to /etc/cloudflared so the systemd service can read it.
mkdir -p /etc/cloudflared
if [[ -f /root/.cloudflared/\$TUNNEL_UUID.json ]]; then
  install -m 0600 /root/.cloudflared/\$TUNNEL_UUID.json /etc/cloudflared/${TUNNEL_NAME}.json
fi

cat > /etc/cloudflared/config.yml <<EOF
tunnel: \$TUNNEL_UUID
credentials-file: /etc/cloudflared/${TUNNEL_NAME}.json

ingress:
  - hostname: $HOSTNAME
    service: http://127.0.0.1:3000
  - service: http_status:404
EOF

# Re-route DNS: replaces the existing CNAME so it points at the new tunnel.
cloudflared tunnel route dns "$TUNNEL_NAME" "$HOSTNAME" || true

# Install as systemd service (idempotent).
cloudflared service install || true
systemctl enable --now cloudflared
systemctl status --no-pager cloudflared | head -12
REMOTE

echo
echo "==> Tunnel up. DNS at $HOSTNAME now resolves to this tunnel."
