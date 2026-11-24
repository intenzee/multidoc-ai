#!/bin/sh
# Put MultiDoc AI online: a free Cloudflare tunnel to the backend on this Mac, and the
# website on Vercel (https://multidoc-ai.vercel.app) pointed at that tunnel.
#
# Start the backend first (in another terminal), then run:   scripts/go_live.sh
# The site works while this script runs. Ctrl-C takes it offline.
# A free "quick tunnel" gets a new address each time, so every start redeploys the site.
set -eu
root="$(cd "$(dirname "$0")/.." && pwd)"
site="https://multidoc-ai.vercel.app"

if ! curl -sf -o /dev/null http://127.0.0.1:8000/api/health; then
  echo "The backend is not running. Start it first:"
  echo "  cd $root && .venv/bin/uvicorn backend.main:app --reload --reload-dir backend"
  exit 1
fi
command -v cloudflared >/dev/null || { echo "Install the tunnel first: brew install cloudflared"; exit 1; }

log="$(mktemp)"
cloudflared tunnel --no-autoupdate --url http://127.0.0.1:8000 >"$log" 2>&1 &
tunnel=$!
trap 'kill $tunnel 2>/dev/null; rm -f "$log"; echo; echo "Offline. The website stays up but cannot answer until you run this again."' EXIT
trap 'exit 130' INT TERM

echo "Starting the tunnel..."
url=""
for _ in $(seq 1 60); do
  url="$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' "$log" | head -1 || true)"
  [ -n "$url" ] && break
  sleep 1
done
[ -n "$url" ] || { echo "The tunnel did not start:"; tail -5 "$log"; exit 1; }

echo "Waiting for $url ..."
for _ in $(seq 1 45); do
  curl -sf -o /dev/null "$url/api/health" && break
  sleep 2
done
curl -sf -o /dev/null "$url/api/health" || { echo "The tunnel is not reachable."; exit 1; }

echo "Deploying the website (about 30 seconds)..."
(cd "$root/frontend" && npx --yes vercel deploy --prod --yes \
   --build-env VITE_API_BASE="$url" >/dev/null 2>&1) \
  || { echo "Vercel deploy failed. Check: cd frontend && npx vercel whoami"; exit 1; }

echo
echo "Online: $site"
echo "Keep this window open. Press Ctrl-C to go offline."
wait $tunnel
