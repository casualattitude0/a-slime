#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RED='\033[0;31m'; YELLOW='\033[1;33m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${CYAN}[info]${NC}  $*"; }
ok()    { echo -e "${GREEN}[ok]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[warn]${NC}  $*"; }
err()   { echo -e "${RED}[error]${NC} $*" >&2; }

free_port() {
  local port="$1"
  if ! command -v lsof >/dev/null 2>&1; then
    return 0
  fi
  local pids
  pids="$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -n "$pids" ]]; then
    warn "Port $port is in use. Stopping process(es): $pids"
    kill $pids 2>/dev/null || true
    sleep 0.4
    pids="$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
    if [[ -n "$pids" ]]; then
      warn "Force stopping process(es) on port $port: $pids"
      kill -9 $pids 2>/dev/null || true
    fi
  fi
}

if [[ -n "${PYTHON:-}" ]]; then
  :
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON="$ROOT/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
elif command -v python >/dev/null 2>&1; then
  PYTHON=python
else
  err "Python not found. Install Python 3 or create a .venv."
  exit 1
fi
ok "Python → $PYTHON ($($PYTHON --version 2>&1))"

if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  info "Creating .venv …"
  $PYTHON -m venv "$ROOT/.venv"
  PYTHON="$ROOT/.venv/bin/python"
  ok ".venv created"
fi

if [[ -f "$ROOT/requirements.txt" ]]; then
  info "Checking Python dependencies …"
  "$ROOT/.venv/bin/pip" install -q --upgrade pip
  "$ROOT/.venv/bin/pip" install -q -r "$ROOT/requirements.txt"
  ok "Python dependencies up to date"
fi

if [[ -f "$ROOT/.env" ]]; then
  set -a; source "$ROOT/.env"; set +a
fi
if [[ -z "${GOOGLE_API_KEY:-}" && -z "${GEMINI_API_KEY:-}" ]]; then
  warn "GOOGLE_API_KEY / GEMINI_API_KEY not set."
fi

FRONTEND="$ROOT/frontend"
if [[ ! -f "$FRONTEND/package.json" ]]; then
  err "frontend/package.json not found."
  exit 1
fi
if ! command -v npm >/dev/null 2>&1; then
  err "npm not found."
  exit 1
fi

if [[ ! -d "$FRONTEND/node_modules" ]]; then
  info "Installing frontend npm packages …"
  npm --prefix "$FRONTEND" install --silent
  ok "npm install done"
fi

if [[ ! -f "$HOME/.cargo/env" ]]; then
  err "Rust toolchain not found at \$HOME/.cargo/env."
  err "Install it with: curl https://sh.rustup.rs -sSf | sh -s -- -y"
  exit 1
fi

pkill -f "uvicorn src.web_server:app" 2>/dev/null && info "Stopped previous web server" || true
pkill -f "tauri dev" 2>/dev/null && info "Stopped previous tauri dev" || true
sleep 0.3

WEB_HOST="${WEB_HOST:-127.0.0.1}"
WEB_PORT="${WEB_PORT:-8765}"
VITE_PORT="${VITE_PORT:-5173}"

free_port "$WEB_PORT"
free_port "$VITE_PORT"

cleanup() {
  kill "${WEB_PID:-}" "${APP_PID:-}" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

info "Starting web server on http://${WEB_HOST}:${WEB_PORT} …"
"$PYTHON" -m uvicorn src.web_server:app \
  --host "$WEB_HOST" \
  --port "$WEB_PORT" \
  --log-level info &
WEB_PID=$!

for i in $(seq 1 30); do
  sleep 0.5
  if curl -sf "http://${WEB_HOST}:${WEB_PORT}/" >/dev/null 2>&1; then
    ok "Web server ready → http://${WEB_HOST}:${WEB_PORT}"
    break
  fi
  if ! kill -0 "$WEB_PID" 2>/dev/null; then
    err "Web server exited unexpectedly. Check logs above."
    exit 1
  fi
done

info "Starting Tauri app …"
(
  cd "$FRONTEND"
  . "$HOME/.cargo/env"
  npm run tauri dev
) &
APP_PID=$!

wait "$APP_PID" || true
wait "$WEB_PID" || true
