#!/usr/bin/env bash
# run_cli_and_web.sh — Setup and start both the CLI agent and web server.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; YELLOW='\033[1;33m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${CYAN}[info]${NC}  $*"; }
ok()    { echo -e "${GREEN}[ok]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[warn]${NC}  $*"; }
err()   { echo -e "${RED}[error]${NC} $*" >&2; }

# ── Resolve Python ────────────────────────────────────────────────────────────
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

# ── Virtualenv: create if missing ─────────────────────────────────────────────
if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  info "Creating .venv …"
  $PYTHON -m venv "$ROOT/.venv"
  PYTHON="$ROOT/.venv/bin/python"
  ok ".venv created"
fi

# ── Pip: install / sync requirements ──────────────────────────────────────────
if [[ -f "$ROOT/requirements.txt" ]]; then
  info "Checking Python dependencies …"
  "$ROOT/.venv/bin/pip" install -q --upgrade pip
  "$ROOT/.venv/bin/pip" install -q -r "$ROOT/requirements.txt"
  ok "Python dependencies up to date"
fi

# ── .env: warn if missing API key ─────────────────────────────────────────────
if [[ -f "$ROOT/.env" ]]; then
  # shellcheck source=/dev/null
  set -a; source "$ROOT/.env"; set +a
fi
if [[ -z "${GOOGLE_API_KEY:-}" && -z "${GEMINI_API_KEY:-}" ]]; then
  warn "GOOGLE_API_KEY / GEMINI_API_KEY not set — web server will refuse to start."
fi

# ── Ingest: build ChromaDB from data if needed ────────────────────────────────
DATA_DIR="${DATA_DIR:-$ROOT/data}"
CHROMA_DIR="${CHROMA_DIR:-$ROOT/chroma_db}"
INGEST_STAMP="$CHROMA_DIR/.ingest.stamp"

if [[ -d "$DATA_DIR" ]]; then
  NEEDS_INGEST=false
  if [[ ! -f "$INGEST_STAMP" ]]; then
    NEEDS_INGEST=true
  else
    while IFS= read -r -d '' f; do
      if [[ "$f" -nt "$INGEST_STAMP" ]]; then
        NEEDS_INGEST=true
        break
      fi
    done < <(find "$DATA_DIR" -type f \( -name "*.pdf" -o -name "*.txt" -o -name "*.md" \) -print0 2>/dev/null)
  fi

  if [[ "$NEEDS_INGEST" == true ]]; then
    if [[ -z "${GOOGLE_API_KEY:-}" && -z "${GEMINI_API_KEY:-}" ]]; then
      warn "Skipping ingest because GOOGLE_API_KEY / GEMINI_API_KEY is not set."
    else
      TMP_CHROMA_DIR="$(mktemp -d "$ROOT/.chroma_ingest_tmp.XXXXXX")"
      info "Building ChromaDB via src/ingest.py …"
      if "$PYTHON" "$ROOT/src/ingest.py" --data-dir "$DATA_DIR" --chroma-dir "$TMP_CHROMA_DIR"; then
        if [[ -d "$CHROMA_DIR" ]]; then
          rm -rf "$CHROMA_DIR"
        fi
        mv "$TMP_CHROMA_DIR" "$CHROMA_DIR"
        touch "$INGEST_STAMP"
        ok "Ingest complete → $CHROMA_DIR"
      else
        warn "Ingest failed; keeping existing ChromaDB unchanged."
        rm -rf "$TMP_CHROMA_DIR"
      fi
    fi
  else
    ok "Ingest is up to date"
  fi
else
  warn "Data directory not found: $DATA_DIR (skipping ingest)."
fi

# ── Node / npm: install and build frontend if needed ─────────────────────────
FRONTEND="$ROOT/frontend"
DIST="$FRONTEND/dist"

if [[ -f "$FRONTEND/package.json" ]]; then
  if ! command -v npm >/dev/null 2>&1; then
    warn "npm not found — skipping frontend build (serve existing dist if present)."
  else
    if [[ ! -d "$FRONTEND/node_modules" ]]; then
      info "Installing frontend npm packages …"
      npm --prefix "$FRONTEND" install --silent
      ok "npm install done"
    fi

    # Rebuild if any source file is newer than dist/index.html
    NEEDS_BUILD=false
    if [[ ! -f "$DIST/index.html" ]]; then
      NEEDS_BUILD=true
    else
      while IFS= read -r -d '' f; do
        if [[ "$f" -nt "$DIST/index.html" ]]; then
          NEEDS_BUILD=true
          break
        fi
      done < <(find "$FRONTEND/src" -type f -print0 2>/dev/null)
    fi

    if [[ "$NEEDS_BUILD" == true ]]; then
      info "Building frontend …"
      npm --prefix "$FRONTEND" run build
      ok "Frontend built → $DIST"
    else
      ok "Frontend dist is up to date"
    fi
  fi
fi

# ── Kill any stale instances ──────────────────────────────────────────────────
pkill -f "src.web_server" 2>/dev/null && info "Stopped previous web server" || true
pkill -f "main\.py" 2>/dev/null && info "Stopped previous CLI agent" || true
sleep 0.3

# ── Launch ────────────────────────────────────────────────────────────────────
WEB_HOST="${WEB_HOST:-127.0.0.1}"
WEB_PORT="${WEB_PORT:-8765}"

cleanup() {
  kill "${WEB_PID:-}" "${CLI_PID:-}" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

info "Starting CLI agent …"
"$PYTHON" main.py &
CLI_PID=$!

info "Starting web server on http://${WEB_HOST}:${WEB_PORT} …"
"$PYTHON" -m uvicorn src.web_server:app \
  --host "$WEB_HOST" \
  --port "$WEB_PORT" \
  --log-level info &
WEB_PID=$!

# Wait for web server to be ready (up to 15 s)
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

wait "${CLI_PID}" || true
wait "${WEB_PID}" || true
