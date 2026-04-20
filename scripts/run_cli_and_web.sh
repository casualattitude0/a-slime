#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -n "${PYTHON:-}" ]]; then
  :
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON="$ROOT/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
elif command -v python >/dev/null 2>&1; then
  PYTHON=python
else
  echo "找不到 python：請安裝 Python 3，或設定 PYTHON、或建立 .venv。" >&2
  exit 1
fi

cleanup() {
  kill "${WEB_PID:-}" "${CLI_PID:-}" 2>/dev/null || true
}
trap cleanup INT TERM

"$PYTHON" main.py &
CLI_PID=$!
"$PYTHON" -m src.web_server &
WEB_PID=$!

wait "${CLI_PID}" || true
wait "${WEB_PID}" || true
