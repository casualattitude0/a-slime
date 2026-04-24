# Agent

Local AI assistant with **RAG** (Chroma + LangChain), tool use, and a **Vue 3** chat UI backed by **FastAPI**. Supports **Google Gemini**, **Ollama**, and optional **NVIDIA NIM** chat models. Optional **Tauri** desktop shell (`AgentChat`).

## Requirements

- Python 3.11+ (recommended)
- Node.js 20+ (for the frontend)
- A Google API key with Gemini access (`GOOGLE_API_KEY` or `GEMINI_API_KEY`) for embeddings and (unless using Ollama-only flows) chat
- [Ollama](https://ollama.com/) if you set `OLLAMA_MODEL`

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env — at minimum set GOOGLE_API_KEY (or GEMINI_API_KEY).
```

Optional Google Calendar tools: install `google-auth`, `google-auth-oauthlib`, `google-auth-httplib2`, then run `python scripts/get_google_oauth_token.py` (see comments in `.env.example`). Do not commit `client_secret*.json`, `google_oauth_token.json`, or other OAuth material.

## RAG ingest

Default source directory is `data/`; vectors go to `chroma_db/`.

```bash
python src/ingest.py
# Or: python src/ingest.py --data-dir /path/to/docs --chroma-dir ./chroma_db
```

If you change `GEMINI_EMBEDDING_MODEL`, remove `chroma_db` and re-ingest.

## Run

### CLI (REPL)

```bash
python main.py
```

### Web UI (recommended)

Terminal 1 — API + static `frontend/dist` when present:

```bash
python src/web_server.py
```

Default bind: `WEB_HOST=127.0.0.1`, `WEB_PORT=8765`.

Terminal 2 — Vite dev server (proxies `/api` to `8765`):

```bash
cd frontend && npm install && npm run dev
```

Open `http://localhost:5173`. For a self-contained server without Vite, build the frontend first:

```bash
cd frontend && npm run build && cd ..
python src/web_server.py
```

Then open `http://127.0.0.1:8765` (or your `WEB_HOST`/`WEB_PORT`).

### Tauri (desktop)

From `frontend/` after backend is running (or adjust dev URL as needed):

```bash
cd frontend && npm run tauri dev
```

## Configuration

See `.env.example` for:

- `OLLAMA_MODEL` / `OLLAMA_BASE_URL` — local chat instead of Gemini when set
- `GEMINI_MODEL`, `GEMINI_EMBEDDING_MODEL`, `GEMINI_REASONING_MODEL`
- `NVIDIA_API_KEY`, `NVIDIA_MODEL` — NIM / API Catalog
- `AGENT_MAX_ITERATIONS` and NVIDIA-specific toggles
- Google Calendar env vars and token cache paths

## Tests

```bash
pytest
```

## Layout

| Path | Role |
|------|------|
| `main.py` | CLI entry |
| `src/web_server.py` | FastAPI app, WebSockets, static UI |
| `src/agent/` | LangChain agent, routing, tools, RAG |
| `src/ingest.py` | PDF/txt/md → Chroma |
| `src/tools.py` | Tool implementations (search, calendar, memory, etc.) |
| `frontend/` | Vue + Vite + Tauri |
| `data/` | Default RAG documents |
| `chroma_db/` | Chroma persistence (gitignored) |
