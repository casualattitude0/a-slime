from __future__ import annotations

import asyncio
import json
import os
import queue
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel, Field

from src.agent import (
    build_executor,
    invoke_executor,
    make_gemini_llm,
    make_ollama_llm,
    normalize_agent_output,
    should_escalate_to_gemini,
)

_ROOT = Path(__file__).resolve().parent.parent
_STATIC = _ROOT / "frontend" / "dist"

_MAX_SESSION_MESSAGES = 40


def _trim_session(msgs: list[BaseMessage]) -> None:
    if len(msgs) <= _MAX_SESSION_MESSAGES:
        return
    del msgs[: len(msgs) - _MAX_SESSION_MESSAGES]


class ChatRequest(BaseModel):
    message: str = Field(default="")
    session_id: str | None = None


class ChatResponse(BaseModel):
    reply: str = ""
    session_id: str = ""
    error: str | None = None


class ClearRequest(BaseModel):
    session_id: str | None = None


def _select_executor(app: FastAPI, user_message: str, history_len: int) -> Any:
    ollama_ex = getattr(app.state, "executor_ollama", None)
    gemini_ex = getattr(app.state, "executor_gemini", None)
    if ollama_ex is not None and should_escalate_to_gemini(user_message, history_len):
        return gemini_ex
    if ollama_ex is not None:
        return ollama_ex
    return gemini_ex


@asynccontextmanager
async def _lifespan(app: FastAPI):
    load_dotenv(_ROOT / ".env")
    if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
        raise RuntimeError(
            "Set GOOGLE_API_KEY or GEMINI_API_KEY in .env (required for RAG embeddings)."
        )
    try:
        app.state.executor_gemini = build_executor(llm=make_gemini_llm())
        if (os.environ.get("OLLAMA_MODEL") or "").strip():
            app.state.executor_ollama = build_executor(llm=make_ollama_llm())
        else:
            app.state.executor_ollama = None
        app.state.executor = app.state.executor_ollama or app.state.executor_gemini
    except (FileNotFoundError, ValueError) as exc:
        raise RuntimeError(str(exc)) from exc
    app.state.sessions: dict[str, list[BaseMessage]] = {}
    app.state.lock = asyncio.Lock()
    yield


app = FastAPI(lifespan=_lifespan)

assets_path = _STATIC / "assets"
if assets_path.is_dir():
    app.mount("/assets", StaticFiles(directory=str(assets_path)), name="assets")

@app.get("/")
async def index() -> FileResponse:
    path = _STATIC / "index.html"
    if not path.is_file():
        raise HTTPException(status_code=500, detail="Missing frontend/dist/index.html. Did you run npm run build?")
    return FileResponse(path)


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    msg = req.message.strip()
    if not msg:
        raise HTTPException(status_code=400, detail="Empty message")

    lock: asyncio.Lock = app.state.lock
    sessions: dict[str, list[BaseMessage]] = app.state.sessions

    async with lock:
        sid: str
        if req.session_id and req.session_id in sessions:
            sid = req.session_id
        else:
            sid = str(uuid.uuid4())
            sessions[sid] = []

        hist = sessions[sid]
        history_for_prompt = list(hist)
        executor = _select_executor(app, msg, len(hist))

        try:
            result = await asyncio.to_thread(
                invoke_executor,
                executor,
                {"input": msg, "chat_history": history_for_prompt},
            )
        except Exception as exc:
            return ChatResponse(reply="", session_id=sid, error=str(exc))

        out = result.get("output")
        reply = normalize_agent_output(out)
        if not reply:
            err = f"No text output; full result: {result!r}"
        else:
            err = None

        hist.append(HumanMessage(content=msg))
        hist.append(AIMessage(content=reply if reply else err or ""))
        _trim_session(hist)

        return ChatResponse(reply=reply, session_id=sid, error=err)


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    msg = req.message.strip()
    if not msg:
        raise HTTPException(status_code=400, detail="Empty message")

    lock: asyncio.Lock = app.state.lock
    sessions: dict[str, list[BaseMessage]] = app.state.sessions

    async with lock:
        if req.session_id and req.session_id in sessions:
            sid = req.session_id
        else:
            sid = str(uuid.uuid4())
            sessions[sid] = []

        hist = sessions[sid]
        history_for_prompt = list(hist)
        executor = _select_executor(app, msg, len(hist))
        payload = {"input": msg, "chat_history": history_for_prompt}

    q: queue.Queue[tuple[str, Any]] = queue.Queue()
    box: dict[str, Any] = {}

    def worker() -> None:
        def sink(ev: dict[str, Any]) -> None:
            q.put(("status", ev))

        try:
            box["result"] = invoke_executor(executor, payload, status_sink=sink)
        except Exception as exc:
            box["error"] = exc
        finally:
            q.put(("finished", None))

    threading.Thread(target=worker, daemon=True).start()

    async def ndjson_gen():
        finished = False
        while not finished:
            await asyncio.sleep(0.02)
            try:
                while True:
                    kind, data = q.get_nowait()
                    if kind == "finished":
                        finished = True
                        break
                    if kind == "status" and isinstance(data, dict):
                        yield json.dumps(
                            {
                                "event": "status",
                                "phase": data.get("phase", ""),
                                "label": data.get("label", ""),
                            },
                            ensure_ascii=False,
                        ) + "\n"
            except queue.Empty:
                continue

        err: str | None = None
        reply = ""
        exc = box.get("error")
        if exc is not None:
            err = str(exc)
        else:
            result = box.get("result") or {}
            out = result.get("output")
            reply = normalize_agent_output(out)
            if not reply:
                err = f"No text output; full result: {result!r}"
            else:
                err = None

        async with lock:
            hist2 = sessions.get(sid, [])
            hist2.append(HumanMessage(content=msg))
            hist2.append(AIMessage(content=reply if reply else err or ""))
            _trim_session(hist2)

        yield json.dumps(
            {
                "event": "done",
                "reply": reply,
                "session_id": sid,
                "error": err,
            },
            ensure_ascii=False,
        ) + "\n"

    return StreamingResponse(ndjson_gen(), media_type="application/x-ndjson")


@app.post("/api/clear")
async def clear_session(req: ClearRequest) -> dict[str, bool | str | None]:
    lock: asyncio.Lock = app.state.lock
    sessions: dict[str, list[BaseMessage]] = app.state.sessions

    async with lock:
        sid = req.session_id
        if sid and sid in sessions:
            sessions[sid].clear()
            return {"ok": True, "session_id": sid}
        return {"ok": True, "session_id": sid}


def main() -> None:
    import uvicorn

    host = os.environ.get("WEB_HOST", "127.0.0.1").strip() or "127.0.0.1"
    port = int(os.environ.get("WEB_PORT", "8765"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
