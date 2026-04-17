from __future__ import annotations

import asyncio
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel, Field

from src.agent import build_executor

_ROOT = Path(__file__).resolve().parent.parent
_STATIC = _ROOT / "static"

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


@asynccontextmanager
async def _lifespan(app: FastAPI):
    load_dotenv(_ROOT / ".env")
    if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
        raise RuntimeError(
            "Set GOOGLE_API_KEY or GEMINI_API_KEY in .env (required for RAG embeddings)."
        )
    try:
        executor = build_executor()
    except (FileNotFoundError, ValueError) as exc:
        raise RuntimeError(str(exc)) from exc
    app.state.executor = executor
    app.state.sessions: dict[str, list[BaseMessage]] = {}
    app.state.lock = asyncio.Lock()
    yield


app = FastAPI(lifespan=_lifespan)


@app.get("/")
async def index() -> FileResponse:
    path = _STATIC / "index.html"
    if not path.is_file():
        raise HTTPException(status_code=500, detail="Missing static/index.html")
    return FileResponse(path)


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    msg = req.message.strip()
    if not msg:
        raise HTTPException(status_code=400, detail="Empty message")

    lock: asyncio.Lock = app.state.lock
    sessions: dict[str, list[BaseMessage]] = app.state.sessions
    executor = app.state.executor

    async with lock:
        sid: str
        if req.session_id and req.session_id in sessions:
            sid = req.session_id
        else:
            sid = str(uuid.uuid4())
            sessions[sid] = []

        hist = sessions[sid]
        history_for_prompt = list(hist)

        try:
            result = await asyncio.to_thread(
                executor.invoke,
                {"input": msg, "chat_history": history_for_prompt},
            )
        except Exception as exc:
            return ChatResponse(reply="", session_id=sid, error=str(exc))

        out = result.get("output")
        if out is None or str(out).strip() == "":
            reply = ""
            err = f"No text output; full result: {result!r}"
        else:
            reply = str(out).strip()
            err = None

        hist.append(HumanMessage(content=msg))
        hist.append(AIMessage(content=reply if reply else err or ""))
        _trim_session(hist)

        return ChatResponse(reply=reply, session_id=sid, error=err)


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
