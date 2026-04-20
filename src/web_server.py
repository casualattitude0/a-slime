from __future__ import annotations

import asyncio
import ast
import json
import os
import re
import threading
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pydantic import BaseModel, Field

from src.agent import (
    LLMErrorInfo,
    astream_executor,
    build_executor,
    classify_llm_error,
    delete_all_rag_items,
    delete_rag_item,
    invoke_executor,
    list_rag_items,
    local_quick_reply,
    make_gemini_llm,
    make_ollama_llm,
    normalize_agent_output,
    should_escalate_to_gemini,
)
from src.chat_registry import ChatRegistry
from src.history_store import (
    append_message,
    clear_session_messages,
    get_session_messages_as_dicts,
    load_session_messages,
    trim_session_messages,
)
from src.tools import (
    delete_all_memory_items,
    delete_memory_item,
    list_memory_items,
)
from src.version_registry import VersionRegistry

_ROOT = Path(__file__).resolve().parent.parent
_STATIC = _ROOT / "frontend" / "dist"
_REGISTRY_PATH = _ROOT / "version_registry.json"
_CHAT_REGISTRY_PATH = _ROOT / "chat_registry.json"

_MAX_SESSION_MESSAGES = 40


def _trim_session(msgs: list[BaseMessage]) -> None:
    if len(msgs) <= _MAX_SESSION_MESSAGES:
        return
    del msgs[: len(msgs) - _MAX_SESSION_MESSAGES]


def _persist_message(app: FastAPI, sid: str, msg: BaseMessage, version_id: str | None = None) -> None:
    """Append a message to the Chroma history store asynchronously-safe (sync call)."""
    try:
        append_message(
            app.state.chroma_dir,
            app.state.embeddings,
            sid,
            msg,
            version_id=version_id,
        )
    except Exception:
        pass


def _persist_trim(app: FastAPI, sid: str) -> None:
    try:
        trim_session_messages(
            app.state.chroma_dir,
            app.state.embeddings,
            sid,
            _MAX_SESSION_MESSAGES,
        )
    except Exception:
        pass


def _load_session_from_store(app: FastAPI, sid: str) -> list[BaseMessage]:
    """Load persisted messages from Chroma for a given session_id."""
    try:
        return load_session_messages(app.state.chroma_dir, app.state.embeddings, sid)
    except Exception:
        return []


def _safe_eval_math(expr: str) -> str | None:
    s = (expr or "").strip()
    if not s or len(s) > 80:
        return None
    if not re.fullmatch(r"[0-9\.\s\+\-\*\/\%\(\)]+", s):
        return None
    try:
        node = ast.parse(s, mode="eval")
    except Exception:
        return None

    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Constant,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Mod,
        ast.USub,
        ast.UAdd,
        ast.Pow,
        ast.FloorDiv,
    )
    if any(not isinstance(n, allowed_nodes) for n in ast.walk(node)):
        return None

    def _eval(n: ast.AST) -> float:
        if isinstance(n, ast.Expression):
            return _eval(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
            v = _eval(n.operand)
            return v if isinstance(n.op, ast.UAdd) else -v
        if isinstance(n, ast.BinOp):
            l = _eval(n.left)
            r = _eval(n.right)
            if isinstance(n.op, ast.Add):
                return l + r
            if isinstance(n.op, ast.Sub):
                return l - r
            if isinstance(n.op, ast.Mult):
                return l * r
            if isinstance(n.op, ast.Div):
                return l / r
            if isinstance(n.op, ast.Mod):
                return l % r
            if isinstance(n.op, ast.FloorDiv):
                return l // r
            if isinstance(n.op, ast.Pow):
                return l**r
        raise ValueError("Unsupported expression")

    try:
        out = _eval(node)
    except Exception:
        return None
    if float(out).is_integer():
        return str(int(out))
    return str(out)


def _simple_local_reply(message: str) -> str | None:
    msg = (message or "").strip()
    if not msg:
        return None

    math_result = _safe_eval_math(msg)
    if math_result is not None:
        return math_result

    low = msg.lower()
    if low in {"hi", "hello", "hey", "嗨", "你好", "哈囉"}:
        return "你好"
    if low in {"thanks", "thank you", "謝謝", "感謝"}:
        return "不客氣"
    if low in {"bye", "掰掰", "再見"}:
        return "再見"
    return None


# ─── Request / Response models ────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(default="")
    session_id: str | None = None


class ChatResponse(BaseModel):
    reply: str = ""
    session_id: str = ""
    error: str | None = None
    llm_error: dict | None = None


class ClearRequest(BaseModel):
    session_id: str | None = None


class TerminateRequest(BaseModel):
    session_id: str


class VersionSwitchRequest(BaseModel):
    version_id: str


class VersionCreateRequest(BaseModel):
    name: str
    model_profile: str = "default"


class DeleteAllRequest(BaseModel):
    confirm_token: str


# ─── Executor helpers ─────────────────────────────────────────────────────────

def _available_model_profiles() -> list[str]:
    profiles = ["default"]
    if (os.environ.get("GEMINI_MODEL") or "").strip():
        profiles.append("gemini")
    if (os.environ.get("OLLAMA_MODEL") or "").strip():
        profiles.append("ollama")
    if (os.environ.get("GEMINI_PRO_MODEL") or "").strip():
        profiles.append("gemini-pro")
    return profiles


def _build_executor_for_profile(
    chroma_dir: Path,
    memory_collection: str,
    rag_collection: str,
    model_profile: str,
) -> Any:
    if model_profile == "ollama":
        llm = make_ollama_llm()
    elif model_profile == "gemini-pro":
        pro = (os.environ.get("GEMINI_PRO_MODEL") or "").strip() or "gemini-2.5-pro"
        llm = make_gemini_llm(model=pro)
    elif model_profile == "gemini":
        llm = make_gemini_llm()
    else:
        llm = make_gemini_llm()

    return build_executor(
        chroma_dir=chroma_dir,
        llm=llm,
        memory_collection=memory_collection,
        rag_collection=rag_collection,
    )


def _get_executor(app: FastAPI, version_id: str, user_message: str, history_len: int) -> Any:
    registry: VersionRegistry = app.state.version_registry
    version = registry.get(version_id) or registry.get_active()

    cache: dict[str, Any] = app.state.executor_cache
    cache_key = f"{version.version_id}:{version.model_profile}"

    if cache_key not in cache:
        cache[cache_key] = _build_executor_for_profile(
            app.state.chroma_dir,
            version.memory_collection,
            version.rag_collection,
            version.model_profile,
        )

    executor = cache[cache_key]

    if version.model_profile == "default":
        ollama_ex = getattr(app.state, "executor_ollama", None)
        gemini_ex = getattr(app.state, "executor_gemini", None)
        if ollama_ex is not None and should_escalate_to_gemini(user_message, history_len):
            return gemini_ex
        if ollama_ex is not None:
            return ollama_ex
        return gemini_ex

    return executor


# ─── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def _lifespan(app: FastAPI):
    load_dotenv(_ROOT / ".env")
    if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
        raise RuntimeError(
            "Set GOOGLE_API_KEY or GEMINI_API_KEY in .env (required for RAG embeddings)."
        )

    embed_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    embed_model = os.environ.get("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
    app.state.embeddings = GoogleGenerativeAIEmbeddings(
        model=embed_model,
        google_api_key=embed_key,
    )
    app.state.chroma_dir = (_ROOT / "chroma_db").resolve()
    app.state.chroma_dir.mkdir(parents=True, exist_ok=True)

    app.state.version_registry = VersionRegistry(_REGISTRY_PATH)
    app.state.chat_registry = ChatRegistry(_CHAT_REGISTRY_PATH)
    app.state.executor_cache: dict[str, Any] = {}
    app.state.available_profiles = _available_model_profiles()

    try:
        app.state.executor_gemini = build_executor(
            chroma_dir=app.state.chroma_dir,
            llm=make_gemini_llm(),
        )
        if (os.environ.get("OLLAMA_MODEL") or "").strip():
            app.state.executor_ollama = build_executor(
                chroma_dir=app.state.chroma_dir,
                llm=make_ollama_llm(),
            )
        else:
            app.state.executor_ollama = None
        app.state.executor = app.state.executor_ollama or app.state.executor_gemini
    except (FileNotFoundError, ValueError) as exc:
        raise RuntimeError(str(exc)) from exc

    app.state.sessions: dict[str, list[BaseMessage]] = {}
    app.state.cancel_events: dict[str, threading.Event] = {}
    app.state.lock = asyncio.Lock()

    # Restore persisted sessions from Chroma and adopt into chat registry.
    registry_for_restore: VersionRegistry = app.state.version_registry
    chat_reg: ChatRegistry = app.state.chat_registry
    for vd in registry_for_restore.list_versions():
        vsid = vd.get("session_id")
        vid = vd.get("version_id", "")
        if vsid:
            if vsid not in app.state.sessions:
                msgs = load_session_messages(app.state.chroma_dir, app.state.embeddings, vsid)
                if msgs:
                    app.state.sessions[vsid] = msgs
            # Ensure chat registry has an entry for this session.
            chat_reg.ensure(vsid, vid)

    yield


app = FastAPI(lifespan=_lifespan)

assets_path = _STATIC / "assets"
if assets_path.is_dir():
    app.mount("/assets", StaticFiles(directory=str(assets_path)), name="assets")


@app.get("/")
async def index() -> FileResponse:
    path = _STATIC / "index.html"
    if not path.is_file():
        raise HTTPException(
            status_code=500,
            detail="Missing frontend/dist/index.html. Did you run npm run build?",
        )
    return FileResponse(path)


# ─── Chat endpoints (original, backward compatible) ───────────────────────────

@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    msg = req.message.strip()
    if not msg:
        raise HTTPException(status_code=400, detail="Empty message")

    lock: asyncio.Lock = app.state.lock
    sessions: dict[str, list[BaseMessage]] = app.state.sessions
    registry: VersionRegistry = app.state.version_registry

    async with lock:
        version = registry.get_active()
        chat_reg_c: ChatRegistry = app.state.chat_registry
        sid: str
        if req.session_id and req.session_id in sessions:
            sid = req.session_id
        elif version.session_id and version.session_id in sessions:
            sid = version.session_id
        else:
            if version.session_id:
                sid = version.session_id
                restored = _load_session_from_store(app, sid)
            else:
                sid = str(uuid.uuid4())
                restored = []
            sessions[sid] = restored
            registry.update_session(version.version_id, sid)
            chat_reg_c.ensure(sid, version.version_id)

        hist = sessions[sid]
        local_reply = _simple_local_reply(msg)
        if local_reply is None:
            local_reply = await asyncio.to_thread(local_quick_reply, msg, hist)
        if local_reply is not None:
            reply = local_reply
            err = None
        else:
            history_for_prompt = list(hist)
            executor = _get_executor(app, version.version_id, msg, len(hist))

            llm_err_info: LLMErrorInfo | None = None
            try:
                result = await asyncio.to_thread(
                    invoke_executor,
                    executor,
                    {"input": msg, "chat_history": history_for_prompt},
                )
            except Exception as exc:
                llm_err_info = classify_llm_error(exc)
                if llm_err_info.is_llm_error:
                    human_msg = HumanMessage(content=msg)
                    ai_msg = AIMessage(content="")
                    hist.append(human_msg)
                    hist.append(ai_msg)
                    _trim_session(hist)
                    await asyncio.to_thread(_persist_message, app, sid, human_msg, version.version_id)
                    await asyncio.to_thread(_persist_message, app, sid, ai_msg, version.version_id)
                    await asyncio.to_thread(_persist_trim, app, sid)
                    return ChatResponse(
                        reply="",
                        session_id=sid,
                        error=llm_err_info.message,
                        llm_error=llm_err_info.to_dict(),
                    )
                return ChatResponse(reply="", session_id=sid, error=str(exc))

            out = result.get("output")
            reply = normalize_agent_output(out)
            err = None if reply else f"No text output; full result: {result!r}"

        human_msg = HumanMessage(content=msg)
        ai_msg = AIMessage(content=reply if reply else err or "")
        hist.append(human_msg)
        hist.append(ai_msg)
        _trim_session(hist)
        await asyncio.to_thread(_persist_message, app, sid, human_msg, version.version_id)
        await asyncio.to_thread(_persist_message, app, sid, ai_msg, version.version_id)
        await asyncio.to_thread(_persist_trim, app, sid)
        # Auto-title from first user message.
        await asyncio.to_thread(chat_reg_c.set_title_if_default, sid, msg)
        await asyncio.to_thread(chat_reg_c.touch, sid)

        return ChatResponse(reply=reply, session_id=sid, error=err)


async def _resolve_session(
    app: FastAPI,
    req_session_id: str | None,
    msg: str,
) -> tuple[str, list[BaseMessage], Any, dict[str, Any] | None, threading.Event]:
    """Resolve (or create) a session and return the objects needed for streaming.

    Returns (sid, hist_snapshot, executor, payload, cancel_event).
    ``payload`` is None and ``executor`` is None when a local reply covers the request;
    in that case ``hist_snapshot`` is the ready-to-use local reply stored as a string
    under the key ``"local_reply"`` of the returned payload dict.
    """
    lock: asyncio.Lock = app.state.lock
    sessions: dict[str, list[BaseMessage]] = app.state.sessions
    cancel_events: dict[str, threading.Event] = app.state.cancel_events
    registry: VersionRegistry = app.state.version_registry

    async with lock:
        version = registry.get_active()
        chat_reg_r: ChatRegistry = app.state.chat_registry
        if req_session_id and req_session_id in sessions:
            sid = req_session_id
        elif req_session_id and not (req_session_id in sessions):
            # Client passed a known chat_id that isn't loaded yet — restore it.
            sid = req_session_id
            sessions[sid] = _load_session_from_store(app, sid)
            chat_reg_r.ensure(sid, version.version_id)
        elif version.session_id and version.session_id in sessions:
            sid = version.session_id
        else:
            if version.session_id:
                sid = version.session_id
                restored = _load_session_from_store(app, sid)
            else:
                sid = str(uuid.uuid4())
                restored = []
            sessions[sid] = restored
            registry.update_session(version.version_id, sid)
            chat_reg_r.ensure(sid, version.version_id)

        hist = sessions[sid]

        cancel_event = cancel_events.get(sid)
        if cancel_event is None:
            cancel_event = threading.Event()
            cancel_events[sid] = cancel_event
        else:
            cancel_event.clear()

        local_reply = _simple_local_reply(msg)
        if local_reply is None:
            local_reply = await asyncio.to_thread(local_quick_reply, msg, hist)

        if local_reply is not None:
            return sid, list(hist), None, {"local_reply": local_reply}, cancel_event

        history_for_prompt = list(hist)
        executor = _get_executor(app, version.version_id, msg, len(hist))
        payload = {"input": msg, "chat_history": history_for_prompt}
        return sid, history_for_prompt, executor, payload, cancel_event


async def _stream_pipeline(
    app: FastAPI,
    sid: str,
    msg: str,
    executor: Any,
    payload: dict[str, Any],
    cancel_event: threading.Event,
) -> AsyncIterator[dict[str, Any]]:
    """Shared async generator yielding client-visible event dicts.

    Event shapes:
      {"event": "start",  "session_id": sid, "request_id": rid}
      {"event": "status", "phase": str, "label": str}
      {"event": "delta",  "text": str}
      {"event": "done",   "reply": str, "session_id": sid,
                          "terminated": bool, "error": str|None, "llm_error": dict|None}
    """
    lock: asyncio.Lock = app.state.lock
    sessions: dict[str, list[BaseMessage]] = app.state.sessions
    request_id = str(uuid.uuid4())
    yield {"event": "start", "session_id": sid, "request_id": request_id}
    yield {"event": "status", "phase": "route_deciding", "label": "路由決策中"}

    reply = ""
    err: str | None = None
    llm_error_payload: dict | None = None
    terminated = False

    async for ev in astream_executor(
        executor,
        payload,
        cancel_check=cancel_event.is_set,
    ):
        ev_name = ev.get("event", "")

        if ev_name == "_cancelled":
            terminated = True
            break

        if ev_name == "_error":
            exc = ev["exc"]
            err_info = classify_llm_error(exc)
            if err_info.is_llm_error:
                err = err_info.message
                llm_error_payload = err_info.to_dict()
            else:
                err = str(exc)
            break

        if ev_name == "_done":
            raw_output = ev.get("output") or {}
            out = raw_output.get("output") if isinstance(raw_output, dict) else None
            reply = normalize_agent_output(out) if out is not None else reply
            if not reply:
                err = f"No text output; full result: {raw_output!r}"
            break

        if ev_name == "delta":
            reply += ev.get("text", "")

        yield ev  # forward status and delta to client

    registry: VersionRegistry = app.state.version_registry
    chat_reg_p: ChatRegistry = app.state.chat_registry
    version_id_for_history = (registry.get_active() or registry.get_active()).version_id

    yield {"event": "status", "phase": "history_persisting", "label": "儲存對話紀錄"}

    async with lock:
        hist2 = sessions.get(sid, [])
        human_msg = HumanMessage(content=msg)
        hist2.append(human_msg)
        await asyncio.to_thread(_persist_message, app, sid, human_msg, version_id_for_history)
        if not terminated:
            ai_msg = AIMessage(content=reply if reply else err or "")
            hist2.append(ai_msg)
            await asyncio.to_thread(_persist_message, app, sid, ai_msg, version_id_for_history)
        _trim_session(hist2)
        await asyncio.to_thread(_persist_trim, app, sid)
        await asyncio.to_thread(chat_reg_p.set_title_if_default, sid, msg)
        await asyncio.to_thread(chat_reg_p.touch, sid)

    yield {
        "event": "done",
        "reply": reply,
        "session_id": sid,
        "terminated": terminated,
        "error": err,
        "llm_error": llm_error_payload,
    }


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    msg = req.message.strip()
    if not msg:
        raise HTTPException(status_code=400, detail="Empty message")

    sid, _hist, executor, payload, cancel_event = await _resolve_session(
        app, req.session_id, msg
    )

    # Fast local reply path.
    if executor is None:
        local_reply: str = payload["local_reply"]  # type: ignore[index]

        async def _local_gen():
            lock: asyncio.Lock = app.state.lock
            sessions: dict[str, list[BaseMessage]] = app.state.sessions
            registry_lr: VersionRegistry = app.state.version_registry
            vid_lr = registry_lr.get_active().version_id
            async with lock:
                hist2 = sessions.get(sid, [])
                human_m = HumanMessage(content=msg)
                ai_m = AIMessage(content=local_reply)
                hist2.append(human_m)
                hist2.append(ai_m)
                _trim_session(hist2)
                await asyncio.to_thread(_persist_message, app, sid, human_m, vid_lr)
                await asyncio.to_thread(_persist_message, app, sid, ai_m, vid_lr)
                await asyncio.to_thread(_persist_trim, app, sid)
            yield (
                json.dumps(
                    {
                        "event": "done",
                        "reply": local_reply,
                        "session_id": sid,
                        "terminated": False,
                        "error": None,
                        "llm_error": None,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

        return StreamingResponse(_local_gen(), media_type="application/x-ndjson")

    async def _ndjson_gen():
        async for ev in _stream_pipeline(app, sid, msg, executor, payload, cancel_event):
            yield json.dumps(ev, ensure_ascii=False) + "\n"

    return StreamingResponse(_ndjson_gen(), media_type="application/x-ndjson")


@app.websocket("/ws/chat/live")
async def ws_chat_live(websocket: WebSocket):
    """WebSocket live chat endpoint.

    Client sends: ``{"message": str, "session_id": str|null}``
    Server sends JSON frames with the same event schema as /api/chat/stream.
    """
    await websocket.accept()
    try:
        while True:
            try:
                raw = await websocket.receive_text()
            except WebSocketDisconnect:
                break

            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_text(
                    json.dumps({"event": "error", "error": "Invalid JSON"}, ensure_ascii=False)
                )
                continue

            msg: str = (data.get("message") or "").strip()
            if not msg:
                await websocket.send_text(
                    json.dumps(
                        {"event": "error", "error": "Empty message"}, ensure_ascii=False
                    )
                )
                continue

            req_session_id: str | None = data.get("session_id") or None

            try:
                sid, _hist, executor, payload, cancel_event = await _resolve_session(
                    app, req_session_id, msg
                )
            except Exception as exc:
                await websocket.send_text(
                    json.dumps({"event": "error", "error": str(exc)}, ensure_ascii=False)
                )
                continue

            # Fast local reply path.
            if executor is None:
                local_reply: str = payload["local_reply"]  # type: ignore[index]
                lock: asyncio.Lock = app.state.lock
                sessions: dict[str, list[BaseMessage]] = app.state.sessions
                registry_ws: VersionRegistry = app.state.version_registry
                vid_ws = registry_ws.get_active().version_id
                async with lock:
                    hist2 = sessions.get(sid, [])
                    human_mw = HumanMessage(content=msg)
                    ai_mw = AIMessage(content=local_reply)
                    hist2.append(human_mw)
                    hist2.append(ai_mw)
                    _trim_session(hist2)
                    await asyncio.to_thread(_persist_message, app, sid, human_mw, vid_ws)
                    await asyncio.to_thread(_persist_message, app, sid, ai_mw, vid_ws)
                    await asyncio.to_thread(_persist_trim, app, sid)
                await websocket.send_text(
                    json.dumps(
                        {
                            "event": "done",
                            "reply": local_reply,
                            "session_id": sid,
                            "terminated": False,
                            "error": None,
                            "llm_error": None,
                        },
                        ensure_ascii=False,
                    )
                )
                continue

            try:
                async for ev in _stream_pipeline(
                    app, sid, msg, executor, payload, cancel_event
                ):
                    await websocket.send_text(json.dumps(ev, ensure_ascii=False))
            except WebSocketDisconnect:
                break
            except Exception as exc:
                try:
                    await websocket.send_text(
                        json.dumps(
                            {"event": "error", "error": str(exc)}, ensure_ascii=False
                        )
                    )
                except Exception:
                    pass

    except WebSocketDisconnect:
        pass


@app.post("/api/chat/fallback", response_model=ChatResponse)
async def chat_fallback(req: ChatRequest) -> ChatResponse:
    """Immediate local/Ollama fallback used when the primary LLM has failed."""
    msg = req.message.strip()
    if not msg:
        raise HTTPException(status_code=400, detail="Empty message")

    lock: asyncio.Lock = app.state.lock
    sessions: dict[str, list[BaseMessage]] = app.state.sessions

    async with lock:
        sid = req.session_id or ""
        if not sid or sid not in sessions:
            raise HTTPException(status_code=400, detail="Invalid or missing session_id")
        hist = list(sessions[sid])

    ollama_ex = getattr(app.state, "executor_ollama", None)
    if ollama_ex is None:
        return ChatResponse(
            reply="本地模型（Ollama）不可用。請檢查 API 配額或帳單後重試。",
            session_id=sid,
        )

    try:
        result = await asyncio.to_thread(
            invoke_executor,
            ollama_ex,
            {"input": msg, "chat_history": hist},
        )
    except Exception as exc:
        return ChatResponse(reply="", session_id=sid, error=str(exc))

    out = result.get("output")
    reply = normalize_agent_output(out)
    err = None if reply else f"No text output; full result: {result!r}"

    async with lock:
        hist2 = sessions.get(sid, [])
        ai_msg_fb = AIMessage(content=reply if reply else err or "")
        hist2.append(ai_msg_fb)
        _trim_session(hist2)
        await asyncio.to_thread(_persist_message, app, sid, ai_msg_fb, None)
        await asyncio.to_thread(_persist_trim, app, sid)

    return ChatResponse(reply=reply, session_id=sid, error=err)


@app.post("/api/clear")
async def clear_session(req: ClearRequest) -> dict[str, bool | str | None]:
    lock: asyncio.Lock = app.state.lock
    sessions: dict[str, list[BaseMessage]] = app.state.sessions

    async with lock:
        sid = req.session_id
        if sid and sid in sessions:
            sessions[sid].clear()
        if sid:
            await asyncio.to_thread(
                clear_session_messages,
                app.state.chroma_dir,
                app.state.embeddings,
                sid,
            )
        return {"ok": True, "session_id": sid}


@app.post("/api/chat/terminate")
async def terminate_chat(req: TerminateRequest) -> dict[str, bool]:
    lock: asyncio.Lock = app.state.lock
    cancel_events: dict[str, threading.Event] = app.state.cancel_events
    async with lock:
        ev = cancel_events.get(req.session_id)
        if ev is not None:
            ev.set()
    return {"ok": True}


# ─── Version endpoints ────────────────────────────────────────────────────────

@app.get("/api/versions")
async def get_versions() -> dict[str, Any]:
    registry: VersionRegistry = app.state.version_registry
    return {
        "versions": registry.list_versions(),
        "active_version_id": registry.get_active_id(),
        "available_model_profiles": app.state.available_profiles,
    }


@app.post("/api/versions/switch")
async def switch_version(req: VersionSwitchRequest) -> dict[str, Any]:
    registry: VersionRegistry = app.state.version_registry
    ok = registry.switch(req.version_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Version '{req.version_id}' not found")
    version = registry.get_active()
    sessions: dict[str, list[BaseMessage]] = app.state.sessions
    if version.session_id and version.session_id not in sessions:
        sessions[version.session_id] = _load_session_from_store(app, version.session_id)
    return {
        "ok": True,
        "active_version": version.to_dict(),
        "session_id": version.session_id,
    }


@app.post("/api/versions/create")
async def create_version(req: VersionCreateRequest) -> dict[str, Any]:
    registry: VersionRegistry = app.state.version_registry
    if req.model_profile not in app.state.available_profiles:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model profile '{req.model_profile}'. Available: {app.state.available_profiles}",
        )
    version = registry.create(req.name, req.model_profile)
    return {"ok": True, "version": {**version.to_dict(), "is_active": False}}


@app.delete("/api/versions/{version_id}")
async def delete_version(version_id: str) -> dict[str, Any]:
    registry: VersionRegistry = app.state.version_registry
    cache: dict[str, Any] = app.state.executor_cache
    version = registry.get(version_id)
    if version is None:
        raise HTTPException(status_code=404, detail=f"Version '{version_id}' not found")
    for profile in app.state.available_profiles + ["default"]:
        cache.pop(f"{version_id}:{profile}", None)
    ok = registry.delete(version_id)
    if not ok:
        raise HTTPException(status_code=400, detail="Cannot delete the last remaining version")
    return {"ok": True, "active_version_id": registry.get_active_id()}


# ─── Chat history endpoints ───────────────────────────────────────────────────


class ChatRenamRequest(BaseModel):
    title: str


@app.get("/api/chats")
async def list_chats() -> dict[str, Any]:
    chat_reg: ChatRegistry = app.state.chat_registry
    return {"chats": chat_reg.list_all()}


@app.post("/api/chats")
async def create_chat(version_id: str | None = None) -> dict[str, Any]:
    registry: VersionRegistry = app.state.version_registry
    chat_reg: ChatRegistry = app.state.chat_registry
    sessions: dict[str, list[BaseMessage]] = app.state.sessions
    lock: asyncio.Lock = app.state.lock

    vid = version_id or registry.get_active().version_id
    entry = chat_reg.create(version_id=vid)
    async with lock:
        sessions[entry.chat_id] = []
    return entry.to_dict()


@app.patch("/api/chats/{chat_id}/title")
async def rename_chat(chat_id: str, req: ChatRenamRequest) -> dict[str, Any]:
    chat_reg: ChatRegistry = app.state.chat_registry
    ok = await asyncio.to_thread(chat_reg.rename, chat_id, req.title)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Chat '{chat_id}' not found")
    return {"ok": True}


@app.delete("/api/chats/{chat_id}")
async def delete_chat(chat_id: str) -> dict[str, Any]:
    chat_reg: ChatRegistry = app.state.chat_registry
    sessions: dict[str, list[BaseMessage]] = app.state.sessions
    lock: asyncio.Lock = app.state.lock

    ok = await asyncio.to_thread(chat_reg.delete, chat_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Chat '{chat_id}' not found")
    await asyncio.to_thread(
        clear_session_messages, app.state.chroma_dir, app.state.embeddings, chat_id
    )
    async with lock:
        sessions.pop(chat_id, None)
    return {"ok": True}


@app.get("/api/chats/{chat_id}/messages")
async def get_chat_messages(chat_id: str) -> dict[str, Any]:
    messages = await asyncio.to_thread(
        get_session_messages_as_dicts,
        app.state.chroma_dir,
        app.state.embeddings,
        chat_id,
    )
    # Also warm the in-memory session cache.
    sessions: dict[str, list[BaseMessage]] = app.state.sessions
    lock: asyncio.Lock = app.state.lock
    async with lock:
        if chat_id not in sessions:
            sessions[chat_id] = _load_session_from_store(app, chat_id)
    return {"chat_id": chat_id, "messages": messages}


# ─── Memory item endpoints ────────────────────────────────────────────────────

@app.get("/api/memory/items")
async def get_memory_items(version_id: str | None = None) -> dict[str, Any]:
    registry: VersionRegistry = app.state.version_registry
    version = (registry.get(version_id) if version_id else None) or registry.get_active()
    items = await asyncio.to_thread(
        list_memory_items,
        app.state.chroma_dir,
        app.state.embeddings,
        version.memory_collection,
    )
    return {"items": items, "collection": version.memory_collection}


@app.delete("/api/memory/items/{item_id}")
async def delete_memory_item_endpoint(
    item_id: str, version_id: str | None = None
) -> dict[str, Any]:
    registry: VersionRegistry = app.state.version_registry
    version = (registry.get(version_id) if version_id else None) or registry.get_active()
    ok = await asyncio.to_thread(
        delete_memory_item,
        app.state.chroma_dir,
        app.state.embeddings,
        item_id,
        version.memory_collection,
    )
    if not ok:
        raise HTTPException(status_code=404, detail=f"Memory item '{item_id}' not found or delete failed")
    return {"ok": True}


@app.delete("/api/memory/all")
async def delete_all_memory(version_id: str | None = None) -> dict[str, Any]:
    registry: VersionRegistry = app.state.version_registry
    version = (registry.get(version_id) if version_id else None) or registry.get_active()
    ok = await asyncio.to_thread(
        delete_all_memory_items,
        app.state.chroma_dir,
        app.state.embeddings,
        version.memory_collection,
    )
    return {"ok": ok}


# ─── RAG item endpoints ───────────────────────────────────────────────────────

@app.get("/api/rag/items")
async def get_rag_items(version_id: str | None = None) -> dict[str, Any]:
    registry: VersionRegistry = app.state.version_registry
    version = (registry.get(version_id) if version_id else None) or registry.get_active()
    items = await asyncio.to_thread(
        list_rag_items,
        app.state.chroma_dir,
        app.state.embeddings,
        version.rag_collection,
    )
    return {"items": items, "collection": version.rag_collection}


@app.delete("/api/rag/items/{item_id}")
async def delete_rag_item_endpoint(
    item_id: str, version_id: str | None = None
) -> dict[str, Any]:
    registry: VersionRegistry = app.state.version_registry
    version = (registry.get(version_id) if version_id else None) or registry.get_active()
    ok = await asyncio.to_thread(
        delete_rag_item,
        app.state.chroma_dir,
        app.state.embeddings,
        item_id,
        version.rag_collection,
    )
    if not ok:
        raise HTTPException(status_code=404, detail=f"RAG item '{item_id}' not found or delete failed")
    return {"ok": True}


@app.delete("/api/rag/all")
async def delete_all_rag(version_id: str | None = None) -> dict[str, Any]:
    registry: VersionRegistry = app.state.version_registry
    version = (registry.get(version_id) if version_id else None) or registry.get_active()
    ok = await asyncio.to_thread(
        delete_all_rag_items,
        app.state.chroma_dir,
        app.state.embeddings,
        version.rag_collection,
    )
    return {"ok": ok}


# ─── Global nuke ─────────────────────────────────────────────────────────────

@app.delete("/api/data/all")
async def delete_all_data(req: DeleteAllRequest) -> dict[str, Any]:
    if req.confirm_token != "DELETE_ALL":
        raise HTTPException(status_code=400, detail="confirm_token must be 'DELETE_ALL'")

    registry: VersionRegistry = app.state.version_registry

    async def _nuke_version(v: Any) -> None:
        await asyncio.to_thread(
            delete_all_memory_items,
            app.state.chroma_dir,
            app.state.embeddings,
            v.memory_collection,
        )
        await asyncio.to_thread(
            delete_all_rag_items,
            app.state.chroma_dir,
            app.state.embeddings,
            v.rag_collection,
        )

    versions_data = registry.list_versions()
    for vd in versions_data:
        v = registry.get(vd["version_id"])
        if v:
            await _nuke_version(v)

    # Clear all persisted session histories.
    for sid_nuke in list(app.state.sessions.keys()):
        try:
            await asyncio.to_thread(
                clear_session_messages,
                app.state.chroma_dir,
                app.state.embeddings,
                sid_nuke,
            )
        except Exception:
            pass

    app.state.sessions.clear()
    app.state.executor_cache.clear()
    return {"ok": True}


def main() -> None:
    import uvicorn

    host = os.environ.get("WEB_HOST", "127.0.0.1").strip() or "127.0.0.1"
    port = int(os.environ.get("WEB_PORT", "8765"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
