from __future__ import annotations

import asyncio
import json
import os
import re
import threading
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pydantic import BaseModel, Field

from src.agent import (
    DEFAULT_NVIDIA_CATALOG_MODEL,
    LLMErrorInfo,
    _is_vague_short_utterance,
    _rich_exception_message,
    astream_executor,
    build_executor,
    classify_llm_error,
    delete_all_rag_items,
    delete_rag_item,
    invoke_executor,
    list_rag_items,
    agent_mode_reply,
    local_quick_reply,
    make_gemini_llm,
    make_nvidia_llm,
    make_ollama_llm,
    normalize_agent_output,
    should_escalate_to_gemini,
)
from src.chat_registry import ChatRegistry
from src.data_collection import (
    read_recent_agent_events,
    read_recent_feedback,
    write_agent_event,
    write_feedback,
)
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
    set_reminder_sink,
)
from src.version_registry import VersionRegistry

_ROOT = Path(__file__).resolve().parent.parent
_STATIC = _ROOT / "frontend" / "dist"
_REGISTRY_PATH = _ROOT / "version_registry.json"
_CHAT_REGISTRY_PATH = _ROOT / "chat_registry.json"

_MAX_SESSION_MESSAGES = 40
class ReminderNotification(BaseModel):
    notification_id: str
    chat_id: str
    source_session_id: str | None = None
    message: str
    created_at: str


def _parse_iso_dt(raw: str) -> datetime:
    s = (raw or "").strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


async def _deliver_reminder_event(app: FastAPI, payload: dict[str, Any]) -> None:
    lock: asyncio.Lock = app.state.lock
    sessions: dict[str, list[BaseMessage]] = app.state.sessions
    registry: VersionRegistry = app.state.version_registry
    chat_reg: ChatRegistry = app.state.chat_registry
    active_version = registry.get_active()
    title = (str(payload.get("title") or "").strip() or "提醒")
    reminder_text = (str(payload.get("message") or "").strip() or f"提醒：{title}")
    source_session_id = str(payload.get("source_session_id") or "").strip() or None

    entry = await asyncio.to_thread(
        chat_reg.create,
        active_version.version_id,
        f"提醒：{title}"[:60],
        None,
        {
            "chat_type": "reminder",
            "source_session_id": source_session_id or "",
            "reminder_at": str(payload.get("reminder_at") or ""),
        },
    )
    async with lock:
        sessions[entry.chat_id] = []
        user_msg = HumanMessage(content="系統提醒觸發")
        bot_msg = AIMessage(content=reminder_text)
        sessions[entry.chat_id].append(user_msg)
        sessions[entry.chat_id].append(bot_msg)
        await asyncio.to_thread(
            _persist_message, app, entry.chat_id, user_msg, active_version.version_id
        )
        await asyncio.to_thread(
            _persist_message, app, entry.chat_id, bot_msg, active_version.version_id
        )
        await asyncio.to_thread(_persist_trim, app, entry.chat_id)
        app.state.reminder_notifications.append(
            ReminderNotification(
                notification_id=str(uuid.uuid4()),
                chat_id=entry.chat_id,
                source_session_id=source_session_id,
                message=reminder_text,
                created_at=datetime.now(timezone.utc).isoformat(),
            ).model_dump()
        )


def _enqueue_reminder(app: FastAPI, payload: dict[str, Any]) -> None:
    reminder_at_raw = str(payload.get("reminder_at") or "").strip()
    if not reminder_at_raw:
        return
    try:
        reminder_at = _parse_iso_dt(reminder_at_raw)
    except Exception:
        return
    loop = getattr(app.state, "main_loop", None)
    if loop is None:
        return
    loop.call_soon_threadsafe(
        app.state.reminder_queue.put_nowait,
        {
            "reminder_at": reminder_at,
            "payload": payload,
        },
    )


async def _reminder_worker(app: FastAPI) -> None:
    while True:
        item = await app.state.reminder_queue.get()
        reminder_at: datetime = item["reminder_at"]
        payload: dict[str, Any] = item["payload"]
        now = datetime.now(timezone.utc)
        delay = (reminder_at - now).total_seconds()
        if delay > 0:
            await asyncio.sleep(delay)
        try:
            await _deliver_reminder_event(app, payload)
        except Exception:
            pass



def _trim_session(msgs: list[BaseMessage]) -> None:
    if len(msgs) <= _MAX_SESSION_MESSAGES:
        return
    del msgs[: len(msgs) - _MAX_SESSION_MESSAGES]


def _persist_message(
    app: FastAPI, sid: str, msg: BaseMessage, version_id: str | None = None
) -> str | None:
    """Append a message to the Chroma history store asynchronously-safe (sync call)."""
    try:
        return append_message(
            app.state.chroma_dir,
            app.state.embeddings,
            sid,
            msg,
            version_id=version_id,
        )
    except Exception:
        return None


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


def _record_agent_event(
    *,
    event_type: str,
    session_id: str,
    version_id: str | None = None,
    request_id: str | None = None,
    payload: dict[str, Any] | None = None,
) -> None:
    try:
        write_agent_event(
            _ROOT,
            event_type=event_type,
            session_id=session_id,
            version_id=version_id,
            request_id=request_id,
            payload=payload,
        )
    except Exception:
        pass


def _normalize_generated_chat_title(raw: str) -> str:
    title = (raw or "").strip()
    if not title:
        return ""
    title = title.splitlines()[0].strip()
    title = title.strip("`'\"")
    title = re.sub(r"^[#*\-\d\.\)\s]+", "", title).strip()
    title = re.sub(r"\s+", " ", title)
    return title[:60].strip()


def _generate_chat_title_with_second_agent(first_message: str) -> str | None:
    msg = (first_message or "").strip()
    if not msg:
        return None
    prompt = (
        "Generate one concise chat title from the user message.\n"
        "Rules:\n"
        "- Output title text only.\n"
        "- Single line.\n"
        "- <= 60 characters.\n"
        "- Keep original language when possible.\n\n"
        f"User message:\n{msg[:1000]}"
    )
    try:
        llm = make_gemini_llm()
        resp = llm.invoke(prompt)
        text = str(getattr(resp, "content", None) or resp)
        normalized = _normalize_generated_chat_title(text)
        return normalized or None
    except Exception:
        return None


async def _update_title_after_first_message(
    app: FastAPI,
    sid: str,
    first_message: str,
) -> str | None:
    chat_reg: ChatRegistry = app.state.chat_registry
    entry = await asyncio.to_thread(chat_reg.get, sid)
    if entry is None:
        return None
    if entry.title != "New Chat":
        return entry.title

    generated: str | None = None
    try:
        generated = await asyncio.wait_for(
            asyncio.to_thread(_generate_chat_title_with_second_agent, first_message),
            timeout=1.5,
        )
    except asyncio.TimeoutError:
        generated = None
    applied_title: str | None = None
    if generated:
        applied_title = await asyncio.to_thread(
            chat_reg.set_generated_title_if_default, sid, generated
        )
    if not applied_title:
        await asyncio.to_thread(chat_reg.set_title_if_default, sid, first_message)
        refreshed = await asyncio.to_thread(chat_reg.get, sid)
        applied_title = refreshed.title if refreshed else None
    return applied_title


# ─── Request / Response models ────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(default="")
    session_id: str | None = None
    llm_mode: Literal["auto", "gemini", "agent", "nvidia"] = "auto"
    system_instruction: str | None = None


class ChatResponse(BaseModel):
    reply: str = ""
    session_id: str = ""
    message_ref: str | None = None
    chat_title: str | None = None
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


class FeedbackRequest(BaseModel):
    session_id: str
    message_ref: str
    rating: int = Field(ge=1, le=5)
    comment: str | None = None


# ─── Executor helpers ─────────────────────────────────────────────────────────

def _available_model_profiles() -> list[str]:
    profiles = ["default"]
    if (os.environ.get("GEMINI_MODEL") or "").strip():
        profiles.append("gemini")
    if (os.environ.get("OLLAMA_MODEL") or "").strip():
        profiles.append("ollama")
    if (os.environ.get("GEMINI_PRO_MODEL") or "").strip():
        profiles.append("gemini-pro")
    if (os.environ.get("NVIDIA_API_KEY") or "").strip():
        profiles.append("nvidia")
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
    elif model_profile == "nvidia":
        llm = make_nvidia_llm()
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


def _get_executor_for_llm_mode(
    app: FastAPI,
    version_id: str,
    user_message: str,
    history_len: int,
    llm_mode: Literal["auto", "gemini", "agent", "nvidia"],
) -> Any:
    if llm_mode == "gemini":
        return getattr(app.state, "executor_gemini", None) or _get_executor(
            app, version_id, user_message, history_len
        )
    if llm_mode == "nvidia":
        return getattr(app.state, "executor_nvidia", None) or _get_executor(
            app, version_id, user_message, history_len
        )

    # Auto mode keeps inference on the agent's own path and avoids large-model escalation.
    ollama_ex = getattr(app.state, "executor_ollama", None)
    if ollama_ex is not None:
        return ollama_ex
    return _get_executor(app, version_id, user_message, history_len)


def _executor_model_label(app: FastAPI, executor: Any, model_profile: str) -> str:
    nv_ex = getattr(app.state, "executor_nvidia", None)
    if nv_ex is not None and executor is nv_ex:
        return "NVIDIA"
    if model_profile == "nvidia":
        return "NVIDIA"
    ollama_ex = getattr(app.state, "executor_ollama", None)
    if ollama_ex is not None and executor is ollama_ex:
        return "Ollama"
    return "Gemini"


def _nvidia_chat_model_display(fa: FastAPI) -> str | None:
    if getattr(fa.state, "executor_nvidia", None) is None:
        return None
    return (os.environ.get("NVIDIA_MODEL") or DEFAULT_NVIDIA_CATALOG_MODEL).strip()


# ─── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def _lifespan(app: FastAPI):
    load_dotenv(_ROOT / ".env")
    app.state.main_loop = asyncio.get_running_loop()
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
        app.state.executor_nvidia = None
        if (os.environ.get("NVIDIA_API_KEY") or "").strip():
            try:
                app.state.executor_nvidia = build_executor(
                    chroma_dir=app.state.chroma_dir,
                    llm=make_nvidia_llm(),
                )
            except ModuleNotFoundError:
                app.state.available_profiles = [
                    p for p in app.state.available_profiles if p != "nvidia"
                ]
                print(
                    "WARNING: NVIDIA_API_KEY is set but langchain-nvidia-ai-endpoints "
                    "is not installed. NVIDIA chat disabled. Run: pip install "
                    "langchain-nvidia-ai-endpoints",
                    flush=True,
                )
        app.state.executor = app.state.executor_ollama or app.state.executor_gemini
    except (FileNotFoundError, ValueError) as exc:
        raise RuntimeError(str(exc)) from exc

    app.state.sessions: dict[str, list[BaseMessage]] = {}
    app.state.cancel_events: dict[str, threading.Event] = {}
    app.state.lock = asyncio.Lock()
    app.state.reminder_queue = asyncio.Queue()
    app.state.reminder_notifications: list[dict[str, Any]] = []
    app.state.reminder_worker_task = asyncio.create_task(_reminder_worker(app))
    set_reminder_sink(lambda payload: _enqueue_reminder(app, payload))

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
    set_reminder_sink(None)
    task = getattr(app.state, "reminder_worker_task", None)
    if task:
        task.cancel()


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
        request_id = str(uuid.uuid4())
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
        _record_agent_event(
            event_type="chat_request",
            session_id=sid,
            version_id=version.version_id,
            request_id=request_id,
            payload={"input_text": msg, "llm_mode": req.llm_mode},
        )
        if req.llm_mode == "agent":
            local_reply = agent_mode_reply(msg, hist)
        elif req.llm_mode == "gemini":
            local_reply = None
        elif req.llm_mode == "nvidia":
            local_reply = None
        else:
            local_reply = await asyncio.to_thread(local_quick_reply, msg, list(hist))
            if local_reply is None and _is_vague_short_utterance(msg):
                local_reply = "我可能會誤解你的指涉內容。請補一句你是指哪個主題（例如：喝水建議、專案文件、或上一句回覆）。"
        used_local_reply = local_reply is not None
        if used_local_reply:
            reply = local_reply
            err = None
        else:
            history_for_prompt = list(hist)
            executor = _get_executor_for_llm_mode(
                app, version.version_id, msg, len(hist), req.llm_mode
            )

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
                    _record_agent_event(
                        event_type="chat_error",
                        session_id=sid,
                        version_id=version.version_id,
                        request_id=request_id,
                        payload={"error": llm_err_info.message, "route": "agent"},
                    )
                    return ChatResponse(
                        reply="",
                        session_id=sid,
                        message_ref=None,
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
        ai_message_ref = await asyncio.to_thread(_persist_message, app, sid, ai_msg, version.version_id)
        await asyncio.to_thread(_persist_trim, app, sid)
        # Auto-title from first user message.
        chat_title = await _update_title_after_first_message(app, sid, msg)

        _record_agent_event(
            event_type="chat_done" if not err else "chat_error",
            session_id=sid,
            version_id=version.version_id,
            request_id=request_id,
            payload={
                "reply_text": reply,
                "error": err,
                "message_ref": ai_message_ref or "",
                "route": "local" if used_local_reply else "agent",
            },
        )

        return ChatResponse(
            reply=reply,
            session_id=sid,
            message_ref=ai_message_ref,
            chat_title=chat_title,
            error=err,
        )


async def _resolve_session(
    app: FastAPI,
    req_session_id: str | None,
    msg: str,
    llm_mode: Literal["auto", "gemini", "agent", "nvidia"] = "auto",
    system_instruction: str | None = None,
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

        if llm_mode == "agent":
            local_reply = agent_mode_reply(msg, hist)
        elif llm_mode == "gemini":
            local_reply = None
        elif llm_mode == "nvidia":
            local_reply = None
        else:
            local_reply = await asyncio.to_thread(local_quick_reply, msg, list(hist))
            if local_reply is None and _is_vague_short_utterance(msg):
                local_reply = "我可能會誤解你的指涉內容。請補一句你是指哪個主題（例如：喝水建議、專案文件、或上一句回覆）。"

        if local_reply is not None:
            return (
                sid,
                list(hist),
                None,
                {
                    "local_reply": local_reply,
                    "version_id": version.version_id,
                },
                cancel_event,
            )

        history_for_prompt = list(hist)
        effective_input = msg
        if system_instruction:
            effective_input = f"{system_instruction.strip()}\n\n{msg}"
        executor = _get_executor_for_llm_mode(
            app, version.version_id, msg, len(hist), llm_mode
        )
        payload = {
            "input": effective_input,
            "chat_history": history_for_prompt,
            "version_id": version.version_id,
            "llm_model_label": _executor_model_label(app, executor, version.model_profile),
        }
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
    version_id_for_events = str(payload.get("version_id") or "")
    _record_agent_event(
        event_type="chat_request",
        session_id=sid,
        version_id=version_id_for_events,
        request_id=request_id,
        payload={"input_text": msg, "transport": "stream"},
    )
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
            _record_agent_event(
                event_type="chat_terminated",
                session_id=sid,
                version_id=version_id_for_events,
                request_id=request_id,
                payload={},
            )
            break

        if ev_name == "_error":
            exc = ev["exc"]
            err_info = classify_llm_error(exc)
            if err_info.is_llm_error:
                err = err_info.message
                llm_error_payload = err_info.to_dict()
            else:
                err = _rich_exception_message(exc)
            _record_agent_event(
                event_type="chat_error",
                session_id=sid,
                version_id=version_id_for_events,
                request_id=request_id,
                payload={"error": err},
            )
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
            _record_agent_event(
                event_type="stream_delta",
                session_id=sid,
                version_id=version_id_for_events,
                request_id=request_id,
                payload={"text": ev.get("text", "")},
            )

        if ev_name == "status":
            phase = ev.get("phase", "")
            label = ev.get("label", "")
            if phase == "llm_requesting":
                phase = "llm_requesting_model"
                model_label = str(payload.get("llm_model_label") or "LLM")
                label = f"正在與 {model_label} 溝通"
                ev["phase"] = phase
                ev["label"] = label
            _record_agent_event(
                event_type="status",
                session_id=sid,
                version_id=version_id_for_events,
                request_id=request_id,
                payload={"phase": phase, "label": label},
            )

        yield ev  # forward status and delta to client

    registry: VersionRegistry = app.state.version_registry
    version_id_for_history = version_id_for_events or registry.get_active().version_id

    yield {"event": "status", "phase": "history_persisting", "label": "儲存對話紀錄"}

    chat_title: str | None = None
    async with lock:
        hist2 = sessions.get(sid, [])
        human_msg = HumanMessage(content=msg)
        hist2.append(human_msg)
        await asyncio.to_thread(_persist_message, app, sid, human_msg, version_id_for_history)
        ai_message_ref = None
        if not terminated:
            ai_msg = AIMessage(content=reply if reply else err or "")
            hist2.append(ai_msg)
            ai_message_ref = await asyncio.to_thread(
                _persist_message, app, sid, ai_msg, version_id_for_history
            )
        _trim_session(hist2)
        await asyncio.to_thread(_persist_trim, app, sid)
    chat_title = await _update_title_after_first_message(app, sid, msg)

    if not terminated:
        _record_agent_event(
            event_type="chat_done" if not err else "chat_error",
            session_id=sid,
            version_id=version_id_for_history,
            request_id=request_id,
            payload={"reply_text": reply, "error": err, "message_ref": ai_message_ref or ""},
        )

    yield {
        "event": "done",
        "reply": reply,
        "session_id": sid,
        "chat_title": chat_title,
        "request_id": request_id,
        "message_ref": ai_message_ref if not terminated else None,
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
        app, req.session_id, msg, req.llm_mode, req.system_instruction
    )

    # Fast local reply path.
    if executor is None:
        local_reply: str = payload["local_reply"]  # type: ignore[index]
        version_id = str(payload.get("version_id") or "")

        async def _local_gen():
            lock: asyncio.Lock = app.state.lock
            sessions: dict[str, list[BaseMessage]] = app.state.sessions
            registry_lr: VersionRegistry = app.state.version_registry
            vid_lr = version_id or registry_lr.get_active().version_id
            request_id = str(uuid.uuid4())
            _record_agent_event(
                event_type="chat_request",
                session_id=sid,
                version_id=vid_lr,
                request_id=request_id,
                payload={"input_text": msg, "transport": "stream-local"},
            )
            async with lock:
                hist2 = sessions.get(sid, [])
                human_m = HumanMessage(content=msg)
                ai_m = AIMessage(content=local_reply)
                hist2.append(human_m)
                hist2.append(ai_m)
                _trim_session(hist2)
                await asyncio.to_thread(_persist_message, app, sid, human_m, vid_lr)
                message_ref = await asyncio.to_thread(_persist_message, app, sid, ai_m, vid_lr)
                await asyncio.to_thread(_persist_trim, app, sid)
            chat_title = await _update_title_after_first_message(app, sid, msg)
            _record_agent_event(
                event_type="chat_done",
                session_id=sid,
                version_id=vid_lr,
                request_id=request_id,
                payload={"reply_text": local_reply, "message_ref": message_ref or "", "route": "local"},
            )
            yield (
                json.dumps(
                    {
                        "event": "done",
                        "reply": local_reply,
                        "session_id": sid,
                        "chat_title": chat_title,
                        "request_id": request_id,
                        "message_ref": message_ref,
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
            llm_mode = data.get("llm_mode")
            if llm_mode not in ("auto", "gemini", "agent", "nvidia"):
                llm_mode = "auto"
            ws_system_instruction: str | None = data.get("system_instruction") or None

            try:
                sid, _hist, executor, payload, cancel_event = await _resolve_session(
                    app, req_session_id, msg, llm_mode, ws_system_instruction
                )
            except Exception as exc:
                await websocket.send_text(
                    json.dumps({"event": "error", "error": str(exc)}, ensure_ascii=False)
                )
                continue

            # Fast local reply path.
            if executor is None:
                local_reply: str = payload["local_reply"]  # type: ignore[index]
                version_id = str(payload.get("version_id") or "")
                lock: asyncio.Lock = app.state.lock
                sessions: dict[str, list[BaseMessage]] = app.state.sessions
                registry_ws: VersionRegistry = app.state.version_registry
                vid_ws = version_id or registry_ws.get_active().version_id
                request_id = str(uuid.uuid4())
                _record_agent_event(
                    event_type="chat_request",
                    session_id=sid,
                    version_id=vid_ws,
                    request_id=request_id,
                    payload={"input_text": msg, "transport": "ws-local"},
                )
                async with lock:
                    hist2 = sessions.get(sid, [])
                    human_mw = HumanMessage(content=msg)
                    ai_mw = AIMessage(content=local_reply)
                    hist2.append(human_mw)
                    hist2.append(ai_mw)
                    _trim_session(hist2)
                    await asyncio.to_thread(_persist_message, app, sid, human_mw, vid_ws)
                    message_ref = await asyncio.to_thread(_persist_message, app, sid, ai_mw, vid_ws)
                    await asyncio.to_thread(_persist_trim, app, sid)
                chat_title = await _update_title_after_first_message(app, sid, msg)
                _record_agent_event(
                    event_type="chat_done",
                    session_id=sid,
                    version_id=vid_ws,
                    request_id=request_id,
                    payload={"reply_text": local_reply, "message_ref": message_ref or "", "route": "local"},
                )
                await websocket.send_text(
                    json.dumps(
                        {
                            "event": "done",
                            "reply": local_reply,
                            "session_id": sid,
                            "chat_title": chat_title,
                            "request_id": request_id,
                            "message_ref": message_ref,
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
    request_id = str(uuid.uuid4())
    _record_agent_event(
        event_type="chat_request",
        session_id=sid,
        version_id="",
        request_id=request_id,
        payload={"input_text": msg, "route": "fallback"},
    )
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
        message_ref = await asyncio.to_thread(_persist_message, app, sid, ai_msg_fb, None)
        await asyncio.to_thread(_persist_trim, app, sid)
    chat_title = await _update_title_after_first_message(app, sid, msg)

    _record_agent_event(
        event_type="chat_done" if not err else "chat_error",
        session_id=sid,
        version_id="",
        request_id=request_id,
        payload={"reply_text": reply, "error": err, "message_ref": message_ref or "", "route": "fallback"},
    )
    return ChatResponse(
        reply=reply,
        session_id=sid,
        message_ref=message_ref,
        chat_title=chat_title,
        error=err,
    )


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
async def get_versions(request: Request) -> dict[str, Any]:
    registry: VersionRegistry = request.app.state.version_registry
    return {
        "versions": registry.list_versions(),
        "active_version_id": registry.get_active_id(),
        "available_model_profiles": request.app.state.available_profiles,
        "nvidia_chat_model": _nvidia_chat_model_display(request.app),
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


@app.get("/api/reminders/pending")
async def get_pending_reminders() -> dict[str, Any]:
    lock: asyncio.Lock = app.state.lock
    async with lock:
        items = list(app.state.reminder_notifications)
        app.state.reminder_notifications.clear()
    return {"items": items}


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


@app.post("/api/feedback")
async def submit_feedback(req: FeedbackRequest) -> dict[str, Any]:
    feedback_id = await asyncio.to_thread(
        write_feedback,
        _ROOT,
        session_id=req.session_id,
        message_ref=req.message_ref,
        rating=req.rating,
        comment=req.comment,
    )
    return {"ok": True, "feedback_id": feedback_id}


@app.get("/api/telemetry/agent-events")
async def get_agent_events(limit: int = 50) -> dict[str, Any]:
    items = await asyncio.to_thread(read_recent_agent_events, _ROOT, limit=limit)
    return {"items": items}


@app.get("/api/telemetry/feedback")
async def get_feedback_items(limit: int = 50) -> dict[str, Any]:
    items = await asyncio.to_thread(read_recent_feedback, _ROOT, limit=limit)
    return {"items": items}


def main() -> None:
    import uvicorn

    host = os.environ.get("WEB_HOST", "127.0.0.1").strip() or "127.0.0.1"
    port = int(os.environ.get("WEB_PORT", "8765"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
