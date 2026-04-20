from __future__ import annotations

import asyncio
import ast
import json
import os
import queue
import re
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
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pydantic import BaseModel, Field

from src.agent import (
    build_executor,
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
from src.tools import (
    delete_all_memory_items,
    delete_memory_item,
    list_memory_items,
)
from src.version_registry import VersionRegistry

_ROOT = Path(__file__).resolve().parent.parent
_STATIC = _ROOT / "frontend" / "dist"
_REGISTRY_PATH = _ROOT / "version_registry.json"

_MAX_SESSION_MESSAGES = 40


def _trim_session(msgs: list[BaseMessage]) -> None:
    if len(msgs) <= _MAX_SESSION_MESSAGES:
        return
    del msgs[: len(msgs) - _MAX_SESSION_MESSAGES]


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


class ClearRequest(BaseModel):
    session_id: str | None = None


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
        sid: str
        if req.session_id and req.session_id in sessions:
            sid = req.session_id
        elif version.session_id and version.session_id in sessions:
            sid = version.session_id
        else:
            sid = str(uuid.uuid4())
            sessions[sid] = []
            registry.update_session(version.version_id, sid)

        hist = sessions[sid]
        local_reply = _simple_local_reply(msg)
        if local_reply is None:
            local_reply = await asyncio.to_thread(local_quick_reply, msg, len(hist))
        if local_reply is not None:
            reply = local_reply
            err = None
        else:
            history_for_prompt = list(hist)
            executor = _get_executor(app, version.version_id, msg, len(hist))

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
            err = None if reply else f"No text output; full result: {result!r}"

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
    registry: VersionRegistry = app.state.version_registry

    async with lock:
        version = registry.get_active()
        if req.session_id and req.session_id in sessions:
            sid = req.session_id
        elif version.session_id and version.session_id in sessions:
            sid = version.session_id
        else:
            sid = str(uuid.uuid4())
            sessions[sid] = []
            registry.update_session(version.version_id, sid)

        hist = sessions[sid]
        local_reply = _simple_local_reply(msg)
        if local_reply is None:
            local_reply = await asyncio.to_thread(local_quick_reply, msg, len(hist))
        if local_reply is None:
            history_for_prompt = list(hist)
            executor = _get_executor(app, version.version_id, msg, len(hist))
            payload = {"input": msg, "chat_history": history_for_prompt}
        else:
            executor = None
            payload = {}

    if local_reply is not None:
        async def local_ndjson_gen():
            async with lock:
                hist2 = sessions.get(sid, [])
                hist2.append(HumanMessage(content=msg))
                hist2.append(AIMessage(content=local_reply))
                _trim_session(hist2)
            yield json.dumps(
                {
                    "event": "done",
                    "reply": local_reply,
                    "session_id": sid,
                    "error": None,
                },
                ensure_ascii=False,
            ) + "\n"

        return StreamingResponse(local_ndjson_gen(), media_type="application/x-ndjson")

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
        sessions[version.session_id] = []
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
