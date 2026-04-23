from __future__ import annotations

import os
import json
import platform
import re
import shlex
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Callable
from urllib.parse import quote
from zoneinfo import ZoneInfo

import httpx
from bs4 import BeautifulSoup
from langchain_community.vectorstores import Chroma
from langchain_core.tools import StructuredTool
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from pydantic import BaseModel, Field, model_validator

_MEMORY_COLLECTION = "agent_memory"
_ReminderSink = Callable[[dict[str, Any]], None]
_reminder_sink_lock = Lock()
_reminder_sink: _ReminderSink | None = None

def set_reminder_sink(sink: _ReminderSink | None) -> None:
    global _reminder_sink
    with _reminder_sink_lock:
        _reminder_sink = sink

def _emit_reminder(payload: dict[str, Any]) -> None:
    with _reminder_sink_lock:
        sink = _reminder_sink
    if sink is None:
        return
    try:
        sink(payload)
    except Exception:
        return

def _memory_store_for(
    chroma_dir: Path,
    embeddings: GoogleGenerativeAIEmbeddings,
    collection_name: str,
) -> Chroma:
    chroma_dir.mkdir(parents=True, exist_ok=True)
    return Chroma(
        persist_directory=str(chroma_dir),
        embedding_function=embeddings,
        collection_name=collection_name,
    )

def list_memory_items(
    chroma_dir: Path,
    embeddings: GoogleGenerativeAIEmbeddings,
    collection_name: str = _MEMORY_COLLECTION,
) -> list[dict]:
    try:
        store = _memory_store_for(chroma_dir, embeddings, collection_name)
        result = store._collection.get(include=["documents", "metadatas"])
        items = []
        ids = result.get("ids") or []
        docs = result.get("documents") or []
        metas = result.get("metadatas") or []
        for i, mem_id in enumerate(ids):
            items.append(
                {
                    "id": mem_id,
                    "content": docs[i] if i < len(docs) else "",
                    "metadata": metas[i] if i < len(metas) else {},
                }
            )
        return items
    except Exception as exc:
        return [{"error": str(exc)}]

def delete_memory_item(
    chroma_dir: Path,
    embeddings: GoogleGenerativeAIEmbeddings,
    item_id: str,
    collection_name: str = _MEMORY_COLLECTION,
) -> bool:
    try:
        store = _memory_store_for(chroma_dir, embeddings, collection_name)
        store._collection.delete(ids=[item_id])
        return True
    except Exception:
        return False

def delete_all_memory_items(
    chroma_dir: Path,
    embeddings: GoogleGenerativeAIEmbeddings,
    collection_name: str = _MEMORY_COLLECTION,
) -> bool:
    try:
        store = _memory_store_for(chroma_dir, embeddings, collection_name)
        result = store._collection.get(include=[])
        ids = result.get("ids") or []
        if ids:
            store._collection.delete(ids=ids)
        return True
    except Exception:
        return False

class WebSearchArgs(BaseModel):
    query: str = Field(description="Web search query")
    max_results: int = Field(default=5, description="Max results (1-10)")

def _run_web_search(query: str, max_results: int = 5) -> str:
    q = (query or "").strip()
    if not q:
        return "Empty search query."
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            return "Web search unavailable: install 'ddgs' (pip install ddgs)."
    n = max(1, min(int(max_results or 5), 10))
    rows: list[str] = []
    try:
        with DDGS() as ddgs:
            for i, r in enumerate(ddgs.text(q, max_results=n), 1):
                title = r.get("title") or ""
                url = r.get("href") or r.get("url") or ""
                body = r.get("body") or ""
                rows.append(f"{i}. {title}\n   {url}\n   {body}")
    except Exception as exc:
        return f"Web search failed: {exc}"
    return "\n\n".join(rows) if rows else "No results."

def make_web_search_tool() -> StructuredTool:
    return StructuredTool.from_function(
        name="web_search",
        description=(
            "Search the public web (DuckDuckGo) for fresh information. Returns "
            "titles, URLs, and snippets. Use for current events, facts not in "
            "memory, or to discover URLs to fetch with web_fetch."
        ),
        func=_run_web_search,
        args_schema=WebSearchArgs,
    )

class WebFetchArgs(BaseModel):
    url: str = Field(description="Absolute http(s) URL to fetch")

def _run_web_fetch(url: str) -> str:
    u = (url or "").strip()
    if not u.startswith(("http://", "https://")):
        return "URL must start with http:// or https://"
    try:
        with httpx.Client(
            follow_redirects=True,
            timeout=20.0,
            headers={"User-Agent": "Mozilla/5.0 (compatible; AgentBot/1.0)"},
        ) as client:
            resp = client.get(u)
            resp.raise_for_status()
            ct = resp.headers.get("content-type", "").lower()
            body = resp.text
    except Exception as exc:
        return f"Fetch failed: {exc}"

    if "html" in ct or "<html" in body[:500].lower():
        soup = BeautifulSoup(body, "html.parser")
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
    else:
        text = body

    if len(text) > 12000:
        text = text[:12000] + "\n... [truncated]"
    return text or "Empty response."

def make_web_fetch_tool() -> StructuredTool:
    return StructuredTool.from_function(
        name="web_fetch",
        description=(
            "Fetch a URL and return readable text (HTML stripped). Use after "
            "web_search to read the contents of a specific page."
        ),
        func=_run_web_fetch,
        args_schema=WebFetchArgs,
    )

def _memory_store(
    chroma_dir: Path,
    embeddings: GoogleGenerativeAIEmbeddings,
    collection_name: str = _MEMORY_COLLECTION,
) -> Chroma:
    return _memory_store_for(chroma_dir, embeddings, collection_name)

def _delete_memory_entries_for_calendar_event(
    chroma_dir: Path,
    embeddings: GoogleGenerativeAIEmbeddings,
    collection_name: str,
    calendar_id: str,
) -> int:
    """Remove memory entries that reference a calendar event id. Returns count deleted."""
    if not calendar_id:
        return 0
    try:
        store = _memory_store_for(chroma_dir, embeddings, collection_name)
        result = store._collection.get(include=["documents"])
        ids = result.get("ids") or []
        docs = result.get("documents") or []
        to_delete = [
            mid for mid, doc in zip(ids, docs)
            if calendar_id in (doc or "")
        ]
        if to_delete:
            store._collection.delete(ids=to_delete)
        return len(to_delete)
    except Exception:
        return 0

def _persist_memory_line(
    chroma_dir: Path,
    embeddings: GoogleGenerativeAIEmbeddings,
    collection_name: str,
    content: str,
    tags: str,
) -> str | None:
    c = (content or "").strip()
    if not c:
        return None
    meta: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "tags": (tags or "").strip(),
    }
    mem_id = f"mem-{uuid.uuid4()}"
    try:
        store = _memory_store(chroma_dir, embeddings, collection_name)
        store.add_texts(texts=[c], metadatas=[meta], ids=[mem_id])
    except Exception:
        return None
    return mem_id

class SaveMemoryArgs(BaseModel):
    content: str = Field(description="Fact or note to store for future recall")
    tags: str = Field(default="", description="Optional comma-separated tags")

class SearchMemoryArgs(BaseModel):
    query: str = Field(description="Question or keywords to search remembered facts")
    k: int = Field(default=5, description="Max number of results (1-10)")

def make_memory_tools(
    chroma_dir: Path,
    embeddings: GoogleGenerativeAIEmbeddings,
    collection_name: str = _MEMORY_COLLECTION,
) -> list[StructuredTool]:
    store = _memory_store(chroma_dir, embeddings, collection_name)

    def _save(content: str, tags: str = "") -> str:
        c = (content or "").strip()
        if not c:
            return "Empty memory content."
        meta: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "tags": (tags or "").strip(),
        }
        mem_id = f"mem-{uuid.uuid4()}"
        try:
            store.add_texts(texts=[c], metadatas=[meta], ids=[mem_id])
        except Exception as exc:
            return f"Memory save failed: {exc}"
        return f"Saved memory ({mem_id})."

    def _search(query: str, k: int = 5) -> str:
        q = (query or "").strip()
        if not q:
            return "Empty memory query."
        n = max(1, min(int(k or 5), 10))
        try:
            docs = store.similarity_search(q, k=n)
        except Exception as exc:
            return f"Memory search failed: {exc}"
        if not docs:
            return "No matching memories."
        rows: list[str] = []
        for i, d in enumerate(docs, 1):
            md = d.metadata or {}
            ts = md.get("ts", "")
            tags = md.get("tags", "")
            head_parts = [p for p in (ts, f"[{tags}]" if tags else "") if p]
            head = " ".join(head_parts)
            rows.append(f"{i}. {head}\n{d.page_content}".strip())
        return "\n\n".join(rows)

    save_tool = StructuredTool.from_function(
        name="save_to_memory",
        description=(
            "Persist a concrete fact, decision, or piece of context for future "
            "sessions. Use sparingly for durable, useful information the user "
            "or agent will benefit from recalling later."
        ),
        func=_save,
        args_schema=SaveMemoryArgs,
    )
    search_tool = StructuredTool.from_function(
        name="search_memory",
        description=(
            "Search the agent's persistent memory for previously saved facts "
            "or notes. Call BEFORE web_search when a question may rely on "
            "earlier context."
        ),
        func=_search,
        args_schema=SearchMemoryArgs,
    )
    return [save_tool, search_tool]

class AskReasoningArgs(BaseModel):
    question: str = Field(
        description="Complex question or task to delegate to a reasoning LLM"
    )
    context: str = Field(default="", description="Optional background information")

def make_reasoning_tool() -> StructuredTool:
    def _ask(question: str, context: str = "") -> str:
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return "Reasoning model unavailable: no Gemini API key."
        model = (
            (os.environ.get("GEMINI_REASONING_MODEL") or "").strip()
            or (os.environ.get("GEMINI_PRO_MODEL") or "").strip()
            or "gemini-2.5-pro"
        )
        try:
            llm = ChatGoogleGenerativeAI(
                model=model, temperature=0, google_api_key=api_key
            )
            prompt = (
                "You are a careful reasoning assistant. Think step by step and "
                "produce a precise, well-justified answer.\n\n"
                f"Context:\n{(context or '').strip() or '(none)'}\n\n"
                f"Question:\n{(question or '').strip()}"
            )
            resp = llm.invoke(prompt)
            text = getattr(resp, "content", None)
            if isinstance(text, list):
                parts = []
                for p in text:
                    if isinstance(p, dict) and isinstance(p.get("text"), str):
                        parts.append(p["text"])
                    elif isinstance(p, str):
                        parts.append(p)
                return "\n\n".join(parts).strip()
            return str(text or resp).strip()
        except Exception as exc:
            return f"Reasoning model error: {exc}"

    return StructuredTool.from_function(
        name="ask_reasoning_model",
        description=(
            "Delegate a complex analytical question to a stronger reasoning "
            "LLM (e.g., Gemini 2.5 Pro). Use for deep analysis, multi-step "
            "logic, or synthesizing large context. Pass relevant context "
            "explicitly in the context argument."
        ),
        func=_ask,
        args_schema=AskReasoningArgs,
    )

class ShellCommandArgs(BaseModel):
    command: str = Field(description="Shell command to execute locally")

def _run_shell_command(command: str) -> str:
    cmd = (command or "").strip()
    if not cmd:
        return "Empty shell command."
    try:
        # Validate basic shell syntax early for clearer feedback.
        shlex.split(cmd)
    except ValueError as exc:
        return f"Invalid shell command: {exc}"
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        return "Shell command timed out after 10 seconds."
    except Exception as exc:
        return f"Shell command failed: {exc}"

    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    parts: list[str] = [f"exit_code: {result.returncode}"]
    if stdout:
        parts.append(f"stdout:\n{stdout}")
    if stderr:
        parts.append(f"stderr:\n{stderr}")
    if len(parts) == 1:
        parts.append("No output.")
    text = "\n\n".join(parts)
    if len(text) > 10000:
        text = text[:10000] + "\n\n... [truncated]"
    return text

def make_shell_tool() -> StructuredTool:
    return StructuredTool.from_function(
        name="execute_shell_command",
        description=(
            "Execute local shell commands (e.g. date, grep, tail, ls) to retrieve "
            "system data, inspect logs, or extract specific text."
        ),
        func=_run_shell_command,
        args_schema=ShellCommandArgs,
    )

def _get_local_datetime() -> str:
    now = datetime.now().astimezone()
    tz = now.tzname() or ""
    return (
        f"local_iso: {now.isoformat(timespec='seconds')}\n"
        f"today_date: {now.strftime('%Y-%m-%d')}\n"
        f"local_clock: {now.strftime('%H:%M:%S')}\n"
        f"timezone_label: {tz}"
    )

def make_local_datetime_tool() -> StructuredTool:
    return StructuredTool.from_function(
        name="get_local_datetime",
        description=(
            "Return current local date and time from the application host without shell "
            "(no arguments). Use for today's date, current time, or anchoring "
            "calendar_create_event to 'today'. Prefer this over execute_shell_command with date."
        ),
        func=_get_local_datetime,
    )

_ISO_DT_PATTERN = re.compile(
    r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?"
)

def _extract_two_iso_datetimes(text: str) -> tuple[str | None, str | None]:
    matches = list(_ISO_DT_PATTERN.finditer(text))
    if len(matches) >= 2:
        return matches[0].group(0), matches[1].group(0)
    return None, None

_LOOSE_CAL_KV_RE = re.compile(
    r'"(start_at|end_at|startAt|endAt|begin_at|title|timezone|description|reminder_message|'
    r"source_session_id|event_id)"
    r'"\s*:\s*"((?:[^"\\]|\\.)*)"',
)

def _unescape_json_string_fragment(s: str) -> str:
    return (
        s.replace("\\n", "\n")
        .replace("\\t", "\t")
        .replace('\\"', '"')
        .replace("\\\\", "\\")
    )

def _calendar_fields_from_jsonish_blob(blob: str) -> dict[str, Any]:
    """Strict json.loads, else quoted key/value pairs (handles truncated invalid JSON)."""
    s = (blob or "").strip()
    if not s.startswith("{"):
        return {}
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
        return {}
    except json.JSONDecodeError:
        pass
    loose: dict[str, str] = {}
    for m in _LOOSE_CAL_KV_RE.finditer(s):
        loose[m.group(1)] = _unescape_json_string_fragment(m.group(2))
    return loose

def _normalize_calendar_create_top_level_keys(data: dict[str, Any]) -> dict[str, Any]:
    """ReAct / models often send camelCase; LangChain may also bind the whole JSON into one field."""
    out = dict(data)
    aliases = (
        (("startAt", "start_at"), ("start", "start_at"), ("begin_at", "start_at")),
        (("endAt", "end_at"), ("end", "end_at")),
        (("timeZone", "timezone"), ("tz", "timezone")),
    )
    for pairs in aliases:
        for src, dst in pairs:
            if dst not in out or not str(out.get(dst) or "").strip():
                v = out.get(src)
                if v is None:
                    continue
                s = v.strip() if isinstance(v, str) else str(v).strip()
                if s:
                    out[dst] = s
                    break
    return out

def _merge_calendar_create_nested(data: dict[str, Any]) -> dict[str, Any]:
    """Fill start_at/end_at/title from nested JSON wrongly passed as title only."""
    out = _normalize_calendar_create_top_level_keys(dict(data))
    st = (out.get("start_at") or "").strip()
    et = (out.get("end_at") or "").strip()
    if st and et:
        return out

    blob0 = str(out.get("title") or "")
    nested = _calendar_fields_from_jsonish_blob(blob0)
    if nested:
        for src_key, dst in (
            ("title", "title"),
            ("start_at", "start_at"),
            ("startAt", "start_at"),
            ("start", "start_at"),
            ("begin_at", "start_at"),
            ("end_at", "end_at"),
            ("endAt", "end_at"),
            ("end", "end_at"),
            ("timezone", "timezone"),
            ("timeZone", "timezone"),
            ("description", "description"),
            ("reminder_message", "reminder_message"),
            ("source_session_id", "source_session_id"),
        ):
            if dst in out and str(out.get(dst) or "").strip():
                continue
            v = nested.get(src_key)
            if v is None:
                continue
            if isinstance(v, str) and v.strip():
                out[dst] = v.strip()
            elif isinstance(v, (int, float)) and dst in ("start_at", "end_at"):
                continue
            elif not isinstance(v, str):
                out[dst] = str(v)

        inner_title = nested.get("title")
        if isinstance(inner_title, str) and inner_title.strip():
            out["title"] = inner_title.strip()

    st2 = (out.get("start_at") or "").strip()
    et2 = (out.get("end_at") or "").strip()
    if not st2 or not et2:
        ds, de = _extract_two_iso_datetimes(blob0)
        if ds and not st2:
            out["start_at"] = ds
        if de and not et2:
            out["end_at"] = de
    return out

def _merge_calendar_update_nested(data: dict[str, Any]) -> dict[str, Any]:
    """Unpack JSON wrongly placed in event_id (or title) for update tool."""
    out = dict(data)
    eid = str(out.get("event_id") or "").strip()
    if eid.startswith("{"):
        nested = _calendar_fields_from_jsonish_blob(eid)
        if nested.get("event_id"):
            out["event_id"] = str(nested["event_id"]).strip()
        blob_alt = str(out.get("title") or "")
        src = nested or _calendar_fields_from_jsonish_blob(blob_alt)
        if src:
            for src_key, dst in (
                ("title", "title"),
                ("start_at", "start_at"),
                ("startAt", "start_at"),
                ("start", "start_at"),
                ("begin_at", "start_at"),
                ("end_at", "end_at"),
                ("endAt", "end_at"),
                ("end", "end_at"),
                ("timezone", "timezone"),
                ("timeZone", "timezone"),
                ("description", "description"),
            ):
                if dst in out and str(out.get(dst) or "").strip():
                    continue
                v = src.get(src_key)
                if isinstance(v, str) and v.strip():
                    out[dst] = v.strip()
    return out

class CalendarCreateEventArgs(BaseModel):
    title: str = Field(description="Event title")
    start_at: str = Field(default="", description="Event start datetime in ISO-8601")
    end_at: str = Field(default="", description="Event end datetime in ISO-8601")
    timezone: str = Field(default="Asia/Taipei", description="IANA timezone")
    description: str = Field(default="", description="Event description")
    reminder_message: str = Field(default="", description="Reminder message for chat")
    source_session_id: str = Field(default="", description="Origin session id")

    @model_validator(mode="before")
    @classmethod
    def _unwrap_nested_json_title(cls, data: Any) -> Any:
        if isinstance(data, str):
            s = data.strip()
            if not s:
                return {}
            if s.startswith("{"):
                try:
                    parsed = json.loads(s)
                    data = parsed if isinstance(parsed, dict) else {"title": str(parsed)}
                except json.JSONDecodeError:
                    return {"title": s}
            else:
                return {"title": s}
        if not isinstance(data, dict):
            return {"title": str(data)}
        return _merge_calendar_create_nested(data)

class CalendarUpdateEventArgs(BaseModel):
    event_id: str = Field(description="Google Calendar event id")
    title: str = Field(default="", description="Event title")
    start_at: str = Field(default="", description="Event start datetime in ISO-8601")
    end_at: str = Field(default="", description="Event end datetime in ISO-8601")
    timezone: str = Field(default="Asia/Taipei", description="IANA timezone")
    description: str = Field(default="", description="Event description")

    @model_validator(mode="before")
    @classmethod
    def _unwrap_nested_json(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        return _merge_calendar_update_nested(data)

class CalendarDeleteEventArgs(BaseModel):
    event_id: str = Field(description="Google Calendar event id")

def _normalize_iso_datetime(raw: str, timezone_name: str) -> datetime:
    s = (raw or "").strip()
    if not s:
        raise ValueError("datetime is required")
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    parsed = datetime.fromisoformat(s)
    tz = ZoneInfo(timezone_name)
    if parsed.tzinfo is None:
        # Interpret naive datetime as local wall-clock time in requested timezone.
        parsed = parsed.replace(tzinfo=tz)
    return parsed.astimezone(tz)

_GOOGLE_CALENDAR_SCOPES = ["https://www.googleapis.com/auth/calendar.events"]

def _calendar_project_root() -> Path:
    return Path(__file__).resolve().parent.parent

def _resolve_google_calendar_access_token() -> str:
    """Prefer OAuth token cache (refreshable); fall back to GOOGLE_CALENDAR_ACCESS_TOKEN."""
    cache_override = (os.environ.get("GOOGLE_OAUTH_TOKEN_CACHE_FILE") or "").strip()
    cache_path = Path(cache_override) if cache_override else _calendar_project_root() / "google_oauth_token.json"

    if cache_path.is_file():
        try:
            from google.auth.transport.requests import Request as GARequest
            from google.oauth2.credentials import Credentials
        except ImportError:
            pass
        else:
            try:
                creds = Credentials.from_authorized_user_file(
                    str(cache_path),
                    scopes=_GOOGLE_CALENDAR_SCOPES,
                )
                if creds.expired and creds.refresh_token:
                    creds.refresh(GARequest())
                    cache_path.write_text(creds.to_json(), encoding="utf-8")
                tok = (creds.token or "").strip()
                if tok:
                    return tok
            except Exception:
                pass

    env_tok = (os.environ.get("GOOGLE_CALENDAR_ACCESS_TOKEN") or "").strip()
    if env_tok:
        return env_tok
    raise RuntimeError(
        "Missing Google Calendar credentials: set GOOGLE_CALENDAR_ACCESS_TOKEN in .env or run "
        "scripts/get_google_oauth_token.py to create google_oauth_token.json (optional: "
        "GOOGLE_OAUTH_TOKEN_CACHE_FILE)."
    )

def _raise_for_calendar_status(resp: httpx.Response, context: str) -> None:
    try:
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        body = (exc.response.text or "").strip()[:1200]
        hint = ""
        if exc.response.status_code == 401:
            hint = (
                " OAuth access token expired, revoked, or wrong type (need OAuth user token with "
                "calendar.events scope, not GEMINI_API_KEY). Refresh: run scripts/get_google_oauth_token.py "
                "and update GOOGLE_CALENDAR_ACCESS_TOKEN, or install google-auth + use google_oauth_token.json "
                "for automatic refresh."
            )
        raise RuntimeError(f"{context}: HTTP {exc.response.status_code}.{hint}\n{body}") from exc

def _create_google_calendar_event(
    *,
    title: str,
    start_at: datetime,
    end_at: datetime,
    timezone_name: str,
    description: str,
) -> dict[str, Any]:
    token = _resolve_google_calendar_access_token()
    calendar_id = (os.environ.get("GOOGLE_CALENDAR_ID") or "primary").strip() or "primary"
    payload = {
        "summary": title,
        "description": description,
        "start": {"dateTime": start_at.isoformat(), "timeZone": timezone_name},
        "end": {"dateTime": end_at.isoformat(), "timeZone": timezone_name},
    }
    url = f"https://www.googleapis.com/calendar/v3/calendars/{quote(calendar_id, safe='')}/events"
    with httpx.Client(timeout=20.0) as client:
        resp = client.post(
            url,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=payload,
        )
        _raise_for_calendar_status(resp, "Google Calendar create event")
        data = resp.json()
    return {"id": data.get("id", ""), "html_link": data.get("htmlLink", "")}

def _google_calendar_headers() -> tuple[str, dict[str, str]]:
    token = _resolve_google_calendar_access_token()
    calendar_id = (os.environ.get("GOOGLE_CALENDAR_ID") or "primary").strip() or "primary"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    return calendar_id, headers

def _update_google_calendar_event(
    *,
    event_id: str,
    title: str,
    start_at: datetime | None,
    end_at: datetime | None,
    timezone_name: str,
    description: str,
) -> dict[str, Any]:
    event_id_clean = (event_id or "").strip()
    if not event_id_clean:
        raise ValueError("event_id is required")
    calendar_id, headers = _google_calendar_headers()
    patch_payload: dict[str, Any] = {}
    if title.strip():
        patch_payload["summary"] = title.strip()
    if description.strip():
        patch_payload["description"] = description.strip()
    if start_at is not None:
        patch_payload["start"] = {"dateTime": start_at.isoformat(), "timeZone": timezone_name}
    if end_at is not None:
        patch_payload["end"] = {"dateTime": end_at.isoformat(), "timeZone": timezone_name}
    if not patch_payload:
        raise ValueError("no updatable fields provided")
    url = (
        "https://www.googleapis.com/calendar/v3/calendars/"
        f"{quote(calendar_id, safe='')}/events/{quote(event_id_clean, safe='')}"
    )
    with httpx.Client(timeout=20.0) as client:
        resp = client.patch(url, headers=headers, json=patch_payload)
        _raise_for_calendar_status(resp, "Google Calendar update event")
        data = resp.json()
    return {"id": data.get("id", ""), "html_link": data.get("htmlLink", "")}

def _delete_google_calendar_event(*, event_id: str) -> None:
    event_id_clean = (event_id or "").strip()
    if not event_id_clean:
        raise ValueError("event_id is required")
    calendar_id, headers = _google_calendar_headers()
    url = (
        "https://www.googleapis.com/calendar/v3/calendars/"
        f"{quote(calendar_id, safe='')}/events/{quote(event_id_clean, safe='')}"
    )
    with httpx.Client(timeout=20.0) as client:
        resp = client.delete(url, headers=headers)
        _raise_for_calendar_status(resp, "Google Calendar delete event")

def make_calendar_tool(
    *,
    chroma_dir: Path | None = None,
    embeddings: GoogleGenerativeAIEmbeddings | None = None,
    memory_collection: str | None = None,
) -> StructuredTool:
    def _create_event(
        title: str,
        start_at: str = "",
        end_at: str = "",
        timezone: str = "Asia/Taipei",
        description: str = "",
        reminder_message: str = "",
        source_session_id: str = "",
    ) -> str:
        t = (title or "").strip()
        sa = (start_at or "").strip()
        ea = (end_at or "").strip()
        tz = (timezone or "Asia/Taipei").strip() or "Asia/Taipei"
        desc = (description or "").strip()
        rem = (reminder_message or "").strip()
        sid = (source_session_id or "").strip()

        # ReAct often binds the entire Action Input JSON string into `title` only.
        if t.startswith("{") and (not sa or not ea):
            try:
                blob = json.loads(t)
            except json.JSONDecodeError:
                blob = None
            if isinstance(blob, dict):
                blob = _merge_calendar_create_nested(blob)
                t = str(blob.get("title") or "").strip() or t
                sa = str(blob.get("start_at") or "").strip() or sa
                ea = str(blob.get("end_at") or "").strip() or ea
                if blob.get("timezone"):
                    tz = str(blob.get("timezone") or "").strip() or tz
                if blob.get("description") is not None and not desc:
                    desc = str(blob.get("description") or "").strip()
                if blob.get("reminder_message") is not None and not rem:
                    rem = str(blob.get("reminder_message") or "").strip()
                if blob.get("source_session_id") is not None and not sid:
                    sid = str(blob.get("source_session_id") or "").strip()

        if not t:
            return '{"ok": false, "error": "title is required"}'
        if not sa or not ea:
            return json.dumps(
                {
                    "ok": False,
                    "error": "start_at and end_at are required (ISO-8601).",
                },
                ensure_ascii=False,
            )
        tz_name = tz
        try:
            start_dt = _normalize_iso_datetime(sa, tz_name)
            end_dt = _normalize_iso_datetime(ea, tz_name)
        except Exception as exc:
            return json.dumps({"ok": False, "error": f"invalid datetime: {str(exc)}"}, ensure_ascii=False)
        if end_dt <= start_dt:
            return json.dumps({"ok": False, "error": "end_at must be after start_at"}, ensure_ascii=False)
        warning_messages: list[str] = []
        google_result: dict[str, Any] = {}
        apple_result: dict[str, Any] = {}

        try:
            google_result = _create_google_calendar_event(
                title=t,
                start_at=start_dt,
                end_at=end_dt,
                timezone_name=tz_name,
                description=desc,
            )
            google_success = True
        except Exception as exc:
            return json.dumps(
                {
                    "ok": False,
                    "google_success": False,
                    "apple_success": False,
                    "error": f"google calendar create failed: {str(exc)}",
                },
                ensure_ascii=False,
            )

        gid = str(google_result.get("id") or "").strip()
        if not gid:
            return json.dumps(
                {
                    "ok": False,
                    "google_success": False,
                    "apple_success": False,
                    "error": "google calendar API returned no event id (event was not created)",
                },
                ensure_ascii=False,
            )

        # Apple Calendar sync disabled (API errors caused agent retry loops).
        # try:
        #     apple_result = _create_apple_calendar_event(
        #         title=t,
        #         start_at=start_dt,
        #         end_at=end_dt,
        #         timezone_name=tz_name,
        #         description=desc,
        #     )
        #     apple_success = True
        # except Exception as exc:
        #     apple_success = False
        #     warning_messages.append(f"apple calendar create skipped/failed: {str(exc)}")
        apple_result = {"id": "", "url": ""}
        apple_success = False

        _emit_reminder(
            {
                "title": t,
                "reminder_at": start_dt.isoformat(),
                "message": rem or f"提醒：{t}",
                "source_session_id": sid,
            }
        )
        memory_item_id: str | None = None
        if chroma_dir is not None and embeddings is not None and (memory_collection or "").strip():
            mem_line = (
                "Google Calendar event created | "
                f"google_calendar_id={gid} | title={json.dumps(t, ensure_ascii=False)} | "
                f"start_at={start_dt.isoformat()} | end_at={end_dt.isoformat()}"
            )
            memory_item_id = _persist_memory_line(
                chroma_dir,
                embeddings,
                memory_collection.strip(),
                mem_line,
                "google-calendar,calendar",
            )

        result = {
            "ok": True,
            "google_success": google_success,
            "apple_success": apple_success,
            "event_ids": {
                "google": gid,
                "apple": apple_result.get("id", ""),
            },
            "event_links": {
                "google": google_result.get("html_link", ""),
                "apple": apple_result.get("url", ""),
            },
            "warnings": warning_messages,
            "saved_to_memory": memory_item_id is not None,
            "memory_item_id": memory_item_id or "",
        }
        return json.dumps(result, ensure_ascii=False)

    return StructuredTool.from_function(
        name="calendar_create_event",
        description=(
            "Create a calendar event in Google Calendar. Also schedules an in-app reminder. "
            "Google failure is fatal. "
            "Success means JSON with ok:true and non-empty event_ids.google; tell the user Google succeeded "
            "only if Observation contains those."
        ),
        func=_create_event,
        args_schema=CalendarCreateEventArgs,
    )

def make_calendar_update_tool(
    *,
    chroma_dir: Path | None = None,
    embeddings: GoogleGenerativeAIEmbeddings | None = None,
    memory_collection: str | None = None,
) -> StructuredTool:
    def _update_event(
        event_id: str,
        title: str = "",
        start_at: str = "",
        end_at: str = "",
        timezone: str = "Asia/Taipei",
        description: str = "",
    ) -> str:
        tz_name = (timezone or "Asia/Taipei").strip() or "Asia/Taipei"
        start_dt: datetime | None = None
        end_dt: datetime | None = None
        if (start_at or "").strip():
            start_dt = _normalize_iso_datetime(start_at, tz_name)
        if (end_at or "").strip():
            end_dt = _normalize_iso_datetime(end_at, tz_name)
        if start_dt and end_dt and end_dt <= start_dt:
            return json.dumps({"ok": False, "error": "end_at must be after start_at"}, ensure_ascii=False)
        try:
            result = _update_google_calendar_event(
                event_id=event_id,
                title=title,
                start_at=start_dt,
                end_at=end_dt,
                timezone_name=tz_name,
                description=description,
            )
        except Exception as exc:
            return json.dumps({"ok": False, "error": f"google calendar update failed: {str(exc)}"}, ensure_ascii=False)

        memory_entries_removed = 0
        memory_item_id: str | None = None
        if chroma_dir is not None and embeddings is not None and (memory_collection or "").strip():
            collection = memory_collection.strip()
            memory_entries_removed = _delete_memory_entries_for_calendar_event(
                chroma_dir,
                embeddings,
                collection,
                (event_id or "").strip(),
            )
            resolved_event_id = str(result.get("id") or "").strip() or (event_id or "").strip()
            mem_line = (
                "Google Calendar event updated | "
                f"google_calendar_id={resolved_event_id} | title={json.dumps((title or '').strip(), ensure_ascii=False)} | "
                f"start_at={(start_dt.isoformat() if start_dt else '')} | end_at={(end_dt.isoformat() if end_dt else '')}"
            )
            memory_item_id = _persist_memory_line(
                chroma_dir,
                embeddings,
                collection,
                mem_line,
                "google-calendar,calendar",
            )

        return json.dumps(
            {
                "ok": True,
                "event_id": result.get("id", ""),
                "event_link": result.get("html_link", ""),
                "memory_entries_removed": memory_entries_removed,
                "saved_to_memory": memory_item_id is not None,
                "memory_item_id": memory_item_id or "",
            },
            ensure_ascii=False,
        )

    return StructuredTool.from_function(
        name="calendar_update_event",
        description="Update an existing Google Calendar event by event_id.",
        func=_update_event,
        args_schema=CalendarUpdateEventArgs,
    )

def make_calendar_delete_tool(
    *,
    chroma_dir: Path | None = None,
    embeddings: GoogleGenerativeAIEmbeddings | None = None,
    memory_collection: str | None = None,
) -> StructuredTool:
    def _delete_event(event_id: str) -> str:
        event_id_clean = (event_id or "").strip()
        try:
            _delete_google_calendar_event(event_id=event_id_clean)
        except Exception as exc:
            return json.dumps({"ok": False, "error": f"google calendar delete failed: {str(exc)}"}, ensure_ascii=False)

        memory_entries_removed = 0
        if chroma_dir is not None and embeddings is not None and (memory_collection or "").strip():
            memory_entries_removed = _delete_memory_entries_for_calendar_event(
                chroma_dir,
                embeddings,
                memory_collection.strip(),
                event_id_clean,
            )
        return json.dumps(
            {
                "ok": True,
                "event_id": event_id_clean,
                "memory_entries_removed": memory_entries_removed,
            },
            ensure_ascii=False,
        )

    return StructuredTool.from_function(
        name="calendar_delete_event",
        description="Delete an existing Google Calendar event by event_id.",
        func=_delete_event,
        args_schema=CalendarDeleteEventArgs,
    )

def _mac_calendar_supported() -> bool:
    return platform.system() == "Darwin"

def _escape_applescript_string(s: str) -> str:
    return (s or "").replace("\\", "\\\\").replace('"', '\\"')

def _sanitize_calendar_text_for_applescript(s: str) -> str:
    """Single-line AppleScript string literals; collapse whitespace/newlines."""
    return " ".join((s or "").split())

def _datetime_to_mac_local(dt: datetime, timezone_name: str) -> datetime:
    tz_name = (timezone_name or "Asia/Taipei").strip() or "Asia/Taipei"
    tz = ZoneInfo(tz_name)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz)
    return dt.astimezone(tz)

def _applescript_assign_date(var_name: str, dt: datetime) -> str:
    return (
        f"set {var_name} to current date\n"
        f"set year of {var_name} to {dt.year}\n"
        f"set month of {var_name} to {dt.month}\n"
        f"set day of {var_name} to {dt.day}\n"
        f"set hours of {var_name} to {dt.hour}\n"
        f"set minutes of {var_name} to {dt.minute}\n"
        f"set seconds of {var_name} to {dt.second}\n"
    )

def _applescript_indent_block(block: str, prefix: str) -> str:
    lines: list[str] = []
    for raw in block.splitlines():
        line = raw.strip()
        if line:
            lines.append(prefix + line)
    return ("\n".join(lines) + "\n") if lines else ""

def _run_applescript(script: str) -> str:
    result = subprocess.run(
        ["osascript", "-"],
        input=script,
        capture_output=True,
        text=True,
        timeout=45,
    )
    if result.returncode != 0:
        err = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(err or f"osascript failed with exit {result.returncode}")
    out = (result.stdout or "").strip()
    return out

def _create_mac_calendar_event(
    *,
    title: str,
    start_at: datetime,
    end_at: datetime,
    timezone_name: str,
    description: str,
    alert_minutes_before: int = 0,
) -> str:
    title_esc = _escape_applescript_string(_sanitize_calendar_text_for_applescript(title))
    desc_esc = _escape_applescript_string(_sanitize_calendar_text_for_applescript(description))
    start_local = _datetime_to_mac_local(start_at, timezone_name)
    end_local = _datetime_to_mac_local(end_at, timezone_name)
    start_block = _applescript_assign_date("startDate", start_local)
    end_block = _applescript_assign_date("endDate", end_local)
    script = (
        'tell application "Calendar"\n'
        "  set targetCalendar to first calendar whose writable is true\n"
        "  tell targetCalendar\n"
        f"{start_block}"
        f"{end_block}"
        "    set newEvent to make new event at end with properties {summary:\""
        f"{title_esc}"
        '", start date:startDate, end date:endDate, description:"'
        f"{desc_esc}"
        '"}\n'
    )
    if alert_minutes_before > 0:
        script += (
            "    tell newEvent\n"
            f"      make new display alarm at end with properties {{trigger interval:-{alert_minutes_before}}}\n"
            "    end tell\n"
        )
    script += (
        "    get uid of newEvent\n"
        "  end tell\n"
        "end tell\n"
    )
    uid = _run_applescript(script).strip()
    if not uid:
        raise RuntimeError("Calendar returned empty event uid")
    return uid

def _update_mac_calendar_event(
    *,
    match_title: str = "",
    match_uid: str = "",
    new_title: str,
    start_at: datetime | None,
    end_at: datetime | None,
    timezone_name: str,
    description: str | None,
    alert_minutes_before: int | None = None,
) -> None:
    uid = (match_uid or "").strip()
    by_uid = bool(uid)
    if by_uid:
        match_esc = _escape_applescript_string(uid)
        whose = "every event whose uid is matchKey"
    else:
        match_esc = _escape_applescript_string(
            _sanitize_calendar_text_for_applescript(match_title)
        )
        whose = "every event whose summary is matchKey"
    new_title_esc = _escape_applescript_string(_sanitize_calendar_text_for_applescript(new_title))
    desc_esc = (
        None
        if description is None
        else _escape_applescript_string(_sanitize_calendar_text_for_applescript(description))
    )

    date_setup = ""
    if start_at is not None:
        sl = _datetime_to_mac_local(start_at, timezone_name)
        date_setup += _applescript_indent_block(_applescript_assign_date("newStart", sl), "  ")
    if end_at is not None:
        el = _datetime_to_mac_local(end_at, timezone_name)
        date_setup += _applescript_indent_block(_applescript_assign_date("newEnd", el), "  ")

    # Mutations run inside tell aCal → tell (item 1 of evList) so event refs stay valid (Calendar.app quirk).
    inner_evt = ""
    if new_title.strip():
        inner_evt += f'        set summary to "{new_title_esc}"\n'
    if start_at is not None:
        inner_evt += "        set start date to newStart\n"
    if end_at is not None:
        inner_evt += "        set end date to newEnd\n"
    if description is not None:
        inner_evt += f'        set description to "{desc_esc}"\n'
    if alert_minutes_before is not None:
        inner_evt += "        delete every display alarm\n"
        if alert_minutes_before > 0:
            inner_evt += (
                "        make new display alarm at end with properties "
                f"{{trigger interval:-{alert_minutes_before}}}\n"
            )

    script = (
        'tell application "Calendar"\n'
        f'  set matchKey to "{match_esc}"\n'
        f"{date_setup}"
        "  set didMutate to false\n"
        "  repeat with aCal in (every calendar whose writable is true)\n"
        "    tell aCal\n"
        f"      set evList to {whose}\n"
        "      if (count of evList) > 0 then\n"
        "        tell (item 1 of evList)\n"
        f"{inner_evt}"
        "        end tell\n"
        "        set didMutate to true\n"
        "        exit repeat\n"
        "      end if\n"
        "    end tell\n"
        "  end repeat\n"
        '  if didMutate is false then error "No matching event on any writable calendar."\n'
        "end tell\n"
    )
    _run_applescript(script)

def _delete_mac_calendar_event(*, title: str = "", match_uid: str = "") -> None:
    uid = (match_uid or "").strip()
    by_uid = bool(uid)
    if by_uid:
        key_esc = _escape_applescript_string(uid)
        whose_cond = "uid is matchKey"
    else:
        key_esc = _escape_applescript_string(_sanitize_calendar_text_for_applescript(title))
        whose_cond = "summary is matchKey"
    script = (
        'tell application "Calendar"\n'
        f'  set matchKey to "{key_esc}"\n'
        "  set didDel to false\n"
        "  set allCals to every calendar whose writable is true\n"
        "  repeat with aCal in allCals\n"
        f"    set evList to (every event of aCal whose {whose_cond})\n"
        "    if (count of evList) > 0 then\n"
        "      tell aCal to delete (item 1 of evList)\n"
        "      set didDel to true\n"
        "    end if\n"
        "    if didDel then exit repeat\n"
        "  end repeat\n"
        '  if didDel is false then error "No matching event on any writable calendar."\n'
        "end tell\n"
    )
    _run_applescript(script)

class MacCalendarCreateEventArgs(BaseModel):
    title: str = Field(description="Event title")
    start_at: str = Field(description="Event start datetime in ISO-8601")
    end_at: str = Field(description="Event end datetime in ISO-8601")
    timezone: str = Field(default="Asia/Taipei", description="IANA timezone")
    description: str = Field(default="", description="Event notes")
    alert_minutes_before: int = Field(default=0, description="Minutes before the event to trigger an alert (0 for no alert)")

class MacCalendarUpdateEventArgs(BaseModel):
    match_title: str = Field(
        default="",
        description="Current event title (exact match, first hit). Omit if apple_calendar_id is set.",
    )
    apple_calendar_id: str = Field(
        default="",
        description="Calendar event uid from mac_calendar_create_event; preferred over match_title.",
    )
    new_title: str = Field(default="", description="New title, if changing")
    start_at: str = Field(default="", description="New start datetime ISO-8601, if changing")
    end_at: str = Field(default="", description="New end datetime ISO-8601, if changing")
    timezone: str = Field(default="Asia/Taipei", description="IANA timezone for interpreting times")
    description: str = Field(default="", description="New notes; omit fields you do not change")
    alert_minutes_before: int | None = Field(default=None, description="New alert time in minutes before event, if changing (0 to remove)")

class MacCalendarDeleteEventArgs(BaseModel):
    title: str = Field(
        default="",
        description="Event title to delete (exact match). Omit if apple_calendar_id is set.",
    )
    apple_calendar_id: str = Field(
        default="",
        description="Event uid from mac_calendar_create_event; preferred over title.",
    )

def make_mac_calendar_create_tool(
    *,
    chroma_dir: Path | None = None,
    embeddings: GoogleGenerativeAIEmbeddings | None = None,
    memory_collection: str | None = None,
) -> StructuredTool:
    def _create(
        title: str,
        start_at: str,
        end_at: str,
        timezone: str = "Asia/Taipei",
        description: str = "",
        alert_minutes_before: int = 0,
    ) -> str:
        if not _mac_calendar_supported():
            return json.dumps(
                {"ok": False, "error": "Mac Calendar tools require macOS (Darwin)."},
                ensure_ascii=False,
            )
        t = (title or "").strip()
        sa = (start_at or "").strip()
        ea = (end_at or "").strip()
        tz_name = (timezone or "Asia/Taipei").strip() or "Asia/Taipei"
        desc = (description or "").strip()
        if not t:
            return json.dumps({"ok": False, "error": "title is required"}, ensure_ascii=False)
        if not sa or not ea:
            return json.dumps(
                {"ok": False, "error": "start_at and end_at are required (ISO-8601)."},
                ensure_ascii=False,
            )
        try:
            start_dt = _normalize_iso_datetime(sa, tz_name)
            end_dt = _normalize_iso_datetime(ea, tz_name)
        except Exception as exc:
            return json.dumps({"ok": False, "error": f"invalid datetime: {exc}"}, ensure_ascii=False)
        if end_dt <= start_dt:
            return json.dumps({"ok": False, "error": "end_at must be after start_at"}, ensure_ascii=False)
        try:
            apple_calendar_id = _create_mac_calendar_event(
                title=t,
                start_at=start_dt,
                end_at=end_dt,
                timezone_name=tz_name,
                description=desc,
                alert_minutes_before=alert_minutes_before,
            )
        except Exception as exc:
            return json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False)

        memory_item_id: str | None = None
        if chroma_dir is not None and embeddings is not None and (memory_collection or "").strip():
            mem_line = (
                "Apple Calendar event created | "
                f"apple_calendar_id={apple_calendar_id} | title={json.dumps(t, ensure_ascii=False)} | "
                f"start_at={start_dt.isoformat()} | end_at={end_dt.isoformat()}"
            )
            memory_item_id = _persist_memory_line(
                chroma_dir,
                embeddings,
                memory_collection.strip(),
                mem_line,
                "mac-calendar,apple-calendar",
            )

        return json.dumps(
            {
                "ok": True,
                "calendar": "mac",
                "title": t,
                "apple_calendar_id": apple_calendar_id,
                "saved_to_memory": memory_item_id is not None,
                "memory_item_id": memory_item_id or "",
            },
            ensure_ascii=False,
        )

    return StructuredTool.from_function(
        name="mac_calendar_create_event",
        description=(
            "Create an event in the macOS Calendar app (Apple Calendar / iCal). "
            "Runs on the Mac where the agent server executes; requires Calendar.app and Automation permission. "
            "Uses the first writable local calendar. On success, returns apple_calendar_id (Calendar uid) and "
            "saves that id plus title and times to persistent memory when the agent memory store is configured; "
            "tell the user apple_calendar_id from the JSON and that memory was updated when saved_to_memory is true."
        ),
        func=_create,
        args_schema=MacCalendarCreateEventArgs,
    )

def make_mac_calendar_update_tool() -> StructuredTool:
    def _update(
        match_title: str = "",
        apple_calendar_id: str = "",
        new_title: str = "",
        start_at: str = "",
        end_at: str = "",
        timezone: str = "Asia/Taipei",
        description: str = "",
        alert_minutes_before: int | None = None,
    ) -> str:
        if not _mac_calendar_supported():
            return json.dumps(
                {"ok": False, "error": "Mac Calendar tools require macOS (Darwin)."},
                ensure_ascii=False,
            )
        mt = (match_title or "").strip()
        aid = (apple_calendar_id or "").strip()
        tz_name = (timezone or "Asia/Taipei").strip() or "Asia/Taipei"
        if not mt and not aid:
            return json.dumps(
                {"ok": False, "error": "Provide match_title or apple_calendar_id."},
                ensure_ascii=False,
            )
        nt = (new_title or "").strip()
        sa = (start_at or "").strip()
        ea = (end_at or "").strip()
        desc_raw = (description or "").strip()

        start_dt: datetime | None = None
        end_dt: datetime | None = None
        if sa:
            try:
                start_dt = _normalize_iso_datetime(sa, tz_name)
            except Exception as exc:
                return json.dumps({"ok": False, "error": f"invalid start_at: {exc}"}, ensure_ascii=False)
        if ea:
            try:
                end_dt = _normalize_iso_datetime(ea, tz_name)
            except Exception as exc:
                return json.dumps({"ok": False, "error": f"invalid end_at: {exc}"}, ensure_ascii=False)
        if start_dt and end_dt and end_dt <= start_dt:
            return json.dumps({"ok": False, "error": "end_at must be after start_at"}, ensure_ascii=False)

        desc_param: str | None = None
        if desc_raw:
            desc_param = desc_raw

        if not nt and start_dt is None and end_dt is None and desc_param is None and alert_minutes_before is None:
            return json.dumps(
                {"ok": False, "error": "provide new_title, start_at, end_at, description, or alert_minutes_before to update"},
                ensure_ascii=False,
            )
        try:
            _update_mac_calendar_event(
                match_title=mt if not aid else "",
                match_uid=aid,
                new_title=nt or mt,
                start_at=start_dt,
                end_at=end_dt,
                timezone_name=tz_name,
                description=desc_param,
                alert_minutes_before=alert_minutes_before,
            )
        except Exception as exc:
            return json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False)
        out: dict[str, Any] = {"ok": True, "calendar": "mac", "match_title": mt}
        if aid:
            out["apple_calendar_id"] = aid
        return json.dumps(out, ensure_ascii=False)

    return StructuredTool.from_function(
        name="mac_calendar_update_event",
        description=(
            "Update an event in macOS Calendar. Prefer apple_calendar_id from mac_calendar_create_event; "
            "otherwise use exact match_title across all writable calendars. "
            "Pass new_title, start_at, end_at (ISO-8601), description, and/or alert_minutes_before."
        ),
        func=_update,
        args_schema=MacCalendarUpdateEventArgs,
    )

def make_mac_calendar_delete_tool(
    *,
    chroma_dir: Path | None = None,
    embeddings: GoogleGenerativeAIEmbeddings | None = None,
    memory_collection: str | None = None,
) -> StructuredTool:
    def _delete(title: str = "", apple_calendar_id: str = "") -> str:
        if not _mac_calendar_supported():
            return json.dumps(
                {"ok": False, "error": "Mac Calendar tools require macOS (Darwin)."},
                ensure_ascii=False,
            )
        t = (title or "").strip()
        aid = (apple_calendar_id or "").strip()
        if not t and not aid:
            return json.dumps(
                {"ok": False, "error": "Provide title or apple_calendar_id."},
                ensure_ascii=False,
            )
        try:
            _delete_mac_calendar_event(title=t if not aid else "", match_uid=aid)
        except Exception as exc:
            return json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False)

        deleted_from_memory = 0
        if chroma_dir is not None and embeddings is not None and (memory_collection or "").strip():
            deleted_from_memory = _delete_memory_entries_for_calendar_event(
                chroma_dir,
                embeddings,
                memory_collection.strip(),
                aid or t,
            )

        out: dict[str, Any] = {"ok": True, "calendar": "mac", "title": t}
        if aid:
            out["apple_calendar_id"] = aid
        out["memory_entries_removed"] = deleted_from_memory
        return json.dumps(out, ensure_ascii=False)

    return StructuredTool.from_function(
        name="mac_calendar_delete_event",
        description=(
            "Delete an event from macOS Calendar. Prefer apple_calendar_id from mac_calendar_create_event; "
            "otherwise exact title match on any writable calendar. Runs on the server Mac. "
            "Saves the deletion to memory if configured."
        ),
        func=_delete,
        args_schema=MacCalendarDeleteEventArgs,
    )
